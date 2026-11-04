import json
import numbers
import os
import re
import time
from pathlib import Path

import duckdb
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="UZEX AI-агент", page_icon="📊", layout="wide")

_BASE = Path(__file__).parent
DATA_DIR = _BASE / "data" if (_BASE / "data").exists() else _BASE
DEFAULT_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
NO_DATA = {"ru": "В данных ответа нет.", "uz": "Maʼlumotda javob yoʻq."}
FAIL = {
    "ru": "Не удалось получить ответ по данным. Попробуйте переформулировать вопрос.",
    "uz": "Maʼlumotlar asosida javob olib boʻlmadi. Savolni boshqacha ifodalab koʻring.",
}

SQL_RULES = """Ты переводишь вопрос о биржевых сделках (на русском или узбекском, латиница или кириллица) в один SQL-запрос DuckDB.
Правила:
1. Разрешён только один SELECT (можно с WITH). Никаких изменений данных.
2. Используй только таблицы и поля из схемы ниже. Не выдумывай поля, значения и справочные данные.
3. Все подсчёты (сумма, количество, среднее, доля, разница) делай в SQL. Дробные результаты округляй ROUND(..., 2).
4. Если вопрос нельзя ответить по этим данным (период вне диапазона дат из схемы, нужного поля нет, вопрос не про сделки биржи), верни sql = null и коротко объясни в reason.
5. Названия компаний, товаров, регионов ищи без учёта регистра через ILIKE '%...%'. Если сущности нет в данных, запрос вернёт пустой результат, это нормально.
6. Если вопрос не называет статус сделки, учитывай все статусы и укажи это в filters_note. Если называет («исполненные», «аннулированные»), фильтруй по соответствующему значению статуса из схемы.
7. Если валюта не уточнена, считай суммы в национальной валюте (сум). Если вопрос про доллары, используй поле в долларах (если есть) или курс из таблицы курсов. Всегда указывай в fields_used, какое поле суммы использовано и в какой валюте.
8. Для «топ-N», «наибольший», «eng koʻp» используй ORDER BY ... DESC и LIMIT; для равных значений добавь второй ключ сортировки. Возвращай и название, и значение.
9. В результат включай читаемые названия и коды (не только числовые ID): присоединяй справочники через JOIN.
10. Периоды: «в 2024 году» = полный год; «за III квартал» = три месяца. Сверяй период с диапазоном дат в схеме.
Словарь UZ: bitim = сделка, summa = сумма, miqdor = количество, mahsulot/tovar = товар, viloyat = регион, sotuvchi = продавец, xaridor = покупатель, shartnoma = контракт, kurs = курс, yil = год, oy = месяц, eng koʻp = наибольший, nechta/qancha = сколько.
Ответь строго JSON-объектом без markdown:
{"lang": "ru" или "uz", "sql": "запрос или null", "fields_used": "таблицы.поля и валюта", "filters_note": "какие фильтры и допущения применены", "reason": "если sql=null, то почему"}"""

ANSWER_SYS = """Ты формулируешь ответ на вопрос строго по результату SQL.
Правила: используй только числа, названия и коды из результата; ничего не считай и не округляй сам; не добавляй фактов из своих знаний.
Язык ответа: русский, если lang=ru; узбекский (латиница), если lang=uz.
Коротко: 1-3 предложения или нумерованный список. Числа копируй как есть, тысячи можно разделять пробелами.
Единицу (сум/USD) указывай только если она названа в fields_used.
В конце одной короткой строкой укажи применённые фильтры из filters_note.
Если total_rows больше числа показанных строк, скажи, что показаны первые строки."""

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|import|call|set|"
    r"read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|glob)\b",
    re.I,
)


def get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def get_models():
    m = get_secret("GEMINI_MODELS")
    if m:
        return [x.strip() for x in str(m).split(",") if x.strip()]
    return DEFAULT_MODELS


def safe_name(stem):
    return re.sub(r"\W+", "_", stem).strip("_").lower()


@st.cache_resource(show_spinner="Загрузка данных...")
def init_db():
    con = duckdb.connect(":memory:")
    if not DATA_DIR.exists():
        return con
    deals = sorted(DATA_DIR.glob("deals*.parquet"))
    if deals:
        pattern = (DATA_DIR / "deals*.parquet").as_posix()
        con.execute(
            f"CREATE TABLE deals AS SELECT * FROM read_parquet('{pattern}', union_by_name=true)"
        )
    existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    for p in sorted(DATA_DIR.glob("*.parquet")):
        if p in deals:
            continue
        name = safe_name(p.stem)
        if name in existing:
            continue
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{p.as_posix()}')")
        existing.add(name)
    for p in sorted(DATA_DIR.glob("*.csv")):
        name = safe_name(p.stem)
        if name in existing:
            continue
        con.execute(
            f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto('{p.as_posix()}', sample_size=-1)"
        )
        existing.add(name)
    return con


def describe_schema(con):
    parts = []
    num_prefix = ("BIGINT", "INTEGER", "DOUBLE", "HUGEINT", "DECIMAL", "FLOAT",
                  "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER")
    for (t,) in con.execute("SHOW TABLES").fetchall():
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        lines = [f"Таблица {t} ({n} строк). Колонки:"]
        for c in con.execute(f'DESCRIBE "{t}"').fetchall():
            name, typ = c[0], str(c[1]).upper()
            q = f'"{name}"'
            extra = ""
            try:
                if typ == "VARCHAR":
                    nd = con.execute(f'SELECT COUNT(DISTINCT {q}) FROM "{t}"').fetchone()[0]
                    if nd <= 40:
                        vals = [r[0] for r in con.execute(
                            f'SELECT DISTINCT {q} FROM "{t}" WHERE {q} IS NOT NULL ORDER BY 1'
                        ).fetchall()]
                        extra = f" | значения: {vals}"[:800]
                    else:
                        extra = f" | {nd} уникальных значений"
                elif typ.startswith(("DATE", "TIMESTAMP")) or typ.startswith(num_prefix):
                    mn, mx = con.execute(f'SELECT MIN({q}), MAX({q}) FROM "{t}"').fetchone()
                    extra = f" | диапазон: {mn} .. {mx}"
            except Exception:
                pass
            lines.append(f"  - {name} ({typ}){extra}")
        try:
            sample = con.execute(f'SELECT * FROM "{t}" LIMIT 3').fetchdf()
            lines.append("Пример строк:\n" + sample.to_string(max_colwidth=40))
        except Exception:
            pass
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def read_readme():
    if not DATA_DIR.exists():
        return ""
    for p in sorted(DATA_DIR.iterdir()):
        if p.name.lower().startswith("readme"):
            return p.read_text(encoding="utf-8", errors="replace")[:15000]
    return ""


@st.cache_resource
def get_context():
    con = init_db()
    return describe_schema(con), read_readme()


def get_client():
    return genai.Client(api_key=get_secret("GEMINI_API_KEY"))


def call_llm(system, prompt, json_mode=True):
    client = get_client()
    last = None
    for m in get_models():
        try:
            cfg = types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                response_mime_type="application/json" if json_mode else None,
            )
            r = client.models.generate_content(model=m, contents=prompt, config=cfg)
            if r.text:
                return r.text
        except Exception as e:
            last = e
            continue
    raise RuntimeError(f"Gemini недоступен: {last}")


def parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def check_sql(sql):
    s = sql.strip().rstrip(";").strip()
    bare = re.sub(r"'[^']*'", "''", s)
    if ";" in bare:
        raise ValueError("Допустим только один запрос")
    if not re.match(r"^(with|select)\b", bare, re.I):
        raise ValueError("Допустим только SELECT")
    if FORBIDDEN.search(bare):
        raise ValueError("Запрещённая конструкция в запросе")
    return s


def run_sql(con, sql):
    s = sql.strip().rstrip(";")
    cur = con.cursor()
    return cur.execute(f"SELECT * FROM ({s}) AS q LIMIT 1000").fetchdf()


def is_empty(df):
    if df.empty or df.isna().all().all():
        return True
    if len(df) == 1:
        vals = [v for v in df.iloc[0].tolist() if v is not None and v == v]
        if vals and all(
            isinstance(v, numbers.Number) and not isinstance(v, bool) and v == 0 for v in vals
        ):
            return True
    return False


def answer_question(question):
    con = init_db()
    schema, readme = get_context()
    sql_sys = SQL_RULES + "\n\nОПИСАНИЕ ПОЛЕЙ (README):\n" + readme + "\n\nСХЕМА ДАННЫХ:\n" + schema

    plan = parse_json(call_llm(sql_sys, f"Вопрос: {question}"))
    lang = "uz" if str(plan.get("lang", "ru")).lower().startswith("uz") else "ru"
    out = {"lang": lang, "plan": plan, "sql": None, "df": None, "answer": None, "error": None}

    sql = plan.get("sql")
    if not sql or str(sql).strip().lower() in ("null", "none"):
        out["answer"] = NO_DATA[lang]
        return out

    df, err = None, None
    for attempt in range(2):
        try:
            sql = check_sql(sql)
            df = run_sql(con, sql)
            err = None
            break
        except Exception as e:
            err = str(e)
            if attempt == 0:
                fix = parse_json(call_llm(
                    sql_sys,
                    f"Вопрос: {question}\nПрошлый SQL:\n{sql}\nОшибка: {err}\n"
                    "Исправь запрос и верни тот же JSON.",
                ))
                plan = {**plan, **{k: v for k, v in fix.items() if k != "lang"}}
                sql = fix.get("sql")
                if not sql:
                    break
    out["plan"] = plan
    out["sql"] = sql
    if df is None:
        out["answer"] = FAIL[lang]
        out["error"] = err
        return out

    out["df"] = df
    if is_empty(df):
        out["answer"] = NO_DATA[lang]
        return out

    shown = df.head(50)
    prompt = (
        f"Вопрос: {question}\nlang={lang}\n"
        f"fields_used: {plan.get('fields_used')}\nfilters_note: {plan.get('filters_note')}\n"
        f"total_rows={len(df)} (показано {len(shown)})\n"
        f"Результат SQL (CSV):\n{shown.to_csv(index=False)}"
    )
    try:
        out["answer"] = call_llm(ANSWER_SYS, prompt, json_mode=False).strip()
    except Exception:
        out["answer"] = "Результат запроса приведён в таблице ниже."
    return out


# ---------------- Интерфейс ----------------
st.title("📊 UZEX AI-агент")
st.caption(
    "Вопросы о сделках биржи на русском или oʻzbek tilida. "
    "Ответы строятся только по данным: SQL и строки-источники показаны под ответом."
)

con = init_db()
tables = con.execute("SHOW TABLES").fetchall()
if not tables:
    st.error("Данные не найдены. Проверьте, что файлы данных загружены в репозиторий.")
    st.stop()
if not get_secret("GEMINI_API_KEY"):
    st.error("Не задан ключ GEMINI_API_KEY (Settings → Secrets в Streamlit).")
    st.stop()

with st.sidebar:
    st.subheader("Данные")
    for (t,) in tables:
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        st.write(f"**{t}**: {n:,} строк".replace(",", " "))
    st.caption("README найден" if read_readme() else "README не найден")

q = st.chat_input("Задайте вопрос / Savol bering")
if q:
    st.chat_message("user").write(q)
    t0 = time.time()
    with st.chat_message("assistant"):
        with st.spinner("Анализирую данные..."):
            try:
                res = answer_question(q)
            except Exception as e:
                res = {"answer": "Сервис временно недоступен: " + str(e)[:300],
                       "sql": None, "df": None, "plan": {}}
        st.markdown(str(res["answer"]).replace("$", "\\$"))
        plan = res.get("plan") or {}
        if res.get("sql"):
            with st.expander("Использованный SQL"):
                st.code(res["sql"], language="sql")
        if plan.get("fields_used") or plan.get("filters_note"):
            with st.expander("Поля и фильтры"):
                st.write(f"**Поля:** {plan.get('fields_used')}")
                st.write(f"**Фильтры:** {plan.get('filters_note')}")
        if res.get("df") is not None and not res["df"].empty:
            with st.expander("Строки результата"):
                st.dataframe(res["df"].head(200))
        if res.get("error"):
            with st.expander("Техническая ошибка"):
                st.code(res["error"])
        st.caption(f"Время ответа: {time.time() - t0:.1f} с")
