import re
from pathlib import Path

import duckdb
import streamlit as st

st.set_page_config(page_title="Диагностика данных", layout="wide")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" if (BASE / "data").exists() else BASE

NORM = (
    "TRIM(regexp_replace(REPLACE(LOWER({c}), '-', ' '), "
    "'[[:space:]]+([0-9]+|[ivx]+|первый|второй|третий|высший)[[:space:]]*(сорт|класс).*$', ''))"
)
NUM_PREFIX = ("BIGINT", "INTEGER", "DOUBLE", "HUGEINT", "DECIMAL", "FLOAT",
              "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER")


def safe_name(stem):
    return re.sub(r"\W+", "_", stem).strip("_").lower()


@st.cache_resource(show_spinner="Загрузка данных...")
def load():
    con = duckdb.connect(":memory:")
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


def df_text(con, sql, transpose=False):
    df = con.execute(sql).fetchdf()
    if transpose:
        df = df.T
    return df.to_string(max_colwidth=70)


def run_report(con):
    out = []

    def add(title, text):
        out.append(f"=== {title} ===\n{text}\n")

    def safe(title, fn):
        try:
            add(title, fn())
        except Exception as e:
            add(title, f"ОШИБКА: {str(e)[:300]}")

    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    for tb in tables:
        n = con.execute(f'SELECT COUNT(*) FROM "{tb}"').fetchone()[0]
        cols = con.execute(f'DESCRIBE "{tb}"').fetchall()
        add(f"ТАБЛИЦА {tb}: строк {n}", "\n".join(f"{c[0]} | {c[1]}" for c in cols))
        safe(f"{tb}: 2 строки-примера", lambda tb=tb: df_text(
            con, f'SELECT * FROM "{tb}" LIMIT 2', transpose=True))

        def nulls(tb=tb, cols=cols, n=n):
            sel = ", ".join(f'COUNT("{c[0]}") AS "c{i}"' for i, c in enumerate(cols))
            row = con.execute(f'SELECT {sel} FROM "{tb}"').fetchone()
            lines = [f"{c[0]}: пустых {n - row[i]}" for i, c in enumerate(cols) if n - row[i] > 0]
            return "\n".join(lines) or "пустых значений нет"
        safe(f"{tb}: пустые значения (NULL)", nulls)

        def details(tb=tb, cols=cols, n=n):
            lines = []
            for c in cols:
                name, typ = c[0], str(c[1]).upper()
                q = f'"{name}"'
                if typ == "VARCHAR":
                    nd = con.execute(f'SELECT COUNT(DISTINCT {q}) FROM "{tb}"').fetchone()[0]
                    if nd <= 30:
                        vc = con.execute(
                            f'SELECT {q}, COUNT(*) FROM "{tb}" GROUP BY 1 ORDER BY 2 DESC'
                        ).fetchall()
                        lines.append(f"[{name}] значений {nd}: " + "; ".join(f"{v} ({k})" for v, k in vc))
                    else:
                        smp = con.execute(
                            f'SELECT v FROM (SELECT DISTINCT {q} AS v FROM "{tb}" '
                            f'WHERE {q} IS NOT NULL) ORDER BY hash(v) LIMIT 8'
                        ).fetchall()
                        lines.append(
                            f"[{name}] уникальных {nd}; примеры: "
                            + " || ".join(str(s[0])[:110] for s in smp)
                        )
                elif typ.startswith(("DATE", "TIMESTAMP")) or typ.startswith(NUM_PREFIX):
                    mn, mx = con.execute(f'SELECT MIN({q}), MAX({q}) FROM "{tb}"').fetchone()
                    lines.append(f"[{name}] {typ}: {mn} .. {mx}")
                    nd = con.execute(f'SELECT COUNT(DISTINCT {q}) FROM "{tb}"').fetchone()[0]
                    if nd == n and n > 0:
                        lines[-1] += "  <- УНИКАЛЬНО (ключ?)"
            return "\n".join(lines)
        safe(f"{tb}: значения колонок", details)

        for c in cols:
            name, typ = c[0], str(c[1]).upper()
            if typ == "VARCHAR" and re.search(r"product|name|товар|наимен|продукт", name, re.I):
                q = f'"{name}"'
                nx = NORM.format(c=q)

                def prod(tb=tb, q=q, nx=nx):
                    raw = con.execute(f'SELECT COUNT(DISTINCT {q}) FROM "{tb}"').fetchone()[0]
                    nrm = con.execute(f'SELECT COUNT(DISTINCT {nx}) FROM "{tb}"').fetchone()[0]
                    top = df_text(con, (
                        f'SELECT {nx} AS product_norm, COUNT(*) AS rows_cnt, '
                        f'COUNT(DISTINCT {q}) AS name_variants FROM "{tb}" '
                        f'GROUP BY 1 ORDER BY 2 DESC LIMIT 12'))
                    resid = con.execute(
                        f"SELECT COUNT(DISTINCT {nx}) FROM \"{tb}\" "
                        f"WHERE regexp_matches({nx}, '(сорт|класс|марка|размер|тип)')"
                    ).fetchone()[0]
                    rs = con.execute(
                        f"SELECT v FROM (SELECT DISTINCT {nx} AS v FROM \"{tb}\" "
                        f"WHERE regexp_matches({nx}, '(сорт|класс|марка|размер|тип)')) "
                        f"ORDER BY hash(v) LIMIT 8").fetchall()
                    return (f"уникальных исходных названий: {raw}\n"
                            f"после формулы приложения: {nrm}\n"
                            f"из них всё ещё содержат сорт/класс/марка/размер/тип: {resid}\n"
                            f"примеры таких: " + " || ".join(str(r[0])[:110] for r in rs)
                            + f"\nТОП-12 групп:\n{top}")
                safe(f"{tb}.{name}: проверка нормализации товаров", prod)
    return "\n".join(out)


st.title("Диагностика данных (временная страница)")
con = load()
if st.button("Собрать отчёт"):
    with st.spinner("Считаю..."):
        text = run_report(con)
    st.download_button("Скачать отчёт .txt", text.encode("utf-8"), "report.txt")
    st.code(text, language="text")
