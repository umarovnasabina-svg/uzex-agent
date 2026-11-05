import re
from pathlib import Path

import duckdb
import streamlit as st

st.set_page_config(page_title="Диагностика 2", layout="wide")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" if (BASE / "data").exists() else BASE
ANN = "Аннулирован(Не оплачен)"
NM = (
    "TRIM(regexp_replace(REPLACE(LOWER(product_name), '-', ' '), "
    "'[[:space:]]+([0-9]+|[ivx]+|первый|второй|третий|высший|высшего)[[:space:]]*(сорт|сорта|класс).*$', ''))"
)


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
    for p in sorted(DATA_DIR.glob("*.csv")):
        name = safe_name(p.stem)
        if name in existing:
            continue
        con.execute(
            f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto('{p.as_posix()}', sample_size=-1)"
        )
        existing.add(name)
    return con


QUERIES = [
    ("1. Статусы: число, сумма в сумах, пустые даты", f"""
        SELECT status, COUNT(*) AS n, CAST(SUM(amount_uzs) AS BIGINT) AS uzs,
               COUNT(*) FILTER (WHERE payment_date IS NULL) AS no_pay,
               COUNT(*) FILTER (WHERE delivery_date IS NULL) AS no_deliv
        FROM deals GROUP BY status ORDER BY n DESC"""),
    ("2. По годам: все сделки и без аннулированных", f"""
        SELECT EXTRACT(year FROM deal_date) AS y, COUNT(*) AS n_all,
               COUNT(*) FILTER (WHERE status <> '{ANN}') AS n_ok,
               CAST(SUM(amount_uzs) AS BIGINT) AS uzs_all,
               CAST(SUM(amount_uzs) FILTER (WHERE status <> '{ANN}') AS BIGINT) AS uzs_ok
        FROM deals GROUP BY 1 ORDER BY 1"""),
    ("3. Валюта: проверка amount_uzs", """
        SELECT currency, COUNT(*) AS n, CAST(SUM(deal_amount) AS BIGINT) AS sum_deal,
               CAST(SUM(amount_uzs) AS BIGINT) AS sum_uzs, CAST(SUM(amount_usd) AS BIGINT) AS sum_usd,
               MAX(ABS(amount_uzs - CASE WHEN currency = 'Доллар США'
                   THEN deal_amount * fx_rate_usd ELSE deal_amount END)) AS max_diff
        FROM deals GROUP BY currency"""),
    ("4. Курс в сделках против usd_rates", """
        SELECT COUNT(*) AS joined,
               COUNT(*) FILTER (WHERE ABS(d.fx_rate_usd - r."Курс") > 0.01) AS fx_mismatch
        FROM deals d JOIN usd_rates r ON CAST(d.deal_date AS DATE) = r."Дата" """),
    ("5. Объёмы: проверки volume и delivered_volume", """
        SELECT COUNT(*) FILTER (WHERE volume <> quantity * lot) AS vol_mismatch,
               COUNT(*) FILTER (WHERE delivered_volume > volume + 0.001) AS deliv_gt_vol,
               COUNT(*) FILTER (WHERE status = 'Исполнен' AND ABS(delivered_volume - volume) > 0.001) AS done_not_full,
               COUNT(*) FILTER (WHERE status = 'Исполнен(Частично)' AND delivered_volume >= volume) AS partial_but_full
        FROM deals"""),
    ("6. Дубликаты сделок (группы)", """
        SELECT COUNT(*) AS dup_groups FROM (
            SELECT 1 FROM deals
            GROUP BY contract_id, deal_date, deal_time, quantity, deal_amount, buyer_inn
            HAVING COUNT(*) > 1)"""),
    ("7. КОД ТН ВЭД -> какие названия товара под ним", f"""
        WITH n AS (SELECT hs_code, {NM} AS nm, COUNT(*) AS c FROM deals GROUP BY 1, 2)
        SELECT hs_code, CAST(SUM(c) AS BIGINT) AS deals, COUNT(*) AS names,
               string_agg(nm || ' (' || CAST(c AS VARCHAR) || ')', '; ' ORDER BY c DESC) AS names_list
        FROM n GROUP BY hs_code ORDER BY deals DESC"""),
    ("8. Названия, у которых больше одного кода ТН ВЭД", f"""
        WITH n AS (SELECT hs_code, {NM} AS nm FROM deals GROUP BY 1, 2)
        SELECT nm, string_agg(hs_code, ', ') AS codes FROM n GROUP BY nm HAVING COUNT(*) > 1"""),
    ("9. Коды ТН ВЭД с разными единицами измерения", """
        SELECT hs_code, string_agg(DISTINCT unit, ', ') AS units
        FROM deals GROUP BY hs_code HAVING COUNT(DISTINCT unit) > 1"""),
    ("10. Топ-8 по названию (формула приложения), 2025", f"""
        SELECT {NM} AS nm, COUNT(*) AS n, CAST(SUM(amount_uzs) AS BIGINT) AS uzs_all,
               CAST(SUM(amount_uzs) FILTER (WHERE status <> '{ANN}') AS BIGINT) AS uzs_ok
        FROM deals WHERE EXTRACT(year FROM deal_date) = 2025
        GROUP BY 1 ORDER BY uzs_all DESC LIMIT 8"""),
    ("11. Топ-8 по коду ТН ВЭД, 2025", f"""
        SELECT hs_code, COUNT(*) AS n, CAST(SUM(amount_uzs) AS BIGINT) AS uzs_all,
               CAST(SUM(amount_uzs) FILTER (WHERE status <> '{ANN}') AS BIGINT) AS uzs_ok
        FROM deals WHERE EXTRACT(year FROM deal_date) = 2025
        GROUP BY 1 ORDER BY uzs_all DESC LIMIT 8"""),
    ("12. Сорт и класс: в названии или в отдельной колонке", """
        SELECT COUNT(*) FILTER (WHERE LOWER(product_name) LIKE '%сорт%') AS sort_in_name,
               COUNT(*) FILTER (WHERE grade IS NOT NULL) AS grade_filled,
               COUNT(*) FILTER (WHERE LOWER(product_name) LIKE '%сорт%' AND grade IS NOT NULL) AS both_,
               COUNT(*) FILTER (WHERE LOWER(product_name) LIKE '%класс%') AS class_in_name,
               COUNT(*) FILTER (WHERE product_class IS NOT NULL) AS class_filled
        FROM deals"""),
    ("13. Хлопок-сырец: варианты названий", """
        SELECT product_name, COUNT(*) AS n FROM deals
        WHERE LOWER(product_name) LIKE '%хлопок%сырец%'
        GROUP BY 1 ORDER BY 2 DESC LIMIT 15"""),
]


def run_report(con):
    out = []
    for title, sql in QUERIES:
        try:
            df = con.execute(sql).fetchdf()
            text = df.to_string(index=False, max_colwidth=700)
        except Exception as e:
            text = f"ОШИБКА: {str(e)[:300]}"
        out.append(f"=== {title} ===\n{text}\n")
    return "\n".join(out)


st.title("Диагностика 2 (временная страница)")
con = load()
if st.button("Собрать отчёт 2"):
    with st.spinner("Считаю..."):
        text = run_report(con)
    st.download_button("Скачать отчёт .txt", text.encode("utf-8"), "report2.txt")
    st.code(text, language="text")
