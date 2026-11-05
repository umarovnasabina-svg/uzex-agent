import datetime
import json
import numbers
import os
import re
import threading
import time
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="UZEX AI-агент", page_icon="📊", layout="wide")

_BASE = Path(__file__).parent
DATA_DIR = _BASE / "data" if (_BASE / "data").exists() else _BASE
DEFAULT_MODELS = ["gemini-3.5-flash-lite"]

NO_DATA = {"ru": "В данных ответа нет.", "uz": "Maʼlumotda javob yoʻq."}
FAIL = {
    "ru": "Не удалось получить ответ по данным. Попробуйте переформулировать вопрос.",
    "uz": "Maʼlumotlar asosida javob olib boʻlmadi. Savolni boshqacha ifodalab koʻring.",
}

# ---------------- Тексты интерфейса ----------------
TR = {
    "ru": {
        "title": "UZEX AI-агент",
        "subtitle": "Вопросы о биржевых сделках на русском или oʻzbek tilida. Ответ строится только по данным: если данных нет, агент так и скажет.",
        "m_deals": "Сделок в базе",
        "m_period": "Период данных",
        "m_tables": "Таблиц",
        "welcome": "Выберите пример или напишите свой вопрос внизу. Можно уточнять: «а за 2022?».",
        "ex_ru": "Примеры (RU)",
        "ex_uz": "Примеры (UZ)",
        "clear": "🗑 Очистить чат",
        "table": "Таблица",
        "chart": "График",
        "sql": "Показать SQL",
        "csv": "📥 CSV",
        "fields": "Поля",
        "filters": "Фильтры",
        "source": "Источник: таблица сделок — {n} записей, период {mn} – {mx}. Строк в результате: {k}.",
        "thanks": "Спасибо за оценку",
        "spinner": "Анализирую данные...",
        "busy": "Сервис ИИ сейчас перегружен или достиг лимита запросов. Подождите минуту и отправьте вопрос ещё раз.",
        "tech": "Техническая ошибка",
        "time": "Время ответа",
        "side_data": "Данные",
        "ui_lang": "Язык интерфейса",
        "no_rows": "По заданным условиям записей не найдено.",
        "period_avail": "Доступный период данных",
        "guard": "Ответ показан без пересказа, чтобы исключить расхождение с таблицей.",
        "rows_word": "строк",
        "result_word": "Результат",
        "see_table": "Результат приведён в таблице ниже.",
        "no_data_err": "Данные не найдены. Проверьте, что файлы данных загружены в репозиторий.",
        "no_key": "Не задан ключ GEMINI_API_KEY (Settings → Secrets в Streamlit).",
    },
    "uz": {
        "title": "UZEX AI-agent",
        "subtitle": "Birja bitimlari haqida rus yoki oʻzbek tilida savol bering. Javob faqat maʼlumotlarga asoslanadi: maʼlumot boʻlmasa, agent shuni aytadi.",
        "m_deals": "Bitimlar soni",
        "m_period": "Maʼlumotlar davri",
        "m_tables": "Jadvallar",
        "welcome": "Namunani tanlang yoki pastda oʻz savolingizni yozing. Savolni aniqlashtirish mumkin: «2022-yilda-chi?».",
        "ex_ru": "Namunalar (RU)",
        "ex_uz": "Namunalar (UZ)",
        "clear": "🗑 Chatni tozalash",
        "table": "Jadval",
        "chart": "Grafik",
        "sql": "SQL koʻrsatish",
        "csv": "📥 CSV",
        "fields": "Maydonlar",
        "filters": "Filtrlar",
        "source": "Manba: bitimlar jadvali — {n} ta yozuv, davr {mn} – {mx}. Natijadagi qatorlar: {k}.",
        "thanks": "Bahoyingiz uchun rahmat",
        "spinner": "Maʼlumotlar tahlil qilinmoqda...",
        "busy": "AI xizmati hozir band yoki soʻrovlar limitiga yetdi. Bir daqiqa kuting va savolni qayta yuboring.",
        "tech": "Texnik xato",
        "time": "Javob vaqti",
        "side_data": "Maʼlumotlar",
        "ui_lang": "Interfeys tili",
        "no_rows": "Berilgan shartlar boʻyicha yozuvlar topilmadi.",
        "period_avail": "Maʼlumotlarning mavjud davri",
        "guard": "Javob jadvaldagi raqamlar bilan farq qilmasligi uchun qayta bayon qilinmadi.",
        "rows_word": "qator",
        "result_word": "Natija",
        "see_table": "Natija quyidagi jadvalda keltirilgan.",
        "no_data_err": "Maʼlumotlar topilmadi. Maʼlumot fayllari repozitoriyga yuklanganini tekshiring.",
        "no_key": "GEMINI_API_KEY kaliti kiritilmagan (Streamlit: Settings → Secrets).",
    },
}

EXAMPLES = {
    "ru": [
        "Сколько всего сделок в данных?",
        "Сколько сделок было в 2024 году?",
        "Какой оборот в сумах был с 2022 по 2024 год?",
        "Какой товар был лидером по сумме сделок в 2025 году?",
        "Сколько сделок было в 2021 году?",
    ],
    "uz": [
        "Jami nechta bitim bor?",
        "2024-yilda nechta bitim boʻlgan?",
        "2022-yildan 2024-yilgacha aylanma qancha soʻm?",
        "2025-yilda summa boʻyicha eng koʻp sotilgan tovar qaysi?",
        "2021-yilda nechta bitim boʻlgan?",
    ],
}

# Колонки в deals английские, а README описывает их русскими названиями.
DEALS_NOTES = (
    "ВАЖНО: в таблице deals колонки английские. Соответствие названиям из README: "
    "Номер сделки = deal_id; Дата сделки = deal_date; Время сделки = deal_time; Тип торгов = trade_type; "
    "Количество (лотов) = quantity; Сумма сделки в валюте контракта = deal_amount (НЕ использовать, "
    "суммы только из amount_uzs и amount_usd); Стартовая цена = start_price; Номер контракта = contract_id; "
    "Наименование товара = product_name (с атрибутами одной строкой: product_full_name); Лот = lot; "
    "Единица измерения = unit; Регион = region_ru (по-русски) и region_uz (латиницей); "
    "Тип контракта = contract_type; Форма контракта = contract_form; Состояние сделки = status; "
    "ИНН продавца и покупателя = seller_inn и buyer_inn; Брокер продавца и покупателя = "
    "seller_broker_name и buyer_broker_name; Дата оплаты = payment_date; Дата поставки = delivery_date; "
    "Поставлено лотов = delivered_lots; объём в единицах измерения = volume (= quantity * lot); "
    "фактически поставленный объём = delivered_volume. Таблицы contracts, participants, brokers, usd_rates "
    "модели недоступны: всё нужное уже есть в deals.\n\n"
)

PRODUCT_RULES_NEW = """15. ТОВАРЫ: в deals есть колонка product_norm, единое название товара (один товар = одно значение; список значений есть в схеме). Для фильтров, группировок, топов и рейтингов по товару используй ТОЛЬКО product_norm. Если запрошенный товар однозначно совпадает со значением из списка схемы, используй равенство product_norm = '...'; иначе product_norm ILIKE '%слово%' (слово в начальной форме: «хлопка» → «хлопок»). Не добавляй своих синонимов и переводов, кроме словаря UZ ниже. В рейтингах возвращай product_norm и метрику. Спот, форвард и фьючерс различаются колонкой contract_form: product_norm «хлопок сырец» включает все три формы; если в вопросе есть «спот», «форвард» или «фьючерс», добавь фильтр по contract_form. В filters_note кратко укажи, по какому товару искали.
16. СОРТ И КЛАСС записаны по-разному, поэтому ищи по всем вариантам, без учёта регистра: LOWER(grade) IN (...). 1 сорт: '1 сорт','1-сорт','1-сорта','i сорт','первый сорт'. 2 сорт: '2 сорт','2-сорта','ii сорт','второй сорт'. 3 сорт: '3 сорт','3-сорта','третий сорт'. Высший сорт: 'высший сорт','высшего сорта','в/с'. 1 класс: '1 класс','1-класс','i класс'. 2 класс: '2 класс','2-класс','ii класс'. 3 класс: '3 класс','3-класс','iii класс'. 4 класс: '4 класс','4-класс','iv класс'. Для product_class так же (у арматуры отдельное значение 'а500с'). У части сделок сорт или класс указан только в product_name (например «... ПЕРВЫЙ СОРТ II КЛАСС»), а колонка пуста, поэтому для хлопка добавляй и условие по product_name; у большинства сделок сорта нет вообще, скажи об этом в filters_note.
"""

PRODUCT_RULES_OLD = """15. ПОИСК ПО НАЗВАНИЮ: ищи только слова из вопроса (в начальной форме: «хлопка» → «хлопок»). Не добавляй свои синонимы, переводы и транслитерации (например «пахта»), если их нет в данных. В данных один товар записан по-разному (регистр, дефис или пробел, суффиксы сорта и класса), поэтому ищи так: REPLACE(LOWER(product_name), '-', ' ') ILIKE '%слово или фраза%', где фразу из вопроса тоже приведи к нижнему регистру и замени дефис на пробел. Поиск по подстроке не захватывает однокоренные, но другие товары (масло хлопковое, шрот, пряжа не содержат «хлопок»). В filters_note кратко укажи, по какому слову искали.
16. РЕЙТИНГИ И СУММЫ ПО ТОВАРАМ: один и тот же товар записан в данных по-разному (регистр, дефис или пробел, сорт и класс внутри названия), поэтому для топов, рейтингов и группировок по товару группируй по нормализованному названию:
TRIM(regexp_replace(REPLACE(LOWER(product_name), '-', ' '), '[[:space:]]+([0-9]+|[ivx]+|первый|второй|третий|высший)[[:space:]]*(сорт|класс).*$', '')) AS product_norm
Возвращай product_norm, нужную метрику и COUNT(DISTINCT product_name) AS name_variants; сортируй и группируй по product_norm. В filters_note напиши, что названия товаров объединены (регистр, дефис, сорт и класс). Эту формулу копируй дословно, не меняя регулярное выражение, и не используй обратные косые черты.
"""

SQL_RULES = """Ты переводишь вопрос о биржевых сделках (на русском или узбекском, латиница или кириллица) в один SQL-запрос DuckDB.
Правила:
1. Разрешён только один SELECT (можно с WITH). Никаких изменений данных, обращений к файлам и комментариев в SQL.
2. Используй только таблицу deals и поля из схемы ниже. Не выдумывай поля, значения и справочные данные.
3. Все подсчёты (сумма, количество, среднее, доля, разница) делай в SQL. Деньги считай ТОЧНО: сумма = ROUND(SUM(CAST(поле AS DECIMAL(38,6))), 2); среднее = ROUND(AVG(CAST(поле AS DECIMAL(38,6))), 2). Округляй только итог, а не отдельные сделки. Никогда не приводи деньги к DOUBLE/FLOAT. Проценты и доли: ROUND(100.0 * a / b, 2).
4. Если вопрос нельзя ответить по этим данным (период вне диапазона дат из схемы, нужного поля нет, вопрос не про сделки биржи), верни sql = null и коротко объясни в reason.
5. Названия компаний ищи без учёта регистра через ILIKE '%...%' по seller_name или buyer_name (имена записаны латиницей и кириллицей; для точной группировки используй seller_inn или buyer_inn). Регионы ищи по правилу 17, товары по правилу 15. Не подменяй запрошенное название похожим: если сущности нет в данных, запрос должен вернуть пустой результат, это нормально. Опечатки и разное написание: если в схеме есть список значений и подходящее значение однозначно, используй его; иначе ILIKE по основной части слова, и скажи об этом в filters_note.
6. СТАТУС. Аннулированные сделки (status = 'Аннулирован(Не оплачен)') не состоялись. Для вопросов про сумму, оборот, объём, цену и количество сделок исключай их (status <> 'Аннулирован(Не оплачен)') и пиши это в filters_note; ОБЯЗАТЕЛЬНЫЙ ШАБЛОН: SQL всегда возвращает две колонки, основную без аннулированных и вторую по всем статусам с суффиксом _all. Количество сделок: deals_count = COUNT(*) FILTER (WHERE status <> 'Аннулирован(Не оплачен)') и deals_count_all = COUNT(*). Сумма: total_uzs = SUM(...) FILTER (WHERE status <> 'Аннулирован(Не оплачен)') и total_uzs_all = SUM(...) по всем строкам (с тем же приведением к DECIMAL и округлением). Объём и цена: так же две колонки, объём по каждой unit. В рейтингах сортируй по основной колонке. В filters_note напиши: аннулированные сделки исключены, в колонке _all учтены все статусы. Исключение: вопрос «сколько всего сделок/записей в данных» считай по всем строкам. Если статус назван: «исполненные» = 'Исполнен' (частично исполненные это отдельный статус 'Исполнен(Частично)'), «оплаченные» = 'Оплачен', «аннулированные» = 'Аннулирован(Не оплачен)'. Статус 'Новый' означает только сделки 30.10.2026.
7. ДЕНЬГИ только из amount_uzs (сум) и amount_usd (доллар США). Поля deal_amount, start_price, unit_price, base_price указаны в валюте контракта (у экспортных сделок это доллары), поэтому суммировать и усреднять их НЕЛЬЗЯ. Без уточнения валюты используй amount_uzs. Цена за единицу = SUM(amount_uzs) / SUM(volume), отдельно по каждой unit; AVG(unit_price) запрещён. Всегда указывай в fields_used, какое поле суммы использовано и в какой валюте.
8. Для «топ-N», «наибольший», «eng koʻp» используй ORDER BY ... DESC и LIMIT; для равных значений добавь второй ключ сортировки. Возвращай и название, и значение.
9. JOIN с другими таблицами запрещён. В deals уже есть названия товаров, продавцов, покупателей, брокеров, регионы, код ТН ВЭД (hs_code), курс и суммы. Имена бери из seller_name, buyer_name, seller_broker_name, buyer_broker_name. Колонкам результата давай понятные имена (total_uzs, deals_count).
10. Периоды: «в 2024 году» = полный год; «за III квартал» = три месяца. Период сделки = deal_date, оплаты = payment_date, поставки = delivery_date. Сверяй период с диапазоном дат в схеме. Данные за 2026 год только до 30.10.2026, скажи об этом в filters_note, если вопрос затрагивает 2026 год. Если часть периода вне диапазона, посчитай только доступную часть и скажи об этом в filters_note.
11. «оборот», «aylanma» = сумма сумм сделок (SUM) за период. Слово «сумма» без уточнения = сумма сделок в сумах; если неоднозначно, укажи это допущение в filters_note.
12. ИСТОРИЯ ДИАЛОГА (если дана): если новый вопрос — уточнение («а за 2022?», «а по Ташкенту?», «2022-yilda-chi?»), возьми условия из предыдущего вопроса и измени только названное. Если вопрос самостоятельный, игнорируй историю.
13. fields_used, filters_note и reason пиши на языке вопроса (русский или узбекский латиницей).
14. ОБЪЁМ: поле volume (= quantity × lot), единица измерения в поле unit (килограмм, тонна, метр и т.д.). Объёмы с разными единицами НЕЛЬЗЯ складывать: при подсчёте объёма всегда делай GROUP BY unit и возвращай колонку unit; одна строка на единицу. Объём volume целое число: суммируй SUM(volume) без приведения к DECIMAL. Фактически поставленный объём = delivered_volume (дробное, суммируй как деньги: ROUND(SUM(CAST(delivered_volume AS DECIMAL(38,6))), 2)). Вопросы «сколько продано/куплено» без уточнения = volume (и добавь количество сделок и сумму в сумах).
@@PRODUCT_RULES@@17. РЕГИОНЫ: «регион» без уточнения роли = регион склада продавца, колонка region_ru; фильтруй точным равенством, без ILIKE. «Ташкент» без уточнения = 'г. Ташкент' (скажи об этом в filters_note); область = 'Ташкентская область'. Для региона покупателя используй buyer_region и значение точно из списка схемы. Соответствие: Toshkent shahri = г. Ташкент, Toshkent viloyati = Ташкентская область, Namangan = Наманганская область, Buxoro = Бухарская область, Samarqand = Самаркандская область, Sirdaryo = Сырдарьинская область, Qashqadaryo = Кашкадарьинская область, Xorazm = Хорезмская область, Andijon = Андижанская область, Farg‘ona = Ферганская область, Surxondaryo = Сурхандарьинская область, Navoiy = Навоийская область, Qoraqalpog‘iston = Республика Каракалпакстан, Jizzax = Джизакская область.
Словарь UZ: bitim = сделка, summa = сумма, miqdor = количество, mahsulot/tovar = товар, viloyat = регион, sotuvchi = продавец, xaridor = покупатель, shartnoma = контракт, kurs = курс, yil = год, oy = месяц, eng koʻp = наибольший, nechta/qancha = сколько, aylanma = оборот, jami = всего.
Названия товаров на узбекском (других переводов не придумывай): paxta = хлопок, paxta yogʻi = масло хлопковое, paxta ipi = пряжа хлопчатобумажная, bugʻdoy = пшеница, un = мука, shakar = сахар, sement = цемент, benzin = бензин, dizel = дизельное топливо, mazut = мазут, gaz = газ, guruch = рис, kungaboqar yogʻi = масло подсолнечное, karbamid/mochevina = карбамид, selitra = аммиачная селитра, gʻisht = кирпич, shisha = стекло, armatura = арматура, bitum = битум, polietilen = полиэтилен, tuz = соль, ohak = известь, kabel = кабель, suv = вода.
Ответь строго JSON-объектом без markdown:
{"lang": "ru" или "uz", "full_question": "самостоятельная формулировка вопроса с учётом истории", "sql": "запрос или null", "fields_used": "таблицы.поля и валюта", "filters_note": "какие фильтры и допущения применены", "reason": "если sql=null, то почему"}"""

ANSWER_SYS = """Ты формулируешь ответ на вопрос строго по результату SQL.
Правила: используй только числа, названия и коды из результата; ничего не считай, не округляй и не сокращай сам (не пиши «млрд», «трлн»); не добавляй фактов из своих знаний.
Числа в результате уже отформатированы (пробелы между тысячами, запятая в дробной части): копируй их дословно.
Язык ответа: русский, если lang=ru; узбекский (латиница), если lang=uz.
Коротко: 1-3 предложения или нумерованный список. Единицу (сум/USD) указывай только если она названа в fields_used.
Денежную единицу пиши «сум» (не «сумов»). Сумму сделок не называй «объёмом»: объём это только количество в единицах измерения (колонка volume).
Если в результате есть колонка unit, ВСЕГДА называй единицу измерения рядом с объёмом (например «1 436 632 219 килограмм»). Если строки с разными единицами, называй каждую отдельно и никогда не складывай их.
В конце одной короткой строкой укажи применённые фильтры из filters_note (на языке ответа).
Если в результате есть колонки с суффиксом _all, назови основное значение, а значение из колонки _all добавь в скобках словами «по всем статусам, включая аннулированные».
Если total_rows больше числа показанных строк, скажи, что показаны первые строки."""

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|import|call|set|"
    r"read_csv\w*|read_parquet|read_json\w*|read_ndjson\w*|read_text|read_blob|parquet_\w+|sniff_csv|"
    r"duckdb_\w+|getenv|current_setting|glob)\b",
    re.I,
)
OTHER_TABLES = re.compile(r"\b(contracts|participants|brokers|usd_rates)\b", re.I)

ID_LIKE = re.compile(
    r"(^|_)(id|code|kod|year|yil|god|month|oy|quarter|inn|stir|tin|y|год|месяц|квартал)($|_)|id$", re.I
)
GREEDY_NUM = re.compile(r"\d[\d \u00a0]*(?:[.,]\d+)?")


# ---------------- Утилиты ----------------
def ui_lang():
    v = st.session_state.get("ui_lang_choice", "Русский")
    return "uz" if str(v).startswith("O") else "ru"


def t(key, lang=None):
    return TR[lang or ui_lang()][key]


def guess_lang(text):
    return "ru" if re.search(r"[А-Яа-яЁё]", text or "") else "uz"


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


def fmt_value(v, name=""):
    if v is None:
        return ""
    try:
        if v != v:
            return ""
    except Exception:
        pass
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (datetime.datetime, pd.Timestamp)):
        if (v.hour, v.minute, v.second) == (0, 0, 0):
            return v.strftime("%Y-%m-%d")
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, datetime.date):
        return v.isoformat()
    if isinstance(v, numbers.Number):
        if ID_LIKE.search(str(name)):
            try:
                return str(int(v)) if float(v).is_integer() else str(v)
            except Exception:
                return str(v)
        if isinstance(v, Decimal):
            d = v
        elif isinstance(v, float):
            d = Decimal(int(v)) if v.is_integer() else Decimal(str(round(v, 2)))
        else:
            d = Decimal(int(v))
        s = format(d, ",f")
        return s.replace(",", " ").replace(".", ",")
    return str(v)


def to_display_df(df):
    cols = list(df.columns)
    rows = [
        [fmt_value(v, cols[j]) for j, v in enumerate(r)]
        for r in df.itertuples(index=False, name=None)
    ]
    return pd.DataFrame(rows, columns=cols)


def build_chart(df):
    if df is None or df.shape[1] < 2 or len(df) < 2 or len(df) > 60:
        return None
    cols = list(df.columns)

    def numeric(j):
        vals = [v for v in df.iloc[:, j] if v is not None]
        return bool(vals) and all(
            isinstance(v, numbers.Number) and not isinstance(v, bool) for v in vals
        )

    cat = next((j for j in range(len(cols)) if (not numeric(j)) or ID_LIKE.search(str(cols[j]))), None)
    if cat is None:
        return None
    val = next(
        (j for j in range(len(cols)) if j != cat and numeric(j) and not ID_LIKE.search(str(cols[j]))),
        None,
    )
    if val is None:
        return None
    try:
        labels = [fmt_value(v, cols[cat]) for v in df.iloc[:, cat]]
        values = [float(v) for v in df.iloc[:, val]]
    except Exception:
        return None
    cdf = pd.DataFrame({str(cols[val]): values}, index=labels)
    is_time = all(re.match(r"^\d{4}(-\d{2}){0,2}$", lb) for lb in labels)
    return cdf, is_time


def num_tokens(text, include_runs=True):
    out = set()
    for m in GREEDY_NUM.finditer(text or ""):
        out.add(re.sub(r"\D", "", m.group(0)))
    if include_runs:
        for m in re.finditer(r"\d+", text or ""):
            out.add(m.group(0))
    return out


def numbers_ok(answer, allowed):
    for m in GREEDY_NUM.finditer(answer):
        d = re.sub(r"\D", "", m.group(0))
        if len(d) < 3 or d in allowed:
            continue
        parts = m.group(0).split()
        if len(parts) > 1 and all(
            len(re.sub(r"\D", "", p)) < 3 or re.sub(r"\D", "", p) in allowed for p in parts
        ):
            continue
        return False
    return True


def det_answer(fdf, lang):
    if len(fdf) == 1:
        return T_det(lang) + ": " + "; ".join(f"{c}: {v}" for c, v in zip(fdf.columns, fdf.iloc[0]))
    return TR[lang]["see_table"]


def T_det(lang):
    return TR[lang]["result_word"]


# ---------------- Данные ----------------
DB_STATUS = {"product_norm": "не запускалось", "lock": "не запускалось"}


def build_product_norm(con):
    """Добавляет в deals колонку product_norm: один товар = одно название (по коду ТН ВЭД).
    Хлопок (спот, форвард, фьючерс) объединён; код 271019 разделён на дизельное топливо и мазут.
    Если что-то пошло не так, deals остаётся как был."""
    try:
        cols = {r[0] for r in con.execute("DESCRIBE deals").fetchall()}
        if "product_norm" in cols:
            DB_STATUS["product_norm"] = "уже есть в данных"
            return
        if "product_name" not in cols or "hs_code" not in cols:
            DB_STATUS["product_norm"] = "нет колонок product_name или hs_code: " + ", ".join(sorted(cols))[:200]
            return
        nm = (
            "TRIM(regexp_replace(REPLACE(LOWER(product_name), '-', ' '), "
            "'[[:space:]]+([0-9]+|[ivx]+|первый|второй|третий|высший|высшего)[[:space:]]*(сорт|сорта|класс).*$', ''))"
        )
        con.execute(
            f"""
            CREATE TABLE deals_new AS
            WITH b AS (SELECT *, {nm} AS nm FROM deals),
            c AS (
                SELECT hs_code, arg_max(nm, cnt) AS canon FROM (
                    SELECT hs_code, nm, COUNT(*) AS cnt FROM b GROUP BY 1, 2
                ) GROUP BY hs_code
            )
            SELECT b.* EXCLUDE (nm),
                   CASE WHEN CAST(b.hs_code AS VARCHAR) = '520100' THEN 'хлопок сырец'
                        WHEN CAST(b.hs_code AS VARCHAR) = '271019' AND b.nm LIKE '%мазут%' THEN 'мазут'
                        WHEN CAST(b.hs_code AS VARCHAR) = '271019' THEN 'дизельное топливо'
                        ELSE COALESCE(c.canon, b.nm) END AS product_norm
            FROM b LEFT JOIN c ON b.hs_code IS NOT DISTINCT FROM c.hs_code
            """
        )
        con.execute("DROP TABLE deals")
        con.execute("ALTER TABLE deals_new RENAME TO deals")
        k = con.execute("SELECT COUNT(DISTINCT product_norm) FROM deals").fetchone()[0]
        DB_STATUS["product_norm"] = f"OK, значений: {k}"
    except Exception as e:
        DB_STATUS["product_norm"] = "ОШИБКА: " + str(e)[:300]
        try:
            con.execute("DROP TABLE IF EXISTS deals_new")
        except Exception:
            pass


def has_product_norm(con):
    try:
        return "product_norm" in {r[0] for r in con.execute("DESCRIBE deals").fetchall()}
    except Exception:
        return False


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
    build_product_norm(con)
    # После загрузки закрываем доступ к файлам: SQL от модели не сможет читать их с диска.
    try:
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        DB_STATUS["lock"] = "OK"
    except Exception as e:
        DB_STATUS["lock"] = "ОШИБКА: " + str(e)[:200]
    return con


def describe_schema(con):
    parts = []
    num_prefix = ("BIGINT", "INTEGER", "DOUBLE", "HUGEINT", "DECIMAL", "FLOAT",
                  "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER")
    for (tb,) in con.execute("SHOW TABLES").fetchall():
        if tb != "deals":
            continue
        n = con.execute(f'SELECT COUNT(*) FROM "{tb}"').fetchone()[0]
        lines = [f"Таблица {tb} ({n} строк). Колонки:"]
        for c in con.execute(f'DESCRIBE "{tb}"').fetchall():
            name, typ = c[0], str(c[1]).upper()
            q = f'"{name}"'
            extra = ""
            try:
                if typ == "VARCHAR":
                    nd = con.execute(f'SELECT COUNT(DISTINCT {q}) FROM "{tb}"').fetchone()[0]
                    if nd <= 60:
                        vals = [r[0] for r in con.execute(
                            f'SELECT DISTINCT {q} FROM "{tb}" WHERE {q} IS NOT NULL ORDER BY 1'
                        ).fetchall()]
                        extra = f" | значения: {vals}"[:2000]
                    else:
                        extra = f" | {nd} уникальных значений"
                elif typ.startswith(("DATE", "TIMESTAMP")) or typ.startswith(num_prefix):
                    mn, mx = con.execute(f'SELECT MIN({q}), MAX({q}) FROM "{tb}"').fetchone()
                    extra = f" | диапазон: {mn} .. {mx}"
            except Exception:
                pass
            lines.append(f"  - {name} ({typ}){extra}")
        try:
            sample = con.execute(f'SELECT * FROM "{tb}" LIMIT 2').fetchdf()
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
    return describe_schema(con), DEALS_NOTES + read_readme()


@st.cache_resource
def get_meta():
    con = init_db()
    meta = {"n": None, "mn": None, "mx": None}
    try:
        n, mn, mx = con.execute("SELECT COUNT(*), MIN(deal_date), MAX(deal_date) FROM deals").fetchone()
        meta = {"n": n, "mn": fmt_value(mn), "mx": fmt_value(mx)}
    except Exception:
        pass
    return meta


# ---------------- Gemini ----------------
class LLMBusy(Exception):
    pass


def get_client():
    return genai.Client(api_key=get_secret("GEMINI_API_KEY"))


def call_llm(system, prompt, json_mode=True):
    client = get_client()
    last = None
    for m in get_models():
        for attempt in range(2):
            try:
                cfg = types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    response_mime_type="application/json" if json_mode else None,
                )
                r = client.models.generate_content(model=m, contents=prompt, config=cfg)
                if r.text:
                    return r.text
                last = "empty response"
                break
            except Exception as e:
                last = e
                msg = str(e)
                transient = any(
                    k in msg
                    for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL",
                              "DEADLINE", "timed out", "timeout")
                )
                if attempt == 0 and transient and "PerDay" not in msg:
                    time.sleep(3)
                    continue
                break
    raise LLMBusy(str(last)[:300])


def parse_json(text):
    tx = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        res = json.loads(tx)
    except Exception:
        m = re.search(r"\{.*\}", tx, re.S)
        if not m:
            raise
        res = json.loads(m.group(0))
    if isinstance(res, list) and res and isinstance(res[0], dict):
        res = res[0]
    if not isinstance(res, dict):
        raise ValueError("Ответ модели не в формате JSON-объекта")
    return res


def check_sql(sql):
    s = sql.strip().rstrip(";").strip()
    bare = re.sub(r"'[^']*'", "''", s)
    if ";" in bare:
        raise ValueError("Допустим только один запрос")
    if "--" in bare or "/*" in bare:
        raise ValueError("Комментарии запрещены")
    if not re.match(r"^(with|select)\b", bare, re.I):
        raise ValueError("Допустим только SELECT")
    if FORBIDDEN.search(bare):
        raise ValueError("Запрещённая конструкция в запросе")
    if re.search(r"\b(from|join)\s*''", bare, re.I):
        raise ValueError("Чтение файлов запрещено")
    if OTHER_TABLES.search(bare):
        raise ValueError("Другие таблицы недоступны: используйте только deals")
    return s


def run_sql(con, sql, timeout=25):
    s = sql.strip().rstrip(";")
    cur = con.cursor()
    timer = threading.Timer(timeout, lambda: cur.interrupt())
    timer.start()
    try:
        res = cur.execute(f"SELECT * FROM ({s}) AS q LIMIT 1000")
        cols = [d[0] for d in res.description]
        rows = res.fetchall()
    finally:
        timer.cancel()
    return pd.DataFrame(rows, columns=cols, dtype=object)


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


def no_data_text(lang, extra=""):
    meta = get_meta()
    parts = [NO_DATA[lang]]
    if extra:
        parts.append(extra[:400])
    if meta.get("mn") and meta.get("mx"):
        parts.append(f"{TR[lang]['period_avail']}: {meta['mn']} – {meta['mx']}.")
    return "\n\n".join(parts)


def answer_question(question, history_text):
    con = init_db()
    schema, readme = get_context()
    meta = get_meta()
    product_rules = PRODUCT_RULES_NEW if has_product_norm(con) else PRODUCT_RULES_OLD
    sql_sys = (
        SQL_RULES.replace("@@PRODUCT_RULES@@", product_rules)
        + "\n\nОПИСАНИЕ ПОЛЕЙ (README):\n" + readme
        + "\n\nСХЕМА ДАННЫХ:\n" + schema
    )

    user_prompt = f"Вопрос: {question}"
    if history_text:
        user_prompt = "ИСТОРИЯ ДИАЛОГА:\n" + history_text + "\n\n" + user_prompt

    plan = parse_json(call_llm(sql_sys, user_prompt))
    lang = "uz" if str(plan.get("lang", "ru")).lower().startswith("uz") else "ru"
    out = {"lang": lang, "plan": plan, "sql": None, "df": None, "answer": None,
           "error": None, "guard": False}

    sql = plan.get("sql")
    if not sql or str(sql).strip().lower() in ("null", "none"):
        out["answer"] = no_data_text(lang, str(plan.get("reason") or ""))
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
                    f"{user_prompt}\nПрошлый SQL:\n{sql}\nОшибка: {err}\n"
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
        note = TR[lang]["no_rows"]
        if plan.get("filters_note"):
            note += " " + str(plan.get("filters_note"))
        out["answer"] = no_data_text(lang, note)
        return out

    fdf = to_display_df(df)
    shown = fdf.head(50)
    full_q = plan.get("full_question") or question
    prompt = (
        f"Вопрос: {full_q}\nlang={lang}\n"
        f"fields_used: {plan.get('fields_used')}\nfilters_note: {plan.get('filters_note')}\n"
        f"total_rows={len(fdf)} (показано {len(shown)})\n"
        f"Результат SQL (CSV, числа уже отформатированы):\n{shown.to_csv(index=False)}"
    )
    try:
        ans = call_llm(ANSWER_SYS, prompt, json_mode=False).strip()
    except LLMBusy:
        ans = ""

    allowed = set()
    for r in fdf.itertuples(index=False, name=None):
        for cell in r:
            allowed |= num_tokens(str(cell), include_runs=False)
    extra_text = " ".join(
        [question, str(full_q), str(plan.get("filters_note") or ""), str(plan.get("fields_used") or ""),
         str(len(fdf)), "50", "1000", str(meta.get("mn")), str(meta.get("mx")), str(meta.get("n"))]
    )
    allowed |= num_tokens(extra_text, include_runs=True)

    if not ans or not numbers_ok(ans, allowed):
        out["answer"] = det_answer(shown, lang)
        out["guard"] = bool(ans)
    else:
        out["answer"] = ans
    return out


# ---------------- Состояние ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "counter" not in st.session_state:
    st.session_state.counter = 0


def set_pending(text):
    st.session_state.pending = text


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending = None


def set_fb(msg_id, value):
    for m in st.session_state.messages:
        if m.get("id") == msg_id:
            m["fb"] = value


def history_text():
    msgs = [m for m in st.session_state.messages if m["role"] == "assistant"][-3:]
    lines = []
    for m in msgs:
        plan = m.get("plan") or {}
        lines.append(
            f"- Вопрос: {m.get('question')}\n  SQL: {str(m.get('sql') or '—')[:1200]}\n"
            f"  Фильтры: {plan.get('filters_note') or '—'}"
        )
    return "\n".join(lines)


def esc(text):
    return str(text).replace("$", "\\$")


def render_assistant(m):
    mid = m["id"]
    st.markdown(esc(m["content"]))
    if m.get("guard"):
        st.caption("ℹ️ " + t("guard"))
    df = m.get("df")
    plan = m.get("plan") or {}
    has_rows = df is not None and not df.empty

    if has_rows:
        with st.expander(f"{t('table')} ({len(df)})"):
            st.dataframe(to_display_df(df).head(200))
        chart = build_chart(df)
        if chart is not None:
            cdf, is_time = chart
            with st.expander(t("chart")):
                if is_time:
                    st.line_chart(cdf)
                else:
                    st.bar_chart(cdf)
    if m.get("sql"):
        with st.expander(t("sql")):
            st.code(m["sql"], language="sql")
    if plan.get("fields_used") or plan.get("filters_note"):
        st.caption(
            f"**{t('fields')}:** {esc(plan.get('fields_used') or '—')}  \n"
            f"**{t('filters')}:** {esc(plan.get('filters_note') or '—')}"
        )
    if has_rows:
        meta = get_meta()
        if meta.get("n"):
            st.caption(
                t("source").format(
                    n=fmt_value(meta["n"]), mn=meta["mn"], mx=meta["mx"], k=fmt_value(len(df))
                )
            )
        st.download_button(
            t("csv"),
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"uzex_answer_{mid}.csv",
            mime="text/csv",
            key=f"dl_{mid}",
        )
    if m.get("error"):
        with st.expander(t("tech")):
            st.code(str(m["error"]))

    c1, c2, c3 = st.columns([1, 1, 8])
    if m.get("fb"):
        c3.caption(("👍 " if m["fb"] == "up" else "👎 ") + t("thanks"))
    else:
        c1.button("👍", key=f"up_{mid}", on_click=set_fb, args=(mid, "up"))
        c2.button("👎", key=f"down_{mid}", on_click=set_fb, args=(mid, "down"))
    if m.get("elapsed") is not None:
        st.caption(f"{t('time')}: {m['elapsed']:.1f} s")


def render_examples(prefix):
    cols = st.columns(2)
    for col, lg in zip(cols, ("ru", "uz")):
        with col:
            st.markdown(f"**{t('ex_' + lg)}**")
            for i, ex in enumerate(EXAMPLES[lg]):
                st.button(ex, key=f"{prefix}_{lg}_{i}", on_click=set_pending, args=(ex,))


# ---------------- Оформление (светлая и тёмная темы) ----------------
# Цвета заданы полупрозрачными, поэтому одинаково хорошо смотрятся в Light и Dark.
st.markdown(
    """<style>
/* Фон: тёплое свечение поверх цвета выбранной темы */
.stApp{
  background-image:
    radial-gradient(1000px 620px at 0% -8%, rgba(255,122,24,.42), transparent 62%),
    radial-gradient(800px 560px at 100% 25%, rgba(214,92,12,.30), transparent 62%),
    radial-gradient(1000px 640px at 40% 115%, rgba(150,60,8,.38), transparent 62%);
  background-attachment: fixed;
}
header[data-testid="stHeader"]{background:transparent;}
/* Скрыть Fork / GitHub / Deploy (меню темы остаётся) */
[data-testid="stToolbarActions"],[data-testid="stAppDeployButton"],.stDeployButton{display:none !important;}
footer{visibility:hidden;}

.block-container{max-width:1100px;padding-top:2rem;}

/* Заголовок */
.ttl{
  background:linear-gradient(90deg,#ff7a18,#ffb066);
  -webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;color:transparent;
  font-weight:800;letter-spacing:-0.5px;
}
.hero{font-size:2.6rem;line-height:1.2;margin:0 0 .4rem 0;padding:0;}

/* Боковая панель */
section[data-testid="stSidebar"]{
  background:rgba(255,122,24,.07) !important;
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border-right:1px solid rgba(255,122,24,.20);
}

/* «Стеклянные» карточки */
div[data-testid="stMetric"],
div[data-testid="stChatMessage"],
div[data-testid="stExpander"],
div[data-testid="stAlert"]{
  background:linear-gradient(135deg, rgba(255,255,255,.12), rgba(255,140,40,.07)) !important;
  border:1px solid rgba(255,170,90,.32) !important;
  border-radius:18px !important;
  box-shadow:0 8px 32px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.22);
  backdrop-filter:blur(16px) saturate(140%);-webkit-backdrop-filter:blur(16px) saturate(140%);
}
div[data-testid="stMetric"]{padding:14px 18px;}
div[data-testid="stMetricValue"]{font-size:1.8rem;}
div[data-testid="stMetricValue"] > div{white-space:normal;overflow:visible;text-overflow:clip;}
div[data-testid="stChatMessage"]{padding:14px 18px;margin-bottom:10px;}
div[data-testid="stExpander"] details{border:none;}

/* Кнопки (примеры, CSV, оценки) */
.stButton > button, .stDownloadButton > button{
  background:linear-gradient(135deg, rgba(255,255,255,.10), rgba(255,140,40,.08));
  border:1px solid rgba(255,170,90,.34);
  border-radius:14px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.18);
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  transition:background .15s, border-color .15s, transform .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  background:rgba(255,122,24,.24);
  border-color:#ff7a18;
}

/* Поле ввода (поиск) */
div[data-testid="stChatInput"]{
  background:linear-gradient(135deg, rgba(255,255,255,.12), rgba(255,140,40,.08)) !important;
  border:1px solid rgba(255,150,70,.55) !important;
  border-radius:28px !important;
  box-shadow:0 8px 28px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.2);
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  padding:2px 6px;
}
div[data-testid="stChatInput"]:focus-within{
  border-color:#ff7a18 !important;
  box-shadow:0 0 0 3px rgba(255,122,24,.28), 0 8px 28px rgba(0,0,0,.25);
}
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInput"] textarea{background:transparent !important;border:none !important;}
div[data-testid="stChatInput"] button{
  background:#ff7a18 !important;color:#fff !important;border-radius:50% !important;
}
div[data-testid="stChatInput"] button:hover{background:#ff9640 !important;}
div[data-testid="stAlert"], div[data-testid="stAlert"] *{color:inherit !important;}
div[data-testid="stBottom"] > div{background:transparent;}
div[data-testid="stBottomBlockContainer"]{background:transparent;}
</style>""",
    unsafe_allow_html=True,
)

# ---------------- Интерфейс ----------------
con = init_db()
tables = con.execute("SHOW TABLES").fetchall()
if not tables:
    st.error(t("no_data_err"))
    st.stop()
if not get_secret("GEMINI_API_KEY"):
    st.error(t("no_key"))
    st.stop()

with st.sidebar:
    st.radio(
        t("ui_lang"), ["Русский", "Oʻzbekcha"], horizontal=True, key="ui_lang_choice"
    )
    st.button(t("clear"), on_click=clear_chat, key="clear_btn")
    st.subheader(t("side_data"))
    for (tb,) in tables:
        n = con.execute(f'SELECT COUNT(*) FROM "{tb}"').fetchone()[0]
        st.write(f"**{tb}**: {fmt_value(n)}")
    # Временная диагностика: перед сдачей этот блок можно удалить.
    st.caption(f"product_norm: {DB_STATUS['product_norm']}  \nзащита файлов: {DB_STATUS['lock']}")

st.markdown(f'<h1 class="hero">📊 <span class="ttl">{t("title")}</span></h1>', unsafe_allow_html=True)
st.caption(t("subtitle"))

meta = get_meta()
if meta.get("n"):
    m1, m2, m3 = st.columns([1, 3, 1])
    m1.metric(t("m_deals"), fmt_value(meta["n"]))
    m2.metric(t("m_period"), f"{meta['mn']} – {meta['mx']}")
    m3.metric(t("m_tables"), str(len(tables)))

if not st.session_state.messages and not st.session_state.pending:
    st.info(t("welcome"))
    render_examples("main")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(esc(msg["content"]))
    else:
        with st.chat_message("assistant"):
            render_assistant(msg)

q = st.chat_input("Задайте вопрос / Savol bering")
if st.session_state.pending:
    q = st.session_state.pending
    st.session_state.pending = None

if q:
    q = q.strip()[:1000]
    hist = history_text()
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(esc(q))

    with st.chat_message("assistant"):
        t0 = time.time()
        with st.spinner(t("spinner")):
            try:
                res = answer_question(q, hist)
            except LLMBusy as e:
                lg = guess_lang(q)
                res = {"answer": TR[lg]["busy"], "sql": None, "df": None, "plan": {},
                       "lang": lg, "error": str(e), "guard": False}
            except Exception as e:
                lg = guess_lang(q)
                res = {"answer": FAIL[lg], "sql": None, "df": None, "plan": {},
                       "lang": lg, "error": str(e)[:500], "guard": False}
        st.session_state.counter += 1
        amsg = {
            "role": "assistant",
            "id": st.session_state.counter,
            "question": q,
            "content": res.get("answer") or "",
            "sql": res.get("sql"),
            "df": res.get("df"),
            "plan": res.get("plan") or {},
            "lang": res.get("lang"),
            "error": res.get("error"),
            "guard": res.get("guard", False),
            "elapsed": time.time() - t0,
            "fb": None,
        }
        st.session_state.messages.append(amsg)
        render_assistant(amsg)
