import os, json, time, re
import feedparser
import requests
from dateutil import parser as dtparser
from openai import OpenAI

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHANNEL = os.environ["TELEGRAM_CHANNEL"]

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is empty. Check GitHub Secrets.")

# Groq — OpenAI-compatible endpoint
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

TOTAL_LIMIT = int(os.environ.get("TOTAL_LIMIT", "5"))
PER_FEED_SCAN = int(os.environ.get("PER_FEED_SCAN", "10"))
MAX_SUMMARY_CHARS = int(os.environ.get("MAX_SUMMARY_CHARS", "280"))

FEEDS_FILE = "feeds.json"
STATE_FILE = "state.json"
from collections import defaultdict

# --- РУБРИКИ / ГОРЯЧЕЕ ---

HOT_HOURS = int(os.environ.get("HOT_HOURS", "6"))  # сколько часов считаем "горячим"
MAX_PER_RUBRIC = int(os.environ.get("MAX_PER_RUBRIC", "2"))  # максимум новостей в рубрике за пост
HOT_MAX = int(os.environ.get("HOT_MAX", "2"))  # максимум "горячих" за пост

RUBRICS = {
    # Служебная рубрика: попадание сюда = горячее
    "🔥 Горячее": [
        "urgente", "último momento", "ultima hora", "en vivo", "ahora", "breaking",
        "alerta", "se confirmó", "confirmó", "confirmaron"
    ],

    "🏛 Политика": [
        "milei", "presidente", "gobierno", "gabinete", "casa rosada", "jefe de gabinete",
        "congreso", "senado", "diputados", "ley", "decreto", "dnu", "boletín oficial",
        "oposición", "peronismo", "kirchnerismo", "cambiemos", "pro", "ucr", "lilia lemoine",
        "kicillof", "massa", "bullrich", "macri", "larreta", "patricia bullrich",
        "elecciones", "balotaje", "campaña"
    ],

    "💰 Экономика": [
        "economía", "inflación", "inflacion", "ipc", "índice", "indec", "recesión", "recesion",
        "dólar", "dolar", "blue", "mep", "ccl", "reservas", "banco central", "bcr", "bCRA",
        "fmi", "deuda", "bonos", "mercados", "riesgo país", "riesgo pais", "tasas",
        "exportaciones", "importaciones", "subsidios", "tarifas", "salarios", "paritarias",
        "pymes", "impuestos", "retenciones", "cepo", "devaluación", "devaluacion"
    ],

    "⚖️ Суд / безопасность": [
        "policía", "policia", "crimen", "delito", "robo", "homicidio", "asesinato",
        "detenido", "detuvieron", "allanamiento", "operativo", "narco", "drogas",
        "juez", "jueza", "fiscal", "tribunal", "causa", "condena", "juicio",
        "seguridad", "gendarmería", "gendarmeria", "prefectura"
    ],

    "🌎 Общество": [
        "salud", "hospital", "educación", "educacion", "escuela", "universidad",
        "paro", "huelga", "sindicato", "cgt", "protesta", "marcha",
        "transporte", "subte", "colectivo", "tren", "aerolineas",
        "vivienda", "alquiler", "inmuebles", "corte", "piquete",
        "servicios", "luz", "gas", "agua", "seguro", "anmat"
    ],

    "🏢 Бизнес / компании": [
        "empresa", "empresas", "negocio", "negocios", "inversión", "inversion",
        "startup", "fintech", "banco", "bancos", "mercado libre", "ypf",
        "telecom", "personal", "movistar", "claro", "aerolíneas", "aerolineas",
        "exportador", "importador", "industria", "comercio"
    ],

    "🧪 Наука / технологии": [
        "tecnología", "tecnologia", "ia", "inteligencia artificial", "software",
        "ciber", "ciberseguridad", "hack", "datos", "internet", "satélite", "satelite",
        "investigación", "investigacion", "conicet"
    ],

    "🌦 Погода / ЧС": [
        "tormenta", "lluvia", "granizo", "ola de calor", "ola de frio", "inundación", "inundacion",
        "alerta meteorológica", "alerta meteorologica", "evacuados", "incendio", "sismo"
    ],

    "🎭 Культура": [
        "cultura", "cine", "teatro", "música", "musica", "festival", "libro", "feria del libro",
        "arte", "exposición", "exposicion", "concierto"
    ],

    "⚽ Спорт": [
        "fútbol", "futbol", "river", "boca", "selección", "seleccion", "messi",
        "copa", "liga", "mundial", "aFA", "racing", "independiente", "san lorenzo"
    ],
}

RUBRIC_ORDER = [
    "🔥 Горячее",
    "💰 Экономика",
    "🏛 Политика",
    "🏢 Бизнес / компании",
    "⚖️ Суд / безопасность",
    "🌎 Общество",
    "🧪 Наука / технологии",
    "🌦 Погода / ЧС",
    "🎭 Культура",
    "⚽ Спорт",
]

# Если хочешь строго "только про Аргентину" — оставь включённым
ARG_FILTER = os.environ.get("ARG_FILTER", "1") == "1"
ARG_HINTS = [
    "argentina", "argentino", "buenos aires", "caba", "amba",
    "córdoba", "cordoba", "rosario", "mendoza", "la plata",
    "santa fe", "tucumán", "tucuman", "salta", "neuquén", "neuquen",
    "milei", "casa rosada", "congreso", "banco central", "indec",
]

def is_argentina_related(title: str, summary: str, link: str = "") -> bool:
    if not ARG_FILTER:
        return True
    t = (title + " " + summary + " " + (link or "")).lower()
    return any(h in t for h in ARG_HINTS)

def is_hot(ts: float, title: str, summary: str) -> bool:
    if (time.time() - ts) <= HOT_HOURS * 3600:
        return True
    t = (title + " " + summary).lower()
    return any(w in t for w in RUBRICS["🔥 Горячее"])

def detect_rubric(ts: float, title: str, summary: str) -> str:
    if is_hot(ts, title, summary):
        return "🔥 Горячее"
    t = (title + " " + summary).lower()
    for rubric in RUBRIC_ORDER:
        if rubric == "🔥 Горячее":
            continue
        keys = RUBRICS.get(rubric, [])
        if any(k in t for k in keys):
            return rubric
    return "🌎 Общество"

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def tg_send_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHANNEL,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    r.raise_for_status()


def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pick_time(entry) -> float:
    for key in ("published", "updated", "created"):
        if key in entry and entry[key]:
            try:
                return dtparser.parse(entry[key]).timestamp()
            except Exception:
                pass
    return time.time()


def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _call_groq_chat(messages, model: str, max_retries: int = 3):
    # простой ретрай на временные ошибки/лимиты
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
        except Exception as e:
            msg = str(e)
            # грубая эвристика: 429/5xx/timeout
            if attempt < max_retries - 1 and (
                "429" in msg or "Rate limit" in msg or "timeout" in msg or "5" in msg
            ):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise


def summarize_to_ru(title: str, snippet: str) -> str:
    title = clean_text(title)
    snippet = clean_text(snippet)

    base = f"Заголовок: {title}\nТекст: {snippet}" if snippet else f"Заголовок: {title}"

    # ✅ Groq модели (выбери одну):
    # - "llama-3.3-70b-versatile" (лучше качество, медленнее/дороже)
    # - "llama-3.1-8b-instant"   (быстрее/дешевле)
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    resp = _call_groq_chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты редактор новостей. Сформулируй краткую выжимку на русском языке. "
                    "Стиль: нейтральный, фактологичный, без оценки и клише. "
                    "Длина: 1–2 предложения. Не добавляй фактов, которых нет во входном тексте."
                ),
            },
            {"role": "user", "content": base},
        ],
    )

    text = (resp.choices[0].message.content or "").strip()
    text = clean_text(text)
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[: MAX_SUMMARY_CHARS - 1].rstrip() + "…"
    return text


def main():
    feeds = load_json(FEEDS_FILE, [])
    state = load_json(STATE_FILE, {"seen_links": []})
    seen = set(state.get("seen_links", []))

    candidates = []

    for f in feeds:
        name, url = f["name"], f["url"]
        d = feedparser.parse(url)

        entries = []
        for e in d.entries[:PER_FEED_SCAN]:
            link = getattr(e, "link", None)
            title = getattr(e, "title", "").strip()
            summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""

            if not link or not title:
                continue
            if link in seen:
                continue

            ts = pick_time(e)
            entries.append((ts, name, title, link, summary))

        entries.sort(key=lambda x: x[0], reverse=True)
        candidates.extend(entries)

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked = candidates[:TOTAL_LIMIT]

    if not picked:
        tg_send_message("Сегодня новых новостей по выбранным источникам не нашёл.")
        return
    # --- группировка по рубрикам + лимиты на рубрику ---
    grouped = defaultdict(list)

    for ts, source, title, link, summary in picked:
        if not is_argentina_related(title, summary, link):
            continue
        rubric = detect_rubric(ts, title, summary)
        grouped[rubric].append((ts, source, title, link, summary))

    # Если после ARG-фильтра ничего не осталось — сообщаем
    if not any(grouped.values()):
        tg_send_message("Сегодня по выбранным источникам не нашёл новостей про Аргентину.")
        return

    lines = ["<b>Аргентина — ежедневная выжимка</b>\n"]
    new_links = []

    # отдельные лимиты для "горячих" и остальных
    hot_left = HOT_MAX

    for rubric in RUBRIC_ORDER:
        items = grouped.get(rubric, [])
        if not items:
            continue

        # сортируем внутри рубрики по свежести
        items.sort(key=lambda x: x[0], reverse=True)

        if rubric == "🔥 Горячее":
            items = items[:hot_left]
            hot_left -= len(items)
            if not items:
                continue
        else:
            items = items[:MAX_PER_RUBRIC]

        lines.append(f"<b>{html_escape(rubric)}</b>")

        for ts, source, title, link, summary in items:
            ru = summarize_to_ru(title, summary)
            lines.append(f"• <a href=\"{html_escape(link)}\">{html_escape(title)}</a> <i>({html_escape(source)})</i>")
            if ru:
                lines.append(f"  {html_escape(ru)}")
            new_links.append(link)

        lines.append("")

    text = "\n".join(lines).strip()
    if len(text) > 3800:
        text = text[:3790] + "…"

    tg_send_message(text)

    state["seen_links"] = (state.get("seen_links", []) + new_links)[-2000:]
    save_json(STATE_FILE, state)

if __name__ == "__main__":
    main()
