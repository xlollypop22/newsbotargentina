import os, json, time, re
from collections import defaultdict
from typing import Optional, List, Tuple

import feedparser
import requests
from dateutil import parser as dtparser
from openai import OpenAI


# ----------------- ENV / CLIENT -----------------

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHANNEL = os.environ["TELEGRAM_CHANNEL"]

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is empty. Check GitHub Secrets.")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

TOTAL_LIMIT = int(os.environ.get("TOTAL_LIMIT", "15"))         # сколько новостей взять в кандидаты (до фильтров)
PER_FEED_SCAN = int(os.environ.get("PER_FEED_SCAN", "20"))     # сколько записей читать из каждого RSS
MAX_SUMMARY_CHARS = int(os.environ.get("MAX_SUMMARY_CHARS", "280"))

HOT_HOURS = int(os.environ.get("HOT_HOURS", "6"))

ARG_FILTER = os.environ.get("ARG_FILTER", "1") == "1"          # 1 = строго Аргентина

FEEDS_FILE = "feeds.json"
STATE_FILE = "state.json"

HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "20"))       # таймаут на загрузку страниц (для og:image)

# Адаптивное число новостей в посте
MIN_NEWS = int(os.environ.get("MIN_NEWS", "2"))
MAX_NEWS = int(os.environ.get("MAX_NEWS", "6"))


# ----------------- RUBRICS -----------------

RUBRICS = {
    "🔥 Горячее": [
        "urgente", "último momento", "ultima hora", "en vivo", "ahora", "breaking",
        "alerta", "se confirmó", "confirmó", "confirmaron"
    ],
    "💰 Экономика": [
        "economía", "economia", "inflación", "inflacion", "ipc", "índice", "indice", "indec",
        "recesión", "recesion", "dólar", "dolar", "blue", "mep", "ccl", "reservas",
        "banco central", "bcr", "bcra", "fmi", "deuda", "bonos", "mercados",
        "riesgo país", "riesgo pais", "tasas", "exportaciones", "importaciones",
        "subsidios", "tarifas", "salarios", "paritarias", "pymes", "impuestos",
        "retenciones", "cepo", "devaluación", "devaluacion"
    ],
    "🏛 Политика": [
        "milei", "presidente", "gobierno", "gabinete", "casa rosada", "jefe de gabinete",
        "congreso", "senado", "diputados", "ley", "decreto", "dnu", "boletín oficial", "boletin oficial",
        "oposición", "oposicion", "peronismo", "kirchnerismo", "pro", "ucr",
        "kicillof", "massa", "bullrich", "macri", "larreta", "elecciones", "balotaje", "campaña"
    ],
    "🏢 Бизнес / компании": [
        "empresa", "empresas", "negocio", "negocios", "inversión", "inversion",
        "startup", "fintech", "banco", "bancos", "mercado libre", "ypf",
        "telecom", "personal", "movistar", "claro", "aerolíneas", "aerolineas",
        "industria", "comercio"
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
        "transporte", "subte", "colectivo", "tren", "vivienda", "alquiler",
        "inmuebles", "piquete", "servicios", "luz", "gas", "agua", "anmat"
    ],
    "🧪 Наука / технологии": [
        "tecnología", "tecnologia", "ia", "inteligencia artificial", "software",
        "ciber", "ciberseguridad", "datos", "internet", "satélite", "satelite",
        "investigación", "investigacion", "conicet"
    ],
    "🌦 Погода / ЧС": [
        "tormenta", "lluvia", "granizo", "ola de calor", "ola de frio",
        "inundación", "inundacion", "alerta meteorológica", "alerta meteorologica",
        "evacuados", "incendio", "sismo"
    ],
    "🎭 Культура": [
        "cultura", "cine", "teatro", "música", "musica", "festival", "libro",
        "feria del libro", "arte", "exposición", "exposicion", "concierto"
    ],
    "⚽ Спорт": [
        "fútbol", "futbol", "river", "boca", "selección", "seleccion", "messi",
        "copa", "liga", "mundial", "afa", "racing", "independiente", "san lorenzo"
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

ARG_HINTS = [
     # страна / прилагательные
    "argentina", "argentino", "argentina",

    # столица и агломерация
    "buenos aires", "caba", "amba", "gba",

    # провинции/города (часто встречаются в локальных новостях)
    "córdoba", "cordoba", "rosario", "mendoza", "la plata",
    "santa fe", "tucumán", "tucuman", "salta", "neuquén", "neuquen",
    "san juan", "san luis", "chaco", "misiones", "corrientes",
    "entre ríos", "entre rios", "río negro", "rio negro",
    "chubut", "santa cruz", "tierra del fuego", "ushuaia",
    "mar del plata", "bahía blanca", "bahia blanca",

    # политика / институты
    "milei", "casa rosada", "gobierno", "presidente",
    "congreso", "senado", "diputados", "boletín oficial", "boletin oficial",

    # экономика / регуляторы
    "indec", "banco central", "bcra", "afip", "anmat",

    # аргентинские маркеры в новостях
    "subte", "colectivo", "tren roca", "tren mitre", "tren sarmiento",
    "aerolineas argentinas", "ypf", "mercado libre", "edenor", "edesur",
]


# ----------------- JSON / TELEGRAM -----------------

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


def tg_send_photo(photo_url: str, caption: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    r = requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHANNEL,
            "photo": photo_url,
            "caption": caption,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    r.raise_for_status()


# ----------------- HELPERS -----------------

def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def pick_time(entry) -> float:
    for key in ("published", "updated", "created"):
        if key in entry and entry[key]:
            try:
                return dtparser.parse(entry[key]).timestamp()
            except Exception:
                pass
    return time.time()


def is_argentina_related(title: str, summary: str, link: str = "") -> bool:
    if not ARG_FILTER:
        return True
    t = (title + " " + summary + " " + (link or "")).lower()
    return any(h in t for h in ARG_HINTS)


def is_hot(ts: float, title: str, summary: str) -> bool:
    # свежее за HOT_HOURS часов — горячее
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


# ----------------- IMAGE EXTRACTION (RSS -> HTML og:image) -----------------

UA = "Mozilla/5.0 (compatible; ArgentinaDigestBot/1.0; +https://github.com/)"

def extract_image_from_rss(entry) -> Optional[str]:
    if hasattr(entry, "media_content"):
        try:
            for m in entry.media_content:
                u = m.get("url")
                if u:
                    return u
        except Exception:
            pass

    if hasattr(entry, "links"):
        try:
            for l in entry.links:
                href = l.get("href")
                ltype = (l.get("type") or "").lower()
                rel = (l.get("rel") or "").lower()
                if href and (ltype.startswith("image/") or rel == "enclosure"):
                    return href
        except Exception:
            pass

    summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
    m = re.search(r'<img[^>]+src="([^"]+)"', summary)
    if m:
        return m.group(1)

    return None


def extract_og_image_from_html(url: str) -> Optional[str]:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": UA},
            timeout=HTTP_TIMEOUT,
            allow_redirects=True,
        )
        if r.status_code >= 400:
            return None
        html = r.text
    except Exception:
        return None

    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
    ]
    for p in patterns:
        m = re.search(p, html, flags=re.IGNORECASE)
        if m:
            img = m.group(1).strip()
            if img.startswith("//"):
                img = "https:" + img
            return img

    return None


def best_image(entry, link: str) -> Optional[str]:
    img = extract_image_from_rss(entry)
    if img:
        return img
    return extract_og_image_from_html(link)


# ----------------- GROQ SUMMARIZER -----------------

def _call_groq_chat(messages, model: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=320,
            )
        except Exception as e:
            msg = str(e)
            if attempt < max_retries - 1 and ("429" in msg or "Rate limit" in msg or "timeout" in msg or "5" in msg):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise


def summarize_to_ru(title: str, snippet: str) -> str:
    title = clean_text(title)
    snippet = clean_text(snippet)

    base = f"Заголовок: {title}\nТекст: {snippet}" if snippet else f"Заголовок: {title}"

    resp = _call_groq_chat(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты редактор и переводчик новостей. Входной текст на испанском (Аргентина). "
                    "Сделай понятную и интересную выжимку на русском языке. "
                    "Стиль: нейтральный, фактологичный, без оценки и клише. "
                    "Пиши просто, как для обычных людей. "
                    "Длина: 2-3 предложения. Не добавляй фактов или идей, которых нет во входном тексте."
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


# ----------------- MAIN -----------------

Item = Tuple[float, str, str, str, str, Optional[str]]  # (ts, source, title, link, summary, image_url)

def main():
    feeds = load_json(FEEDS_FILE, [])
    state = load_json(STATE_FILE, {"seen_links": []})
    seen = set(state.get("seen_links", []))

    candidates: List[Item] = []

    for f in feeds:
        name, url = f["name"], f["url"]
        d = feedparser.parse(url)

        entries: List[Item] = []
        for e in d.entries[:PER_FEED_SCAN]:
            link = getattr(e, "link", None)
            title = getattr(e, "title", "").strip()
            summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""

            if not link or not title:
                continue
            if link in seen:
                continue

            ts = pick_time(e)
            image_url = best_image(e, link)

            entries.append((ts, name, title, link, summary, image_url))

        entries.sort(key=lambda x: x[0], reverse=True)
        candidates.extend(entries)

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked = candidates[:TOTAL_LIMIT]

    if not picked:
        tg_send_message("Сегодня новых новостей по выбранным источникам не нашёл.")
        return

    # --- группировка по рубрикам ---
    grouped = defaultdict(list)
    for ts, source, title, link, summary, image_url in picked:
        if not is_argentina_related(title, summary, link):
            continue
        rubric = detect_rubric(ts, title, summary)
        grouped[rubric].append((ts, source, title, link, summary, image_url))

    if not any(grouped.values()):
        tg_send_message("Сегодня по выбранным источникам не нашёл новостей про Аргентину.")
        return

    # -------- АДАПТИВНЫЙ ОТБОР MIN_NEWS–MAX_NEWS --------
    selected: List[Tuple[str, Item]] = []

    # 1) горячие первыми
    hot_items = grouped.get("🔥 Горячее", [])
    hot_items.sort(key=lambda x: x[0], reverse=True)
    for item in hot_items:
        if len(selected) >= MAX_NEWS:
            break
        selected.append(("🔥 Горячее", item))

    # 2) затем рубрики по приоритету
    for rubric in RUBRIC_ORDER:
        if rubric == "🔥 Горячее":
            continue
        items = grouped.get(rubric, [])
        items.sort(key=lambda x: x[0], reverse=True)
        for item in items:
            if len(selected) >= MAX_NEWS:
                break
            selected.append((rubric, item))
        if len(selected) >= MAX_NEWS:
            break

    # 3) добираем до MIN_NEWS из общего пула, если вдруг мало
    if len(selected) < MIN_NEWS:
        flat: List[Tuple[str, Item]] = []
        for r, items in grouped.items():
            for it in items:
                flat.append((r, it))
        flat.sort(key=lambda x: x[1][0], reverse=True)

        existing = set((r, it[3]) for r, it in selected)  # (rubric, link)
        for r, it in flat:
            key = (r, it[3])
            if key in existing:
                continue
            selected.append((r, it))
            existing.add(key)
            if len(selected) >= MIN_NEWS:
                break

    if not selected:
        tg_send_message("Сегодня по выбранным источникам не нашёл новостей про Аргентину.")
        return

    # -------- ФОРМИРОВАНИЕ ПОСТА --------
    lines = ["<b>Аргентина — ежедневная выжимка</b>\n"]
    new_links: List[str] = []

    current_rubric = None
    for rubric, (ts, source, title, link, summary, image_url) in selected:
        if rubric != current_rubric:
            lines.append(f"<b>{html_escape(rubric)}</b>")
            current_rubric = rubric

        ru = summarize_to_ru(title, summary)
        lines.append(
            f"• <a href=\"{html_escape(link)}\">{html_escape(title)}</a> "
            f"<i>({html_escape(source)})</i>"
        )
        if ru:
            lines.append(f"  {html_escape(ru)}")
        lines.append("")
        new_links.append(link)

    text = "\n".join(lines).strip()
    if len(text) > 3800:
        text = text[:3790] + "…"

    # ---- Variant B: one lead image (берём первую картинку среди выбранных) ----
    lead_image = None
    for rubric, (ts, source, title, link, summary, image_url) in selected:
        if image_url:
            lead_image = image_url
            break

    if lead_image:
        # caption limit ~1024, оставим запас
        if len(text) <= 950:
            tg_send_photo(lead_image, text)
        else:
            short_caption = "<b>Аргентина — ежедневная выжимка</b>\n\nСводка ниже 👇"
            tg_send_photo(lead_image, short_caption)
            tg_send_message(text)
    else:
        tg_send_message(text)

    state["seen_links"] = (state.get("seen_links", []) + new_links)[-2000:]
    save_json(STATE_FILE, state)


if __name__ == "__main__":
    main()
