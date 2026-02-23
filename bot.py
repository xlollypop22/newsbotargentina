import os, json, time, re
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

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

FEEDS_FILE = os.environ.get("FEEDS_FILE", "feeds.json")
STATE_FILE = os.environ.get("STATE_FILE", "state.json")

TOTAL_LIMIT = int(os.environ.get("TOTAL_LIMIT", "30"))        # кандидаты до фильтров
PER_FEED_SCAN = int(os.environ.get("PER_FEED_SCAN", "25"))    # записей из RSS на фид

HOT_HOURS = int(os.environ.get("HOT_HOURS", "6"))

ARG_FILTER = os.environ.get("ARG_FILTER", "1") == "1"         # 1 = строго Аргентина

HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "18"))

# Отбор новостей и “сбалансированность”
MAX_NEWS = int(os.environ.get("MAX_NEWS", "5"))               # всего новостей в посте
MIN_PER_TARGET = int(os.environ.get("MIN_PER_TARGET", "1"))   # хотим хотя бы по 1 из каждой целевой рубрики (если есть)

# Ограничения Telegram caption (для фото) — максимально безопасно держать < 900–950
CAPTION_LIMIT = int(os.environ.get("CAPTION_LIMIT", "950"))

# Короткая выжимка, чтобы помещаться в caption
MAX_SUMMARY_CHARS = int(os.environ.get("MAX_SUMMARY_CHARS", "170"))  # 1 предложение, компактно


# ----------------- RUBRICS -----------------
# Оставил расширенные словари, но целевые рубрики ниже.

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
    "🏢 Бизнес": [
        "empresa", "empresas", "negocio", "negocios", "inversión", "inversion",
        "startup", "fintech", "banco", "bancos", "mercado libre", "ypf",
        "telecom", "personal", "movistar", "claro", "aerolíneas", "aerolineas",
        "industria", "comercio", "inmobiliaria", "energía", "energia"
    ],
    "🎭 Культура": [
        "cultura", "cine", "teatro", "música", "musica", "festival", "libro",
        "feria del libro", "arte", "exposición", "exposicion", "concierto", "museo"
    ],
    "⚽ Спорт": [
        "fútbol", "futbol", "river", "boca", "selección", "seleccion", "messi",
        "copa", "liga", "mundial", "afa", "racing", "independiente", "san lorenzo",
        "tenis", "nba", "f1", "gran premio"
    ],
    # резерв
    "🌎 Общество": [
        "salud", "hospital", "educación", "educacion", "escuela", "universidad",
        "paro", "huelga", "sindicato", "cgt", "protesta", "marcha",
        "transporte", "subte", "colectivo", "tren", "vivienda", "alquiler",
        "servicios", "luz", "gas", "agua", "anmat"
    ],
}

# ЦЕЛЕВЫЕ рубрики, которые ты просишь (в таком порядке в посте)
TARGET_RUBRICS = ["🏛 Политика", "💰 Экономика", "🏢 Бизнес", "🎭 Культура", "⚽ Спорт"]


ARG_HINTS = [
    "argentina", "argentino", "buenos aires", "caba", "amba", "gba",
    "córdoba", "cordoba", "rosario", "mendoza", "la plata",
    "santa fe", "tucumán", "tucuman", "salta", "neuquén", "neuquen",
    "san juan", "san luis", "chaco", "misiones", "corrientes",
    "entre ríos", "entre rios", "río negro", "rio negro",
    "chubut", "santa cruz", "tierra del fuego", "ushuaia",
    "mar del plata", "bahía blanca", "bahia blanca",
    "milei", "casa rosada", "gobierno", "presidente",
    "congreso", "senado", "diputados", "boletín oficial", "boletin oficial",
    "indec", "banco central", "bcra", "afip", "anmat",
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


def is_argentina_related(title: str, summary: str, link: str) -> bool:
    if not ARG_FILTER:
        return True

    t = (title + " " + summary + " " + (link or "")).lower()

    # домены-«якоря» (можешь расширить своим списком)
    if any(d in t for d in ("infobae.com", "lanacion.com.ar", "clarin.com", "perfil.com")):
        return True

    return any(h in t for h in ARG_HINTS)


def is_hot(ts: float, title: str, summary: str) -> bool:
    if (time.time() - ts) <= HOT_HOURS * 3600:
        return True
    t = (title + " " + summary).lower()
    return any(w in t for w in RUBRICS["🔥 Горячее"])


def detect_rubric(ts: float, title: str, summary: str) -> str:
    # горячее не используем как отдельную рубрику в финальном посте,
    # но используем для приоритезации внутри целевых рубрик
    t = (title + " " + summary).lower()

    # сначала целевые
    for rubric in TARGET_RUBRICS:
        if any(k in t for k in RUBRICS.get(rubric, [])):
            return rubric

    # потом всё остальное — пусть упадёт в "Общество"
    return "🌎 Общество"


# ----------------- IMAGE EXTRACTION (RSS -> HTML og:image) -----------------

UA = "Mozilla/5.0 (compatible; ArgentinaDigestBot/1.1; +https://github.com/)"

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
                max_tokens=180,
            )
        except Exception as e:
            msg = str(e)
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

    resp = _call_groq_chat(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты редактор и переводчик новостей. Входной текст на испанском (Аргентина). "
                    "Сделай выжимку на русском: 1 или 2 небольших предложений (без оценки, без клише). "
                    "Не добавляй много фактов. Пиши максимально компактно и понятным языком."
                ),
            },
            {"role": "user", "content": base},
        ],
    )

    text = clean_text((resp.choices[0].message.content or "").strip())
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[: MAX_SUMMARY_CHARS - 1].rstrip() + "…"
    return text


# ----------------- MAIN -----------------

Item = Tuple[float, str, str, str, str, Optional[str]]  # (ts, source, title, link, summary, image_url)

def score_item(ts: float, title: str, summary: str) -> int:
    """
    Простая приоритезация: свежесть + «горячее».
    """
    s = 0
    if is_hot(ts, title, summary):
        s += 3
    # свежесть: ближе к сейчас — выше
    age_hours = max(0.0, (time.time() - ts) / 3600.0)
    if age_hours <= 6:
        s += 3
    elif age_hours <= 24:
        s += 2
    elif age_hours <= 72:
        s += 1
    return s


def build_single_caption(selected: List[Tuple[str, Item]]) -> str:
    """
    Собираем ОДИН caption под фото. Должно влезть в CAPTION_LIMIT.
    Формат: заголовок + рубрики + 1-2 строки на новость.
    """
    lines: List[str] = ["<b>Аргентина — дайджест</b>"]

    current = None
    for rubric, (ts, source, title, link, summary, image_url) in selected:
        if rubric != current:
            lines.append("")
            lines.append(f"<b>{html_escape(rubric)}</b>")
            current = rubric

        ru = summarize_to_ru(title, summary)

        # В caption места мало: заголовок — кликабельный, затем 1 короткая фраза.
        lines.append(f"• <a href=\"{html_escape(link)}\">{html_escape(clean_text(title))}</a>")
        if ru:
            lines.append(f"  {html_escape(ru)}")

        # мягкий разделитель
        lines.append("")

        # если уже близко к лимиту — стоп
        if len("\n".join(lines)) > CAPTION_LIMIT:
            # откатываем последние строки и ставим многоточие
            while lines and len("\n".join(lines)) > (CAPTION_LIMIT - 10):
                lines.pop()
            lines.append("…")
            break

    text = "\n".join(lines).strip()
    if len(text) > CAPTION_LIMIT:
        text = text[: CAPTION_LIMIT - 1].rstrip() + "…"
    return text


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

        # свежие сверху
        entries.sort(key=lambda x: x[0], reverse=True)
        candidates.extend(entries)

    # общий пул — свежие сверху
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:TOTAL_LIMIT]

    # фильтр Аргентины
    filtered: List[Item] = []
    for it in candidates:
        ts, source, title, link, summary, image_url = it
        if is_argentina_related(title, summary, link):
            filtered.append(it)

    if not filtered:
        tg_send_message("Сегодня по выбранным источникам не нашёл новостей про Аргентину.")
        return

    # группировка по рубрикам + сортировка внутри рубрик по (score, ts)
    grouped: Dict[str, List[Item]] = defaultdict(list)
    for it in filtered:
        ts, source, title, link, summary, image_url = it
        r = detect_rubric(ts, title, summary)
        grouped[r].append(it)

    for r, items in grouped.items():
        items.sort(key=lambda x: (score_item(x[0], x[2], x[4]), x[0]), reverse=True)

    # -------- СБАЛАНСИРОВАННЫЙ ОТБОР --------
    selected: List[Tuple[str, Item]] = []

    # 1) возьмём минимум по 1 из каждой целевой рубрики (если есть)
    for r in TARGET_RUBRICS:
        items = grouped.get(r, [])
        take = min(MIN_PER_TARGET, len(items))
        for i in range(take):
            selected.append((r, items[i]))

    # 2) добиваем до MAX_NEWS лучшими оставшимися из целевых рубрик
    if len(selected) < MAX_NEWS:
        used_links = {it[3] for _, it in selected}
        pool: List[Tuple[str, Item]] = []
        for r in TARGET_RUBRICS:
            for it in grouped.get(r, []):
                if it[3] not in used_links:
                    pool.append((r, it))

        pool.sort(key=lambda x: (score_item(x[1][0], x[1][2], x[1][4]), x[1][0]), reverse=True)

        for r, it in pool:
            if len(selected) >= MAX_NEWS:
                break
            selected.append((r, it))
            used_links.add(it[3])

    # 3) если целевых не хватило — добираем из “Общество” (или любых) по свежести/score
    if len(selected) < MAX_NEWS:
        used_links = {it[3] for _, it in selected}
        pool2: List[Tuple[str, Item]] = []
        for r, items in grouped.items():
            for it in items:
                if it[3] not in used_links:
                    pool2.append((r, it))
        pool2.sort(key=lambda x: (score_item(x[1][0], x[1][2], x[1][4]), x[1][0]), reverse=True)
        for r, it in pool2:
            if len(selected) >= MAX_NEWS:
                break
            selected.append((r, it))
            used_links.add(it[3])

    if not selected:
        tg_send_message("Сегодня не удалось собрать подборку.")
        return

    # упорядочим в посте строго по TARGET_RUBRICS (остальные — в конце)
    order_index = {r: i for i, r in enumerate(TARGET_RUBRICS)}
    selected.sort(key=lambda x: (order_index.get(x[0], 999), -x[1][0]))

    # -------- ОДИН ПОСТ В TG: ФОТО + CAPTION --------
    # берём первую доступную картинку среди выбранных
    lead_image = None
    for _, it in selected:
        if it[5]:
            lead_image = it[5]
            break

    caption = build_single_caption(selected)

    if lead_image:
        tg_send_photo(lead_image, caption)
    else:
        # если ни у одной новости нет картинки — придётся без фото (телега иначе не умеет)
        tg_send_message(caption)

    # сохраняем seen
    new_links = [it[3] for _, it in selected]
    state["seen_links"] = (state.get("seen_links", []) + new_links)[-2500:]
    save_json(STATE_FILE, state)


if __name__ == "__main__":
    main()
