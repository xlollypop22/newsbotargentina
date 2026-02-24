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
FEEDS_EXTRA_FILE = os.environ.get("FEEDS_EXTRA_FILE", "feeds_extra.json")
STATE_FILE = os.environ.get("STATE_FILE", "state.json")

TOTAL_LIMIT = int(os.environ.get("TOTAL_LIMIT", "120"))
PER_FEED_SCAN = int(os.environ.get("PER_FEED_SCAN", "40"))

HOT_HOURS = int(os.environ.get("HOT_HOURS", "6"))
ARG_FILTER = os.environ.get("ARG_FILTER", "1") == "1"
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "18"))

MIN_NEWS = int(os.environ.get("MIN_NEWS", "3"))
MAX_NEWS = int(os.environ.get("MAX_NEWS", "6"))
MIN_PER_TARGET = int(os.environ.get("MIN_PER_TARGET", "1"))

TG_TEXT_LIMIT = int(os.environ.get("TG_TEXT_LIMIT", "3900"))
MAX_SUMMARY_CHARS = int(os.environ.get("MAX_SUMMARY_CHARS", "260"))

# Кликабельная иконка для ссылки (вариант A)
LINK_ICON = os.environ.get("LINK_ICON", "↗").strip() or "↗"


# ----------------- RUBRICS -----------------

RUBRICS = {
    "🔥 Горячее": [
        "urgente", "último momento", "ultima hora", "en vivo", "ahora", "breaking",
        "alerta", "se confirmó", "confirmó", "confirmaron", "anuncio", "anunció",
        "habrá", "habra", "se cayó", "se cae", "renuncia", "renunció", "dimisión", "dimision"
    ],
    "💰 Экономика": [
        "economía", "economia", "inflación", "inflacion", "ipc", "índice", "indice", "indec",
        "recesión", "recesion", "dólar", "dolar", "blue", "mep", "ccl",
        "reservas", "banco central", "bcra", "fmi", "deuda", "bonos", "mercados",
        "riesgo país", "riesgo pais", "tasas", "exportaciones", "importaciones",
        "subsidios", "tarifas", "salarios", "paritarias", "pymes", "impuestos",
        "retenciones", "cepo", "devaluación", "devaluacion",
        "actividad", "pbi", "gasto", "déficit", "deficit", "superávit", "superavit",
        "licitación", "licitacion", "suba", "baja", "cae", "sube", "subieron", "cayeron",
        "precio", "precios", "consumo", "crédito", "credito", "finanzas", "billetera", "cuota"
    ],
    "🏛 Политика": [
        "milei", "presidente", "gobierno", "gabinete", "casa rosada",
        "jefe de gabinete", "ministro", "ministerio", "secretaría", "secretaria",
        "congreso", "senado", "diputados", "ley", "decreto", "dnu",
        "boletín oficial", "boletin oficial",
        "oposición", "oposicion", "peronismo", "kirchnerismo", "pro", "ucr", "lla",
        "kicillof", "massa", "bullrich", "macri", "larreta",
        "elecciones", "balotaje", "campaña", "alianza", "bloque",
        "corte suprema", "corte", "justicia electoral", "asamblea", "comisión", "comision"
    ],
    "🏢 Бизнес": [
        "empresa", "empresas", "negocio", "negocios", "inversión", "inversion",
        "startup", "fintech", "banco", "bancos", "mercado libre", "ml",
        "ypf", "telecom", "personal", "movistar", "claro", "aerolíneas", "aerolineas",
        "industria", "comercio", "inmobiliaria", "energía", "energia",
        "prepagas", "obra social", "aseguradora",
        "supermercado", "carrefour", "coto", "disco", "jumbo", "dia%",
        "exportador", "importador", "inversores", "producción", "produccion"
    ],
    "🎭 Культура": [
        "cultura", "cine", "teatro", "música", "musica", "festival", "libro",
        "feria del libro", "arte", "exposición", "exposicion", "concierto", "museo",
        "show", "estreno", "serie", "película", "pelicula", "streaming"
    ],
    "⚽ Спорт": [
        "fútbol", "futbol", "river", "boca", "selección", "seleccion", "messi",
        "copa", "liga", "mundial", "afa", "racing", "independiente", "san lorenzo",
        "newell", "rosario central", "estudiantes", "gimnasia", "velez", "huracán", "huracan",
        "tenis", "nba", "f1", "gran premio", "boxeo", "pumas", "rugby"
    ],
    "🌎 Общество": [
        "salud", "hospital", "educación", "educacion", "escuela", "universidad",
        "paro", "huelga", "sindicato", "cgt", "cta", "protesta", "marcha",
        "transporte", "subte", "colectivo", "tren", "vivienda", "alquiler",
        "servicios", "luz", "gas", "agua", "anmat",
        "seguridad", "policía", "policia", "crimen", "delito", "robo"
    ],
}

TARGET_RUBRICS = ["🏛 Политика", "💰 Экономика", "🏢 Бизнес", "🎭 Культура", "⚽ Спорт"]


# ----------------- ARGENTINA FILTER -----------------

ARG_DOMAINS = (
    "lanacion.com.ar",
    "clarin.com",
    "infobae.com",
    "perfil.com",
    "ambito.com",
    "cronista.com",
    "pagina12.com.ar",
    "tn.com.ar",
    "c5n.com",
    "ole.com.ar",
    "tycsports.com",
    "telesurtv.net",
)

ARG_STRONG_HINTS = [
    "argentina", "argentino", "argentinos", "república argentina", "republica argentina",
    "buenos aires", "caba", "amba", "gran buenos aires", "gba",
    "la plata", "mar del plata", "bahía blanca", "bahia blanca",
    "córdoba", "cordoba", "rosario", "mendoza", "salta", "tucumán", "tucuman",
    "neuquén", "neuquen", "san juan", "san luis", "santa fe",
    "entre ríos", "entre rios", "corrientes", "misiones", "chaco",
    "río negro", "rio negro", "chubut", "santa cruz", "tierra del fuego", "ushuaia",
    "quilmes", "avellaneda", "lanús", "lanus", "morón", "moron", "san isidro",
    "vicente lópez", "vicente lopez", "san martín", "san martin",
    "lomas de zamora", "la matanza", "tigre", "pilar", "escobar",
    "casa rosada", "congreso", "senado", "diputados",
    "boletín oficial", "boletin oficial",
    "bcra", "banco central", "indec", "afip", "arba", "anmat", "anses",
    "prefectura", "gendarmería", "gendarmeria",
    "ministerio", "secretaría", "secretaria",
    "subte", "colectivo", "metrobús", "metrobus",
    "tren roca", "tren mitre", "tren sarmiento", "tren belgrano",
    "aerolineas argentinas", "ypf", "edenor", "edesur",
    "mercado libre", "mercadopago", "mercado pago",
    "prepagas", "obra social",
    "dólar blue", "dolar blue", "cepo",
    "paritarias", "piquete", "cgt",
    "quilombo"
]

ARG_WEAK_HINTS = [
    "milei", "kicillof", "massa", "bullrich", "macri",
    "boca", "river", "afa",
    "independiente", "racing", "san lorenzo",
    "ole", "tyc", "tn",
    "patagonia", "pampa", "pampeano",
]

ARG_URL_MARKERS = [
    "/argentina", "/buenos-aires", "/caba", "/amba",
    "/cordoba", "/rosario", "/mendoza", "/la-plata",
    "/santa-fe", "/tucuman", "/salta", "/neuquen",
    "/san-juan", "/san-luis", "/chaco", "/misiones", "/corrientes",
    "/entre-rios", "/rio-negro", "/chubut", "/santa-cruz", "/tierra-del-fuego",
    "/mar-del-plata", "/bahia-blanca", "/ushuaia",
]


# ----------------- JSON / TELEGRAM -----------------

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def load_feeds() -> List[Dict[str, str]]:
    base = load_json(FEEDS_FILE, [])
    extra = load_json(FEEDS_EXTRA_FILE, [])
    all_feeds = []
    seen_urls = set()
    for arr in (base, extra):
        if not isinstance(arr, list):
            continue
        for f in arr:
            if not isinstance(f, dict):
                continue
            name = (f.get("name") or "").strip()
            url = (f.get("url") or "").strip()
            if not name or not url:
                continue
            if url in seen_urls:
                continue
            all_feeds.append({"name": name, "url": url})
            seen_urls.add(url)
    return all_feeds


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


def tg_send_photo(photo_url: str, caption: str = ""):
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


def _is_arg_domain(link: str) -> bool:
    u = (link or "").lower()
    return any(d in u for d in ARG_DOMAINS)


def is_argentina_related(title: str, summary: str, link: str) -> bool:
    if not ARG_FILTER:
        return True

    title = title or ""
    summary = summary or ""
    link = link or ""
    blob = (title + " " + summary + " " + link).lower()

    if any(h in blob for h in ARG_STRONG_HINTS):
        return True

    url_l = link.lower()
    if any(m in url_l for m in ARG_URL_MARKERS):
        return True

    if _is_arg_domain(link) and any(h in blob for h in ARG_WEAK_HINTS):
        return True

    return False


def is_hot(ts: float, title: str, summary: str) -> bool:
    if (time.time() - ts) <= HOT_HOURS * 3600:
        return True
    t = ((title or "") + " " + (summary or "")).lower()
    return any(w in t for w in RUBRICS["🔥 Горячее"])


def detect_rubric(ts: float, title: str, summary: str) -> str:
    t = ((title or "") + " " + (summary or "")).lower()
    for rubric in TARGET_RUBRICS:
        if any(k in t for k in RUBRICS.get(rubric, [])):
            return rubric
    return "🌎 Общество"


# ----------------- IMAGE EXTRACTION -----------------

UA = "Mozilla/5.0 (compatible; ArgentinaDigestBot/1.4; +https://github.com/)"

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
                max_tokens=260,
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
                    "Сделай выжимку на русском: 1–2 предложения, нейтрально и фактологично, без оценки и клише. "
                    "Не добавляй фактов, которых нет в исходном тексте."
                ),
            },
            {"role": "user", "content": base},
        ],
    )

    text = clean_text((resp.choices[0].message.content or "").strip())
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[: MAX_SUMMARY_CHARS - 1].rstrip() + "…"
    return text


# ----------------- PICKING / FORMATTING -----------------

Item = Tuple[float, str, str, str, str, Optional[str]]  # (ts, source, title, link, summary, image_url)

def score_item(ts: float, title: str, summary: str) -> int:
    s = 0
    if is_hot(ts, title, summary):
        s += 3
    age_hours = max(0.0, (time.time() - ts) / 3600.0)
    if age_hours <= 6:
        s += 3
    elif age_hours <= 24:
        s += 2
    elif age_hours <= 72:
        s += 1
    return s


def build_text_message(selected: List[Tuple[str, Item]]) -> str:
    """
    Вариант A: заголовок БЕЗ ссылки, а ссылка "вшита" в кликабельную иконку после выжимки.
    Пример:
      • Заголовок
        Выжимка… ↗ (Источник)
    """
    lines: List[str] = [
        "<b>Аргентина — подборка новостей за день</b>",
        "Подборка новостей за день ниже 👇",
        "",
    ]

    current = None
    for rubric, (ts, source, title, link, summary, image_url) in selected:
        if rubric != current:
            lines.append(f"<b>{html_escape(rubric)}</b>")
            current = rubric

        ru = summarize_to_ru(title, summary)

        # Заголовок без ссылки (не громоздко)
        lines.append(f"• {html_escape(clean_text(title))}")

        # Кликабельная иконка на первоисточник
        icon_link = f"<a href=\"{html_escape(link)}\">{html_escape(LINK_ICON)}</a>"

        if ru:
            lines.append(f"  {html_escape(ru)} {icon_link} <i>({html_escape(source)})</i>")
        else:
            lines.append(f"  {icon_link} <i>({html_escape(source)})</i>")

        lines.append("")

        if len("\n".join(lines)) > TG_TEXT_LIMIT:
            while lines and len("\n".join(lines)) > (TG_TEXT_LIMIT - 20):
                lines.pop()
            lines.append("…")
            break

    text = "\n".join(lines).strip()
    if len(text) > TG_TEXT_LIMIT:
        text = text[: TG_TEXT_LIMIT - 1].rstrip() + "…"
    return text


# ----------------- MAIN -----------------

def main():
    feeds = load_feeds()
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
    candidates = candidates[:TOTAL_LIMIT]

    # фильтр Аргентины
    filtered: List[Item] = []
    for it in candidates:
        ts, source, title, link, summary, image_url = it
        if is_argentina_related(title, summary, link):
            filtered.append(it)

    # fallback: если фильтр строгий
    if len(filtered) < MIN_NEWS:
        fallback: List[Item] = []
        for it in candidates:
            if _is_arg_domain(it[3]):  # link
                fallback.append(it)
        seen_links = set(x[3] for x in filtered)
        for it in fallback:
            if it[3] not in seen_links:
                filtered.append(it)
                seen_links.add(it[3])
        filtered.sort(key=lambda x: x[0], reverse=True)

    if not filtered:
        tg_send_message("Сегодня по выбранным источникам не нашёл новостей.")
        return

    grouped: Dict[str, List[Item]] = defaultdict(list)
    for it in filtered:
        ts, source, title, link, summary, image_url = it
        r = detect_rubric(ts, title, summary)
        grouped[r].append(it)

    for r, items in grouped.items():
        items.sort(key=lambda x: (score_item(x[0], x[2], x[4]), x[0]), reverse=True)

    # -------- СБАЛАНСИРОВАННЫЙ ОТБОР 3–6 --------
    selected: List[Tuple[str, Item]] = []

    # 1) по 1 из каждой целевой рубрики
    for r in TARGET_RUBRICS:
        items = grouped.get(r, [])
        take = min(MIN_PER_TARGET, len(items))
        for i in range(take):
            selected.append((r, items[i]))

    used_links = {it[3] for _, it in selected}

    # 2) добиваем до MAX_NEWS лучшими оставшимися из целевых
    if len(selected) < MAX_NEWS:
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

    # 3) если всё ещё < MIN_NEWS — добираем из любых рубрик
    if len(selected) < MIN_NEWS:
        pool2: List[Tuple[str, Item]] = []
        for r, items in grouped.items():
            for it in items:
                if it[3] not in used_links:
                    pool2.append((r, it))
        pool2.sort(key=lambda x: (score_item(x[1][0], x[1][2], x[1][4]), x[1][0]), reverse=True)

        for r, it in pool2:
            if len(selected) >= MIN_NEWS:
                break
            selected.append((r, it))
            used_links.add(it[3])

    # финальная страховка
    if len(selected) < MIN_NEWS:
        flat: List[Tuple[str, Item]] = []
        for r, items in grouped.items():
            for it in items:
                flat.append((r, it))
        flat.sort(key=lambda x: x[1][0], reverse=True)
        for r, it in flat:
            if len(selected) >= MIN_NEWS:
                break
            if it[3] in used_links:
                continue
            selected.append((r, it))
            used_links.add(it[3])

    if not selected:
        tg_send_message("Сегодня по выбранным источникам не удалось собрать подборку.")
        return

    if len(selected) > MAX_NEWS:
        selected = selected[:MAX_NEWS]

    order_index = {r: i for i, r in enumerate(TARGET_RUBRICS)}
    selected.sort(key=lambda x: (order_index.get(x[0], 999), -x[1][0]))

    # -------- отправка: фото отдельно, текст отдельно --------
    lead_image = None
    for _, it in selected:
        if it[5]:
            lead_image = it[5]
            break

    if lead_image:
        tg_send_photo(lead_image, "<b>Аргентина — дайджест</b>")

    text = build_text_message(selected)
    tg_send_message(text)

    # сохраняем seen
    new_links = [it[3] for _, it in selected]
    state["seen_links"] = (state.get("seen_links", []) + new_links)[-3000:]
    save_json(STATE_FILE, state)


if __name__ == "__main__":
    main()
