import os, json, time, re
import feedparser
import requests
from dateutil import parser as dtparser
from openai import OpenAI

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHANNEL = os.environ["TELEGRAM_CHANNEL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

TOTAL_LIMIT = int(os.environ.get("TOTAL_LIMIT", "5"))
PER_FEED_SCAN = int(os.environ.get("PER_FEED_SCAN", "10"))
MAX_SUMMARY_CHARS = int(os.environ.get("MAX_SUMMARY_CHARS", "280"))

FEEDS_FILE = "feeds.json"
STATE_FILE = "state.json"

client = OpenAI(api_key=OPENAI_API_KEY)

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
    r = requests.post(url, json={
        "chat_id": TELEGRAM_CHANNEL,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }, timeout=30)
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

def summarize_to_ru(title: str, snippet: str) -> str:
    title = clean_text(title)
    snippet = clean_text(snippet)

    base = f"Заголовок: {title}\nТекст: {snippet}" if snippet else f"Заголовок: {title}"

    resp = client.responses.create(
        model="gpt-5-mini",
        input=[
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

    text = (resp.output_text or "").strip()
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

    lines = ["<b>Аргентина — ежедневная выжимка</b>\n"]
    new_links = []

    for ts, source, title, link, summary in picked:
        ru = summarize_to_ru(title, summary)
        lines.append(f"<b>{html_escape(source)}</b>")
        lines.append(f"• <a href=\"{html_escape(link)}\">{html_escape(title)}</a>")
        if ru:
            lines.append(f"  {html_escape(ru)}")
        lines.append("")
        new_links.append(link)

    text = "\n".join(lines).strip()
    if len(text) > 3800:
        text = text[:3790] + "…"

    tg_send_message(text)

    state["seen_links"] = (state.get("seen_links", []) + new_links)[-2000:]
    save_json(STATE_FILE, state)

if __name__ == "__main__":
    main()
