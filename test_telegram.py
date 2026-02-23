import os
import requests

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHANNEL = os.environ["TELEGRAM_CHANNEL"]

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
r = requests.post(url, json={
    "chat_id": CHANNEL,
    "text": "✅ TEST: бот может постить в канал",
    "disable_web_page_preview": True
}, timeout=30)

print("status:", r.status_code)
print("response:", r.text)
r.raise_for_status()
