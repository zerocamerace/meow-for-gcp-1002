from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    Response,
    abort,
)
import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
from firebase_admin.exceptions import FirebaseError
import os
from datetime import datetime
import logging
import time
import requests  # 🟡 0929修改：呼叫外部貓圖來源
import random  # 🟡 0929修改：貓咪圖卡風格隨機與備援使用
import textwrap  # 🟡 0929修改：圖卡文字換行處理
import hashlib  # 🟡 0929修改：圖卡輸出避免檔名衝突
import imghdr  # 🟡 0929修改：驗證下載圖片格式
from pathlib import Path  # 🟡 0929修改：設定圖卡輸出路徑
from io import BytesIO  # 🟡 0929修改：處理圖片位元組資料
from urllib.parse import urlparse  # 🟡 0929修改：驗證圖片網址安全性

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter  # 🟡 0929修改：繪製圖卡
from health_report_module import analyze_health_report
from google.cloud.firestore import SERVER_TIMESTAMP
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
import json
import re

def extract_json_from_response(text: str) -> dict:
    """抽取第一個 JSON 物件並解析。"""  # 0929修改03：強化解析容錯
    if text is None:
        raise ValueError("LLM returned None")

    raw = str(text).strip()

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"No JSON object found in: {raw[:200]}")

    candidate = match.group(0)
    candidate = candidate.replace("＂", '"').replace("＇", "'").replace("\ufeff", "")

    return json.loads(candidate)


def _build_genai_contents(system_instruction, conversation_history):
    contents = []

    if system_instruction:
        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=str(system_instruction))],
            )
        )

    for msg in conversation_history:
        role = msg.get("role", "user")
        parts = msg.get("parts", [])
        if not parts:
            logging.warning(f"Empty parts in message: {msg}")
            continue

        part_obj = parts[0]
        if isinstance(part_obj, dict):
            text = part_obj.get("text", "")
        else:
            text = str(part_obj)

        if not text:
            logging.warning(f"Empty text in message: {msg}")
            continue

        genai_role = "model" if role == "model" else "user"
        contents.append(
            genai_types.Content(
                role=genai_role,
                parts=[genai_types.Part(text=text)],
            )
        )

    return contents


def _generate_with_retry(contents, generation_config=None):
    model_candidates = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]  # 0929修改04：移除 1.5 系列，改用 2.5 模型

    last_error = None

    for model_name in model_candidates:
        for attempt in range(3):
            try:
                kwargs = {
                    "model": model_name,
                    "contents": contents,
                }
                if generation_config is not None:
                    kwargs["config"] = generation_config

                response = genai_chat_client.models.generate_content(**kwargs)
                if getattr(response, "candidates", None):
                    if attempt > 0 or model_name != model_candidates[0]:
                        logging.warning(
                            f"Gemini model '{model_name}' succeeded on attempt {attempt + 1}"
                        )
                    return response
                logging.warning(
                    f"Gemini model '{model_name}' returned empty candidates on attempt {attempt + 1}"
                )
            except Exception as e:
                logging.warning(
                    f"Gemini model '{model_name}' failed on attempt {attempt + 1}: {e}"
                )
                last_error = e
            time.sleep(1)

    if last_error:
        raise last_error
    raise RuntimeError("Gemini API returned empty response for all candidate models")


# 🟡 0929修改：共用工具
def _safe_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            return url
    except ValueError:
        pass
    return None


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        font_path = Path(path)
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception:
                continue
    logging.debug("Font fallback engaged for size %s", size)
    return ImageFont.load_default()


def _wrap_text(text: str | None, max_chars: int = 18) -> str:
    if not text:
        return ""
    collapsed = str(text).replace("\n", " ")
    return "\n".join(textwrap.wrap(collapsed, width=max_chars, break_long_words=True))


def _hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    hex_value = hex_value.lstrip("#")
    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))


def _hash_for_filename(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8", errors="ignore"))
    return hasher.hexdigest()[:8]


def _to_datetime(value):
    if value is None:
        return datetime.min
    if hasattr(value, "to_datetime"):
        try:
            return value.to_datetime()
        except TypeError:
            pass
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return datetime.min


def _cleanup_old_cards(max_files: int = 40):  # 🟡 0929修改：限制圖卡輸出數量
    try:
        files = sorted(CAT_CARD_DIR.glob("catcard_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[max_files:]:
            stale.unlink(missing_ok=True)
    except Exception as exc:
        logging.warning("Failed to cleanup old cat cards: %s", exc)


def _normalize_health_data(report: dict):
    """Collect warnings與重要指標，確保與舊版呈現一致。"""  # 🟡 0929修改：整理健檢資料給前端顯示
    warnings = []
    for key in ("health_warnings", "warnings", "alert_list", "warning_details"):
        value = report.get(key)
        if not value:
            continue
        if isinstance(value, list):
            warnings.extend(str(item) for item in value if item)
        elif isinstance(value, dict):
            warnings.extend(str(item) for item in value.values() if item)
        else:
            warnings.append(str(value))
    warnings = [w.strip() for w in warnings if w and isinstance(w, str)]

    vitals_display = []
    vitals = report.get("vital_stats") or report.get("vitals") or {}
    if isinstance(vitals, dict):
        vitals_iter = vitals.items()
    elif isinstance(vitals, list):
        vitals_iter = []
        for item in vitals:
            if isinstance(item, dict):
                vitals_iter.extend(item.items())
            else:
                vitals_display.append((str(item), ""))
    else:
        vitals_iter = []

    for key, value in vitals_iter:
        if value is None or value == "":
            continue
        vitals_display.append((str(key), str(value)))

    return warnings, vitals_display

# 🟡 0929修改：九宮格貓咪分區
def _score_to_interval(score) -> int | None:
    """將數值分數換成 1~3 區間。"""  
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value <= 33:
        return 1
    if value <= 66:
        return 2
    return 3

def _resolve_persona_key(health_score, mind_score) -> str | None:
    """根據身心分數挑選對應的既有貓咪圖。"""  # 🟡 0929修改：依分數選擇貓咪類型
    physical_zone = _score_to_interval(health_score)
    mental_zone = _score_to_interval(mind_score)
    if not physical_zone or not mental_zone:
        return None
    prefix = {1: "C", 2: "B", 3: "A"}.get(mental_zone)
    if not prefix:
        return None
    return f"{prefix}{physical_zone}"


def _validate_report_schema(payload: dict) -> dict:
    """驗證 Gemini 報告 JSON 結構，避免後續操作失敗。"""  # 🟡 0929修改05：補上遺失的 schema 檢查 helper
    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")

    for key in ("summary", "keywords", "emotionVector"):
        if key not in payload:
            raise ValueError(f"Missing key: {key}")

    if not isinstance(payload["summary"], str):
        raise TypeError("summary must be string")

    keywords = payload.get("keywords")
    if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
        raise TypeError("keywords must be list[str]")

    emotion_vector = payload.get("emotionVector")
    if not isinstance(emotion_vector, dict):
        raise TypeError("emotionVector must be object")

    for key in ("valence", "arousal", "dominance"):
        if key not in emotion_vector:
            raise ValueError(f"emotionVector missing key: {key}")
        if not isinstance(emotion_vector[key], (int, float)):
            raise TypeError(f"emotionVector.{key} must be number")

    return payload


def fetch_cat_image(max_retries: int = 3, timeout: int = 12, max_bytes: int = 8_000_000):
    """從 TheCatAPI 取得貓圖，失敗時改用備援圖庫。"""  # 🟡 0929修改：新增貓圖來源
    api_url = "https://api.thecatapi.com/v1/images/search?size=med&mime_types=jpg,png"
    headers = {}
    api_key = os.getenv("CAT_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    backoff = 1
    for attempt in range(max_retries):
        try:
            resp = requests.get(api_url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                payload = resp.json() or []
                if not payload:
                    raise ValueError("Cat API returned empty list")
                img_url = payload[0].get("url")
                if not img_url:
                    raise ValueError("Cat API payload missing url")
                image_bytes, final_url = _download_image(img_url, timeout, max_bytes)
                if image_bytes:
                    return image_bytes, final_url
            elif resp.status_code in {429, 500, 502, 503, 504}:
                logging.warning("Cat API temporary failure %s, backoff %ss", resp.status_code, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 8)
                continue
            else:
                raise ValueError(f"Cat API status {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            logging.warning("Cat API request failed (attempt %s/%s): %s", attempt + 1, max_retries, exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 8)

    logging.warning("Cat API all retries exhausted, switching to fallback image pool")
    fallback_url = random.choice(CAT_FALLBACK_IMAGES)
    image_bytes, final_url = _download_image(fallback_url, timeout, max_bytes, allow_fallback_errors=False)
    if image_bytes:
        return image_bytes, final_url
    logging.error("Fallback gallery also failed, using placeholder")
    placeholder = Image.new("RGB", (512, 512), "#fddde6")
    return placeholder, None


def _download_image(url: str, timeout: int, max_bytes: int, allow_fallback_errors: bool = True):
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            raise ValueError(f"Image status {resp.status_code}")
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type:
            raise ValueError(f"Unexpected content-type {content_type}")
        content_length = int(resp.headers.get("Content-Length", "0"))
        if content_length and content_length > max_bytes:
            raise ValueError(f"Image too large: {content_length}")
        data = resp.content
        if len(data) > max_bytes:
            raise ValueError("Image exceeds max_bytes")
        kind = imghdr.what(None, data)
        if kind not in {"jpeg", "png", "webp"}:
            raise ValueError(f"Unsupported image type: {kind}")
        return data, url
    except Exception as exc:
        if allow_fallback_errors:
            logging.warning("Download image failed for %s: %s", url, exc)
        else:
            logging.error("Download fallback image failed for %s: %s", url, exc)
        return None, None


def generate_cat_card_text(report: dict, psychology: dict, preferred_style: str):
    """呼叫 Gemini 產生貓卡敘述。"""  # 🟡 0929修改：貓卡文案
    prompt = (
        "你是一位數位貓咪圖卡設計師，會根據使用者的健康與心理測驗資料提供一隻陪伴貓咪。\n"
        "回傳 JSON，欄位包含 styleKey (bright/steady/healer 其一)、persona、name、speech (15 字內)、"
        "summary (60 字內)、insight (50 字內)、action (40 字內)、keywords (陣列，可空)。"
        "所有文字使用繁體中文。\n"
        f"建議風格：{preferred_style}\n"
        f"健康資料：{json.dumps(report, ensure_ascii=False, default=str)}\n"
        f"心理測驗：{json.dumps(psychology, ensure_ascii=False, default=str)}"
    )

    contents = _build_genai_contents(prompt, [])
    try:
        response = _generate_with_retry(contents, generation_config=JSON_RESPONSE_CONFIG)
        if not response or not getattr(response, "candidates", None):
            return None
        candidate = response.candidates[0]
        text = ""
        parts = getattr(candidate.content, "parts", None) or []  # 0929修改03：parts 可能為 None，改採空清單避免迴圈錯誤
        for part in parts:
            if getattr(part, "text", None):
                text += part.text
        try:
            parsed = extract_json_from_response(text)
        except Exception:
            logging.exception("0929修改03：Cat card JSON parse failed; raw snippet=%r", text[:500])
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        logging.warning("Cat card text fallback due to unparsable response")
    except Exception as exc:
        logging.error(f"generate_cat_card_text failed: {exc}")
    return None


CAT_STYLES = {
    "bright": {
        "title": "陽光守護者",
        "names": ["小橘光", "暖暖", "Sunny 喵"],
        "speech": ["今天也要補充水分喵！", "保持笑容，活力滿分！"],
        "description": "我感受到你{mood}的能量，讓我們一起維持 {health} 分的好狀態。",
        "actions": [
            "午休時間散步 10 分鐘，讓身體熱起來",
            "今天晚餐試試多彩蔬菜盤，補充維生素",
        ],
        "palette": ("#FFEAA7", "#FD79A8", "#FFAFCC", "#2d3436"),
    },
    "steady": {
        "title": "溫柔照護隊長",
        "names": ["小霧", "Cotton", "霜霜"],
        "speech": ["放慢腳步，我陪著你喵。", "今天也記得深呼吸三次。"],
        "description": "你的關鍵字是 {mood}，我會在日常提醒你保持節奏，讓 {health} 分更穩定。",
        "actions": [
            "睡前做 5 分鐘伸展，放鬆肌肉",
            "把今天的情緒寫在手帳，整理一下心緒",
        ],
        "palette": ("#E0FBFC", "#98C1D9", "#3D5A80", "#2d3436"),
    },
    "healer": {
        "title": "療癒訓練師",
        "names": ["小湯圓", "Mochi", "露露"],
        "speech": ["我們慢慢來，沒關係的喵。", "先照顧好自己，我在旁邊。"],
        "description": "看見你需要休息的訊號，我會當你的提醒小鬧鐘，陪你把 {health} 分調整回來。",
        "actions": [
            "安排 15 分鐘的呼吸練習，舒緩壓力",
            "今天對自己說聲辛苦了，給自己一個擁抱",
        ],
        "palette": ("#E8EAF6", "#C5CAE9", "#9FA8DA", "#2d3436"),
    },
}


def build_cat_card(report: dict, psychology: dict):
    """根據健康與心理測驗資料建立貓卡內容。"""  # 🟡 0929修改：組裝貓卡資料
    health_score = report.get("health_score")
    mood_score = (
        psychology.get("combined_score")
        or psychology.get("combinedScore")
        or psychology.get("mind_score")
    )
    keywords = psychology.get("keywords") or []

    health_value = float(health_score) if health_score is not None else 72.0
    mood_value = float(mood_score) if mood_score is not None else 68.0

    if health_value >= 80 or mood_value >= 80:
        suggested_style = "bright"
    elif health_value < 60:
        suggested_style = "healer"
    else:
        suggested_style = "steady"

    ai_payload = generate_cat_card_text(report, psychology, suggested_style)
    style_key = ai_payload.get("styleKey") if ai_payload and ai_payload.get("styleKey") in CAT_STYLES else suggested_style
    style = CAT_STYLES[style_key]

    # Finalize fields with AI payload or defaults
    # 🟡 0929修改：先試圖抓對應圖檔，失敗再退回 TheCatAPI
    name = (ai_payload or {}).get("name") or random.choice(style["names"])
    persona_key = _resolve_persona_key(health_value, mood_value)
    persona_label = CAT_PERSONA_METADATA.get(persona_key)
    persona = persona_label or (ai_payload or {}).get("persona") or style["title"]
    speech = (ai_payload or {}).get("speech") or random.choice(style["speech"])

    model_keywords = (ai_payload or {}).get("keywords") or keywords
    if isinstance(model_keywords, str):
        model_keywords = [k.strip() for k in model_keywords.split(",") if k.strip()]
    mood_label = "、".join(model_keywords[:3]) if model_keywords else "平衡"

    description = (ai_payload or {}).get("summary") or style["description"].format(
        mood=mood_label,
        health=int(round(health_value)),
    )
    insight = (ai_payload or {}).get("insight") or psychology.get("summary") or f"當下情緒偏向 {mood_label}，記得照顧自己。"
    action = (ai_payload or {}).get("action") or random.choice(style["actions"])

    vitality = max(0, min(100, int(round(health_value))))
    companionship = max(0, min(100, int(round(mood_value))))
    stability = max(0, min(100, int((vitality + companionship) / 2 + random.randint(-4, 4))))

    return {
        "persona": persona,
        "name": name,
        "speech": speech,
        "description": description,
        "insight": insight,
        "action": action,
        "stats": [
            {"label": "活力指數", "value": f"{vitality}%"},
            {"label": "陪伴力", "value": f"{companionship}%"},
            {"label": "穩定度", "value": f"{stability}%"},
        ],
        "style_key": style_key,
        "palette": style.get("palette"),
        "keywords_list": model_keywords,
        "persona_key": persona_key,
        "persona_label": persona_label,
    }


def circle_crop_image(image_bytes, diameter: int = 260) -> Image.Image:
    if isinstance(image_bytes, Image.Image):
        img = image_bytes
    else:
        img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    min_side = min(img.size)
    left = (img.width - min_side) // 2
    top = (img.height - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))
    img = img.resize((diameter, diameter), Image.LANCZOS)

    mask = Image.new("L", (diameter, diameter), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, diameter, diameter), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(0.6))

    output = Image.new("RGBA", (diameter, diameter))
    output.paste(img, (0, 0), mask)
    return output

    # 🟡 0929修改：繪製圖卡(先試圖抓對應圖檔，失敗再退回 TheCatAPI)
def render_cat_card_image(card: dict, user_id: str, cache_key: str | None = None):
    """生成圖卡 PNG，並回傳檔名與來源 URL。"""  # 🟡 0929修改：圖卡繪製
    timeout = 12
    max_bytes = 8_000_000
    image_bytes = None
    source_url = None

    persona_key = card.get("persona_key")
    if persona_key:
        persona_entry = CAT_PERSONA_IMAGES.get(persona_key)
        if persona_entry:
            local_path = persona_entry.get("local_path")
            if local_path and Path(local_path).exists():
                try:
                    image_bytes = Path(local_path).read_bytes()
                    static_path = persona_entry.get("static_path")
                    if static_path:
                        try:
                            source_url = url_for("static", filename=static_path, _external=True)
                        except RuntimeError:
                            source_url = f"/static/{static_path}"
                except Exception as exc:
                    logging.warning("Failed to load local persona image %s: %s", local_path, exc)
                    image_bytes = None
            else:
                candidate_url = persona_entry.get("url") if isinstance(persona_entry, dict) else persona_entry
                if candidate_url:
                    image_bytes, source_url = _download_image(candidate_url, timeout, max_bytes)
                    if not image_bytes:
                        logging.warning("Persona image download failed for %s", persona_key)

    if not image_bytes:
        image_bytes, source_url = fetch_cat_image(timeout=timeout, max_bytes=max_bytes)

    cat_image = circle_crop_image(image_bytes)

    width, height = 900, 600
    palette = card.get("palette", ("#FFEAA7", "#FD79A8", "#FFAFCC", "#2d3436"))
    bg_start, bg_end, accent, text_color = palette

    base = Image.new("RGB", (width, height), bg_start)
    draw = ImageDraw.Draw(base)

    start_rgb = _hex_to_rgb(bg_start)
    end_rgb = _hex_to_rgb(bg_end)
    for y in range(height):
        ratio = y / max(height - 1, 1)
        r = int(start_rgb[0] * (1 - ratio) + end_rgb[0] * ratio)
        g = int(start_rgb[1] * (1 - ratio) + end_rgb[1] * ratio)
        b = int(start_rgb[2] * (1 - ratio) + end_rgb[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    draw.rounded_rectangle((40, 40, width - 40, height - 40), radius=35, fill="white")

    title_font = _load_font(44)
    name_font = _load_font(56)
    body_font = _load_font(28)
    small_font = _load_font(24)
    caption_font = _load_font(22)

    x_margin = 80
    y = 90
    draw.text((x_margin, y), card.get("persona", "療癒系貓咪"), font=title_font, fill=accent)
    y += 70
    draw.text((x_margin, y), card.get("name", "專屬貓咪"), font=name_font, fill=text_color)
    y += 80

    speech_text = _wrap_text(card.get("speech"), 14)
    draw.text((x_margin, y), speech_text, font=body_font, fill=text_color)
    y += 110

    summary_text = _wrap_text(card.get("description"), 18)
    draw.text((x_margin, y), summary_text, font=body_font, fill=text_color)
    y += 120

    insight_text = _wrap_text(card.get("insight"), 20)
    if insight_text:
        draw.text((x_margin, y), f"心情結論：\n{insight_text}", font=small_font, fill=text_color)
        y += 120

    for stat in card.get("stats", []):
        draw.text((x_margin, y), f"{stat.get('label')}: {stat.get('value')}", font=small_font, fill=text_color)
        y += 40

    action_text = _wrap_text(card.get("action"), 18)
    if action_text:
        draw.text((x_margin, y), f"建議行動：{action_text}", font=small_font, fill=text_color)

    circle_x = width - 320
    circle_y = 120
    highlight_box = (circle_x - 20, circle_y - 20, circle_x + cat_image.width + 20, circle_y + cat_image.height + 20)
    draw.ellipse(highlight_box, fill="#fdf6ff")
    base.paste(cat_image, (circle_x, circle_y), cat_image)

    filename = f"catcard_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_hash_for_filename(user_id, str(time.time()), cache_key or '')}.png"
    output_path = CAT_CARD_DIR / filename
    base.save(output_path, format="PNG")
    _cleanup_old_cards()

    return filename, _safe_url(source_url)

# 載入 .env 檔案
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")  # 從 .env 載入或使用預設值
logging.basicConfig(level=logging.DEBUG)

# 🟡 0929修改：設定圖卡輸出位置與備援資料
BASE_DIR = Path(__file__).resolve().parent
CAT_CARD_DIR = BASE_DIR / "static" / "cat_cards"
CAT_CARD_DIR.mkdir(parents=True, exist_ok=True)

CAT_FALLBACK_IMAGES = [
    "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?auto=format&fit=crop&w=1000&q=80",
    "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?auto=format&fit=crop&w=1000&q=80",
    "https://images.unsplash.com/photo-1504208434309-cb69f4fe52b0?auto=format&fit=crop&w=1000&q=80",
    "https://images.unsplash.com/photo-1583083527882-4bee9aba2eea?auto=format&fit=crop&w=1000&q=80",
]

# 0929修改04：統一設定模型回傳純 JSON
JSON_RESPONSE_CONFIG = genai_types.GenerateContentConfig(
    response_mime_type="application/json",
    candidate_count=1,
    temperature=0.6,
)

# 🟡 0929修改：貓咪九宮格對應既有圖庫
_CAT_LOCAL_IMAGE_DIR = CAT_CARD_DIR / "images" / "cats"
_CAT_LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

CAT_PERSONA_IMAGES = {
    key: {
        "local_path": _CAT_LOCAL_IMAGE_DIR / f"{key}.png",
        "static_path": f"cat_cards/images/cats/{key}.png",
    }
    for key in ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3")
}

CAT_PERSONA_METADATA = {
    "A1": "布偶貓｜心理樂觀・身體指標待加油",
    "A2": "橘貓｜情緒穩定・生活節奏良好",
    "A3": "俄羅斯藍貓｜活力均衡・能量充沛",
    "B1": "波斯貓｜身心提醒・適度調養",
    "B2": "三花貓｜日常波動・持續照顧",
    "B3": "銀漸層貓｜外強內柔・記得舒壓",
    "C1": "折耳貓｜雙重負擔・先好好休息",
    "C2": "黑貓｜心理調適中・需要陪伴",
    "C3": "暹羅貓｜內在壓力大・身體仍有力",
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB W3.ttc",
    "/Library/Fonts/NotoSansCJK-Regular.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]

# 初始化 Firebase
firebase_credentials_env = os.getenv("FIREBASE_CREDENTIALS")
firebase_storage_bucket = os.getenv(
    "FIREBASE_STORAGE_BUCKET", "gold-chassis-473807-j1.firebasestorage.app"
)

try:
    if firebase_credentials_env:
        try:
            credential_payload = json.loads(firebase_credentials_env)
        except json.JSONDecodeError as exc:
            logging.error(f"FIREBASE_CREDENTIALS contains invalid JSON: {exc}")
            raise ValueError("Invalid FIREBASE_CREDENTIALS JSON payload") from exc

        cred = credentials.Certificate(credential_payload)
        logging.debug("Firebase credentials loaded from environment variable")
    else:
        credential_path = BASE_DIR / "firebase_credentials" / "service_account.json"
        cred = credentials.Certificate(str(credential_path))
        logging.debug(f"Firebase credentials loaded from file: {credential_path}")

    firebase_admin.initialize_app(cred, {"storageBucket": firebase_storage_bucket})
    logging.debug(
        "Firebase initialized successfully with bucket: %s", firebase_storage_bucket
    )
except FileNotFoundError as e:
    logging.error(f"Firebase credential file not found: {e}")
    raise
except ValueError as e:
    logging.error(f"Firebase initialization failed: {e}")
    raise

db = firestore.client()
try:
    bucket = storage.bucket()
    logging.debug(f"Storage bucket initialized: {bucket.name}")
except Exception as e:
    logging.error(f"Storage bucket initialization failed: {str(e)}")
    raise

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in .env file")
    raise ValueError("GEMINI_API_KEY is required")

try:
    genai_chat_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialise google-genai client: {e}")
    raise

# 🟢 修改：啟動時列印路由表（Flask 3 不支援 before_first_request，故保留註解）
# @app.before_first_request
# def _print_url_map():
#    logging.debug("URL Map:\n" + "\n".join([str(r) for r in app.url_map.iter_rules()]))

# 圖片代理：避免跨域限制影響下載圖卡
@app.route("/proxy_image")
def proxy_image():
    image_url = request.args.get("url", "")
    safe_url = _safe_url(image_url)
    if not safe_url:
        abort(400, description="Invalid image URL")

    try:
        upstream = requests.get(safe_url, timeout=6)
        upstream.raise_for_status()
    except requests.RequestException as exc:
        logging.warning("Image proxy fetch failed: %s", exc)
        abort(502, description="Image fetch failed")

    content_type = upstream.headers.get("Content-Type", "").lower()
    if not content_type.startswith("image"):
        detected = imghdr.what(None, upstream.content)
        if detected:
            content_type = f"image/{detected}"
        else:
            abort(415, description="Unsupported media type")

    response = Response(upstream.content, content_type=content_type or "image/png")
    response.headers["Cache-Control"] = "public, max-age=86400"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# 首頁
@app.route("/")
def home():
    is_logged_in = "user_id" in session
    return render_template("home.html", is_logged_in=is_logged_in)

# 註冊
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        session.pop("_flashes", None)

    if request.method == "POST":
        logging.debug(f"Received POST request with form data: {request.form}")
        email = request.form.get("email")
        password = request.form.get("password")
        # 🟢 修改開始：新增生理性別欄位
        gender = request.form.get("gender")
        logging.debug(
            f"Parsed form data: email={email}, password={'*' * len(password) if password else None}, gender={gender}"
        )

        if not email or not password or not gender:
            flash("請輸入電子郵件、密碼和生理性別！", "error")
            logging.warning("Missing email, password, or gender in form submission")
            return render_template("register.html", error="請輸入電子郵件、密碼和生理性別")
        # 🟢 修改結束
        try:
            user = auth.create_user(email=email, password=password)
            logging.debug(f"User created: uid={user.uid}, email={email}")
            db.collection("users").document(user.uid).set(
                {
                    "email": email,
                    # 🟢 修改開始：Firestore 儲存生理性別
                    "gender": gender,
                    # 🟢 修改結束
                    "created_at": SERVER_TIMESTAMP,
                    "last_login": None,
                }
            )
            logging.debug(f"User document created in Firestore for uid: {user.uid}")
            session["user_id"] = user.uid
            flash("註冊成功！請上傳健康報告。", "success")
            return redirect(url_for("upload_health"))
        except FirebaseError as e:
            error_message = str(e)
            logging.error(f"Firebase error during registration: {error_message}")
            flash(f"註冊失敗：{error_message}", "error")
            return render_template("register.html", error=f"註冊失敗：{error_message}")
        except Exception as e:
            logging.error(f"Unexpected error during registration: {str(e)}")
            flash(f"註冊失敗：{str(e)}", "error")
            return render_template("register.html", error=f"註冊失敗：{str(e)}")

    return render_template("register.html")

# 登入
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        session.pop("_flashes", None)

    if request.method == "POST":
        logging.debug(f"Received POST request with form data: {request.form}")
        email = request.form.get("email")
        password = request.form.get("password")
        logging.debug(
            f"Login attempt: email={email}, password={'*' * len(password) if password else None}"
        )

        if not email or not password:
            flash("請輸入電子郵件和密碼！", "error")
            logging.warning("Missing email or password in login submission")
            return render_template("login.html", error="請輸入電子郵件和密碼")

        try:
            user = auth.get_user_by_email(email)
            db.collection("users").document(user.uid).update(
                {"last_login": SERVER_TIMESTAMP}
            )
            logging.debug(f"User login updated in Firestore for uid: {user.uid}")
            session["user_id"] = user.uid
            flash("登入成功！", "success")
            return redirect(url_for("home"))
        except FirebaseError as e:
            error_message = str(e)
            logging.error(f"Login failed: {error_message}")
            flash(f"登入失敗：{error_message}", "error")
            return render_template("login.html", error=f"登入失敗：{error_message}")
        except Exception as e:
            logging.error(f"Unexpected login error: {str(e)}")
            flash(f"登入失敗：{str(e)}", "error")
            return render_template("login.html", error=f"登入失敗：{str(e)}")

    return render_template("login.html")

# 登出
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("_flashes", None)
    flash("已成功登出！", "success")
    return redirect(url_for("home"))

# 九宮格貓咪頁面
@app.route("/featured_cats")
def featured_cats():
    is_logged_in = "user_id" in session
    return render_template("featured_cats.html", is_logged_in=is_logged_in)

# 上傳健康報告
@app.route("/upload_health", methods=["GET", "POST"])
def upload_health():
    if "user_id" not in session:
        flash("請先登錄！", "error")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    logging.debug(f"Current user_id from session: {user_id}")

    # 🟢 修改開始：取得使用者生理性別
    user_gender = None
    try:
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists:
            flash("找不到使用者資料！", "error")
            logging.warning(f"User document not found for uid: {user_id}")
            return redirect(url_for("register"))
        user_data = user_doc.to_dict()
        user_gender = user_data.get("gender")
        if not user_gender:
            flash("請先完成註冊並提供生理性別資料！", "error")
            logging.warning(f"User gender missing for uid: {user_id}")
            return redirect(url_for("register"))
        logging.debug(f"Retrieved user gender from Firestore: {user_gender}")
    except Exception as e:
        logging.error(f"Failed to retrieve user gender: {str(e)}")
        flash(f"取得使用者資料失敗：{str(e)}", "error")
        return redirect(url_for("login"))
    # 🟢 修改結束

    # 🟢 修改開始：已有健檢報告時自動導向心理測驗
    reupload_requested = request.args.get("reupload") == "1"
    try:
        existing_reports = list(
            db.collection("health_reports")
            .where("user_uid", "==", user_id)
            .limit(1)
            .stream()
        )
    except Exception as e:
        logging.error(f"Failed to check existing health reports: {str(e)}")
        existing_reports = []

    has_existing_report = bool(existing_reports)

    auto_redirect = False
    if has_existing_report and not reupload_requested and request.method == "GET":
        logging.debug("Existing health report found; enabling auto redirect to psychology_test")
        auto_redirect = True
    # 🟢 修改結束

    if request.method == "POST":
        if "health_report" not in request.files:
            flash("未選擇檔案！", "error")
            return redirect(request.url)

        file = request.files["health_report"]
        if file.filename == "":
            flash("未選擇檔案！", "error")
            return redirect(request.url)

        logging.debug(
            f"Received POST request with form data: {request.form}, files: {request.files}"
        )

        # 檢查檔案類型
        is_image = file.mimetype in ["image/jpeg", "image/png"]
        is_pdf = file.mimetype == "application/pdf"
        if not (is_image or is_pdf):
            flash("僅支援 JPEG、PNG 或 PDF 檔案！", "error")
            return redirect(request.url)

        # 上傳檔案到 Firebase Storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        blob_path = f"health_reports/{user_id}/{filename}"
        logging.debug(f"Uploading file: {file.filename}")
        logging.debug(f"Uploading to Storage: {blob_path}")

        blob = bucket.blob(blob_path)
        blob.upload_from_file(file, content_type=file.mimetype)
        #blob.make_public()
        file_url = blob.public_url
        logging.debug(f"File uploaded successfully to Storage: {file_url}")

        # 分析健康報告
        logging.debug("Starting health report analysis...")
        try:
            file.seek(0)  # 重置檔案指針
            file_data = file.read()
            file_type = "image" if is_image else "pdf"
            analysis_data, health_score, health_warnings = analyze_health_report(
                file_data, user_id, file_type, gender=user_gender  # 🟢 修改：將生理性別傳遞至分析模組
            )
            logging.debug(
                f"Analysis result - data: {analysis_data is not None}, score: {health_score}, warnings: {len(health_warnings)}"
            )
            if not analysis_data:
                logging.warning("Health report analysis returned no data")
                flash("健康報告分析失敗，請確保檔案包含清晰數據！", "warning")
        except Exception as analysis_e:
            logging.error(f"Health report analysis failed: {str(analysis_e)}")
            flash(f"健康報告分析失敗：{str(analysis_e)}", "warning")
            analysis_data, health_score, health_warnings = None, 0, []

        # 準備 Firestore 文檔
        health_report_doc = {
            "user_uid": user_id,
            "report_date": datetime.now().strftime("%Y/%m/%d"),
            "filename": file.filename,
            "url": file_url,
            "file_type": file_type,
            "created_at": SERVER_TIMESTAMP,
        }
        if analysis_data:
            health_report_doc.update(
                {
                    "vital_stats": analysis_data.get("vital_stats", {}),
                    "health_score": health_score,
                    "health_warnings": health_warnings,
                }
            )
            logging.debug(
                f"Adding analysis data to doc: score={health_score}, warnings={health_warnings}"
            )

        # 儲存到 Firestore
        doc_ref = db.collection("health_reports").document()
        doc_ref.set(health_report_doc)
        report_id = doc_ref.id
        logging.debug(
            f"Health report SAVED to Firestore for user: {user_id}, report_id: {report_id}"
        )
        logging.debug(f"Saved document content: {health_report_doc}")

        # 驗證寫入
        saved_doc = db.collection("health_reports").document(report_id).get()
        if saved_doc.exists:
            logging.debug(
                f"Firestore write verified - document exists: {saved_doc.to_dict()}"
            )
        else:
            logging.error("Firestore write failed - document does not exist")

        flash(
            f"上傳成功！健康分數：{health_score}，警告：{'; '.join(health_warnings) if health_warnings else '無'}",
            "success",
        )
        return redirect(url_for("psychology_test"))

    return render_template(
        "upload_health.html",
        force_reupload=reupload_requested,
        has_existing_report=has_existing_report,
        auto_redirect=auto_redirect,
        psychology_url=url_for("psychology_test"),
    )

# 心理測驗
@app.route("/psychology_test", methods=["GET", "POST"])  # 🟢 修改：允許 POST 以處理心理測驗提交
def psychology_test():
    if "user_id" not in session:
        flash("請先登入！", "error")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    try:
        # 🟢 修改：改為查詢頂層 health_reports 並依 user_uid 過濾，避免找不到文件
        health_reports = list(
            db.collection("health_reports")
              .where("user_uid", "==", user_id)
              .stream()
        )  # 🟢 修改：原本是 users/{uid}/health_reports
        logging.debug(
            f"Psychology test check - existing reports: {len(health_reports)}"
        )
        if not health_reports:
            flash("請先上傳健康報告！", "error")
            return redirect(url_for("upload_health"))
    except Exception as e:
        logging.error(f"Error checking health reports: {str(e)}")
        flash(f"檢查健康報告失敗：{str(e)}", "error")
        return redirect(url_for("upload_health"))

    # 🟢 修改開始：支援心理測驗表單提交流程
    if request.method == "GET":
        session.pop("_flashes", None)

        latest_report_data = None
        try:
            def _report_sort_key(doc_snapshot):
                data = doc_snapshot.to_dict() or {}
                created = data.get("created_at")
                if hasattr(created, "timestamp"):
                    return created.timestamp()
                return 0.0

            if health_reports:
                latest_snapshot = max(health_reports, key=_report_sort_key)
                latest_report_data = latest_snapshot.to_dict() or {}
                created_at = latest_report_data.get("created_at")
                if hasattr(created_at, "isoformat"):
                    latest_report_data["created_at"] = created_at.isoformat()
        except Exception as e:
            logging.warning(f"Failed to prepare latest health report for template: {e}")
            latest_report_data = None

        return render_template(
            "psychology_test.html",
            is_logged_in=True,
            latest_health_report=latest_report_data,
        )

    question1 = request.form.get("question1")
    question2 = request.form.get("question2")
    if not question1 or not question2:
        flash("請回答所有問題！", "error")
        return render_template(
            "psychology_test.html", error="請回答所有問題", is_logged_in=True
        )

    try:
        db.collection("users").document(user_id).collection("psychology_tests").add(
            {
                "question1": question1,
                "question2": question2,
                "submit_time": SERVER_TIMESTAMP,
            }
        )
        logging.debug(f"Psychology test saved to Firestore for uid: {user_id}")
        flash("測驗提交成功！請生成貓咪圖卡。", "success")
        return redirect(url_for("generate_card"))
    except Exception as e:
        logging.error(f"Psychology test error: {str(e)}")
        flash(f"提交失敗：{str(e)}", "error")
        return render_template(
            "psychology_test.html", error=f"提交失敗：{str(e)}", is_logged_in=True
        )
    # 🟢 修改結束

# 聊天 API 端點（代理 Gemini API）
@app.route("/chat_api", methods=["POST"])
def chat_api():
    if "user_id" not in session:
        return jsonify({"error": "未登入"}), 401

    data = request.get_json()
    if not data or "conversationHistory" not in data or "systemInstruction" not in data:
        logging.error(f"Invalid request data: {data}")
        return jsonify({"error": "缺少必要的參數"}), 400

    try:
        logging.debug(f"Received conversationHistory: {data['conversationHistory']}")

        contents = _build_genai_contents(
            data.get("systemInstruction"), data["conversationHistory"]
        )

        if not contents:
            return jsonify({"error": "conversationHistory 為空或格式無效"}), 400

        try:
            response = _generate_with_retry(contents, generation_config=JSON_RESPONSE_CONFIG)
        except Exception as e:
            logging.error(f"Gemini generation failed: {e}")
            return jsonify({"nextPrompt": "AI 助手暫時無法回應，請稍後再試。"}), 200

        if not response or not getattr(response, "candidates", None):
            logging.error("Gemini API returned no candidates")
            return jsonify({"nextPrompt": "無法取得回應，請稍後再試。"}), 200

        candidate = response.candidates[0]
        reply = ""
        parts = getattr(candidate.content, "parts", None) or []  # 0929修改03：parts 可能為 None，改採空清單避免迴圈錯誤
        for part in parts:
            if getattr(part, "text", None):
                reply += part.text

        if not reply:
            logging.error("Gemini candidate did not include textual content")
            return jsonify({"nextPrompt": "無法取得回應，請稍後再試。"}), 200

        logging.debug(f"Raw reply: {reply}")

        try:
            parsed_json = extract_json_from_response(reply)
            logging.debug(f"Successfully parsed JSON: {parsed_json}")
        except Exception:
            logging.exception("0929修改03：chat_api JSON parse failed; raw snippet=%r", reply[:500])
            parsed_json = None

        if parsed_json and isinstance(parsed_json, dict):
            if "nextPrompt" in parsed_json or "summary" in parsed_json:
                return jsonify(parsed_json)
            return jsonify({"nextPrompt": reply})

        logging.warning(f"Could not parse JSON from reply, returning as plain text: {reply}")
        return jsonify({"nextPrompt": reply})
    
    except Exception as e:
        logging.error(f"Unexpected error in chat_api: {str(e)}, data: {data}")
        return jsonify({"error": f"伺服器錯誤：{str(e)}"}), 500

# 報告 API 端點（代理 Gemini API）
@app.route("/report_api", methods=["POST"])
def report_api():
    if "user_id" not in session:
        return jsonify({"error": "未登入"}), 401

    data = request.get_json()
    if not data or "conversationHistory" not in data or "systemInstruction" not in data:
        logging.error(f"Invalid request data: {data}")
        return jsonify({"error": "缺少必要的參數"}), 400

    try:
        logging.debug(f"Received conversationHistory for report: {len(data['conversationHistory'])} messages")

        contents = _build_genai_contents(
            data.get("systemInstruction"), data["conversationHistory"]
        )

        if not contents:
            return jsonify({"error": "conversationHistory 為空或格式無效"}), 400

        try:
            response = _generate_with_retry(contents, generation_config=JSON_RESPONSE_CONFIG)
        except Exception as e:
            logging.error(f"Gemini report generation failed: {e}")
            return jsonify({"summary": "模型沒有產生報告內容，請稍後再試。", "keywords": [], "emotionVector": {"valence": 50, "arousal": 50, "dominance": 50}}), 200

        if not response or not getattr(response, "candidates", None):
            logging.warning("Gemini report: no candidates, fallback to empty summary")
            report_json = {
                "summary": "模型沒有產生報告內容，請稍後再試。",
                "keywords": [],
                "emotionVector": {"valence": 50, "arousal": 50, "dominance": 50}
            }
            return jsonify(report_json), 200

        candidate = response.candidates[0]
        summary_text = ""
        parts = getattr(candidate.content, "parts", None) or []  # 0929修改03：parts 可能為 None，改採空清單避免迴圈錯誤
        for part in parts:
            if getattr(part, "text", None):
                summary_text += part.text

        if not summary_text:
            logging.warning("Gemini report: candidate present but empty text")
            report_json = {
                "summary": "模型沒有提供完整內容。",
                "keywords": [],
                "emotionVector": {"valence": 50, "arousal": 50, "dominance": 50}
            }
            return jsonify(report_json), 200

        logging.debug(f"Raw report summary: {summary_text}")

        try:
            parsed_json = extract_json_from_response(summary_text)
            parsed_json = _validate_report_schema(parsed_json)
            logging.debug(f"Successfully parsed report JSON: {parsed_json}")
            return jsonify(parsed_json)
        except Exception as exc:
            logging.exception("0929修改03：report_api JSON/schema failed: %s", exc)
            return (
                jsonify(
                    {
                        "error": "LLM returned invalid JSON",
                        "detail": str(exc),
                        "raw": summary_text[:500],
                    }
                ),
                502,
            )
    
    except Exception as e:
        logging.error(f"Unexpected error in report_api: {str(e)}, data: {data}")
        return jsonify({"error": f"伺服器錯誤：{str(e)}"}), 500

# 儲存心理測驗分數
# 🟢 修改：明確指定 endpoint 名稱，避免因函式名或載入順序造成的註冊差異
@app.route("/save_psychology_scores", methods=["POST"], endpoint="save_psychology_scores")  # 🟢 修改
def save_psychology_scores():
    if "user_id" not in session:
        return jsonify({"error": "未登入"}), 401

    data = request.get_json()
    if not data or not all(key in data for key in ["mindScore", "bodyScore", "combinedScore"]):
        return jsonify({"error": "缺少必要的分數參數"}), 400

    try:
        user_id = session["user_id"]
        test_id = db.collection("users").document(user_id).collection("psychology_tests").document().id
        db.collection("users").document(user_id).collection("psychology_tests").document(test_id).set(
            {
                "mind_score": data["mindScore"],
                "body_score": data["bodyScore"],
                "combined_score": data["combinedScore"],
                "summary": data.get("summary", ""),
                "keywords": data.get("keywords", []),
                "emotion_vector": data.get("emotionVector", {}),
                "conversation_history": data.get("conversationHistory", []),
                "submit_time": SERVER_TIMESTAMP
            }
        )
        logging.debug(f"Psychology scores saved for user {user_id}, test {test_id}")
        return jsonify({"status": "success", "test_id": test_id})
    except Exception as e:
        logging.error(f"Error saving psychology scores: {str(e)}")
        return jsonify({"error": f"儲存分數失敗：{str(e)}"}), 500

# 生成貓咪圖卡
@app.route("/generate_card")
def generate_card():
    if "user_id" not in session:
        flash("請先登入！", "error")
        return redirect(url_for("login"))

    session.pop("_flashes", None)

    try:
        user_id = session["user_id"]
        # 🟢 修改：同樣改為查詢頂層 health_reports
        health_report_docs = (
            db.collection("health_reports")
            .where("user_uid", "==", user_id)
            .stream()
        )
        reports = []
        for doc in health_report_docs:
            data = doc.to_dict() or {}
            data["id"] = doc.id
            reports.append(data)
        logging.debug(f"Generate card - reports found: {len(reports)}")
        if not reports:
            flash("請先上傳健康報告！", "error")
            return redirect(url_for("upload_health"))

        psych_docs = (
            db.collection("users")
            .document(user_id)
            .collection("psychology_tests")
            .stream()
        )
        tests = []
        for doc in psych_docs:
            data = doc.to_dict() or {}
            data["id"] = doc.id
            tests.append(data)
        if not tests:
            flash("請先完成心理測驗！", "error")  # 🟡 0929修改：修正提示字串
            return redirect(url_for("psychology_test"))

        latest_report = max(
            reports,
            key=lambda r: _to_datetime(r.get("created_at") or r.get("report_date")),
        )
        warnings, vitals_display = _normalize_health_data(latest_report)  # 🟡 0929修改：整理健檢提醒與指標
        latest_report["_display_warnings"] = warnings
        latest_report["_display_vitals"] = vitals_display

        latest_test = max(
            tests,
            key=lambda t: _to_datetime(t.get("submit_time") or t.get("created_at")),
        )

        use_cache = request.args.get("nocache") != "1"
        cache_key_current = f"{latest_report.get('id')}_{latest_test.get('id')}"
        cache_entry = session.get("cat_card_cache") if use_cache else None
        card_payload = None
        image_filename = None
        cat_source = None

        if cache_entry:
            cache_path = CAT_CARD_DIR / cache_entry.get("filename", "")
            cache_age = time.time() - cache_entry.get("timestamp", 0)
            cache_key_match = cache_entry.get("cache_key") == cache_key_current  # 🟡 1001修改01：僅當最新報告/測驗與快取一致時才沿用
            if cache_path.exists() and cache_age < 3600 and cache_key_match:
                logging.debug("Using cached cat card for user %s", user_id)
                card_payload = cache_entry.get("card", {})
                image_filename = cache_entry.get("filename")
                cat_source = _safe_url(cache_entry.get("cat_source"))

        if not card_payload or not image_filename:
            card_payload = build_cat_card(latest_report, latest_test)
            if warnings:
                card_payload["warnings"] = warnings
            cache_key = cache_key_current
            image_filename, cat_source = render_cat_card_image(card_payload, user_id, cache_key=cache_key)
            session["cat_card_cache"] = {
                "timestamp": time.time(),
                "filename": image_filename,
                "cat_source": cat_source,
                "card": card_payload,
                "cache_key": cache_key,  # 🟡 1001修改01：記錄本次使用的報告/測驗組合避免取用過期圖卡
            }

        card_image_url = url_for("static", filename=f"cat_cards/{image_filename}")
        card_payload["image_url"] = card_image_url
        card_payload["cat_image_source"] = cat_source
        card_payload.setdefault("warnings", warnings)

        return render_template(
            "generate_card.html",
            card=card_payload,
            card_image_url=card_image_url,
            report=latest_report,
            psychology=latest_test,
            is_logged_in=True,
        )
    except Exception as e:
        logging.error(f"Generate card error: {str(e)}")
        flash(f"生成圖卡失敗：{str(e)}", "error")
        return render_template(
            "generate_card.html", error=f"生成圖卡失敗：{str(e)}", is_logged_in=True
        )

if __name__ == "__main__":
    # 若要列印路由表，可在這裡印出（避免 Flask 3 的 before_first_request）
    # logging.debug("URL Map:\n" + "\n".join([str(r) for r in app.url_map.iter_rules()]))
    app.run(debug=True, port=5001)
