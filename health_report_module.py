import json
import logging
import os
import time
from io import BytesIO
from PIL import Image
# 🟢 修改（google-genai）：切換到新的 google-genai 套件
from google import genai
from google.genai import types as genai_types
from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting
from dotenv import load_dotenv
import datetime
import pdfplumber

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# --- 1. Settings ---
HEALTH_STANDARDS_FILE = "health_standards.json"

# --- 2. Initialize Services ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    _MODEL_PRIMARY = "gemini-2.5-flash"
    _MODEL_FALLBACKS = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.5-flash-lite",
    ]
    logging.debug("Gemini API initialized successfully.")
except Exception as e:
    logging.error(f"Gemini API initialization failed: {e}")
    raise Exception("Gemini API initialization failed.")


def _generate_gemini_content(parts, generation_config=None):
    """🟢 修改（google-genai）：透過新版 client 呼叫 Gemini。"""
    if not isinstance(parts, (list, tuple)):
        parts = [parts]

    if parts and isinstance(parts[0], genai_types.Content):
        contents = list(parts)
    else:
        user_parts = []
        for part in parts:
            if isinstance(part, genai_types.Part):
                user_parts.append(part)
            else:
                user_parts.append(genai_types.Part(text=str(part)))
        contents = [genai_types.Content(role="user", parts=user_parts)]

    model_candidates = [_MODEL_PRIMARY, *_MODEL_FALLBACKS]
    last_error = None

    for model_name in model_candidates:
        kwargs = {
            "model": model_name,
            "contents": contents,
        }
        if generation_config is not None:
            kwargs["generation_config"] = generation_config

        for attempt in range(3):
            try:
                response = genai_client.models.generate_content(**kwargs)
                if getattr(response, "candidates", None):
                    if model_name != _MODEL_PRIMARY or attempt > 0:
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

# Global variables to store health standards and alias mappings
HEALTH_STANDARDS = {}
HEALTH_ALIASES = {}

def load_health_standards():
    """Load health standards from a JSON file and create an alias mapping."""
    global HEALTH_STANDARDS, HEALTH_ALIASES
    try:
        with open(HEALTH_STANDARDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            HEALTH_STANDARDS = data.get("health_standards", {})
        
        # Create a reverse mapping from aliases to standard keys
        for key, value in HEALTH_STANDARDS.items():
            if 'aliases' in value and isinstance(value['aliases'], list):
                for alias in value['aliases']:
                    HEALTH_ALIASES[alias.strip().lower()] = key
        
        logging.debug(f"Health standards loaded: {list(HEALTH_STANDARDS.keys())}")
        logging.debug(f"Alias mapping created: {list(HEALTH_ALIASES.keys())}")
    except FileNotFoundError:
        logging.error(f"Failed to load health standards: {HEALTH_STANDARDS_FILE} not found.")
        raise
    except Exception as e:
        logging.error(f"Failed to load health standards: {e}")
        raise

load_health_standards()

# GALING 避免傳非健檢報告 10/5
def _is_valid_metric_value(value):
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return False
        invalid_tokens = {
            'null', 'n/a', 'na', 'none', 'unknown', '未提供', '未檢測', '未測', '無', '無資料', 'n.d.', '-', '—'
        }
        if cleaned in invalid_tokens:
            return False
        qualitative_tokens = {
            'positive', 'negative', '+', '-', '+/-', '++', '+++', '++++', '陽性', '陰性', 'trace', '微量'
        }
        if cleaned in qualitative_tokens:
            return True
        if any(ch.isdigit() for ch in cleaned):
            return True
    return False

# GALING 避免傳非健檢報告 10/5
def _count_recognized_metrics(vital_stats):
    if not isinstance(vital_stats, dict):
        return 0
    recognized = 0
    for key, value in vital_stats.items():
        if key in HEALTH_STANDARDS and _is_valid_metric_value(value):
            recognized += 1
    return recognized

# --- 3. Core Function Modules ---
def extract_pdf_text(pdf_data):
    """從 PDF 檔案提取文本"""
    try:
        with pdfplumber.open(BytesIO(pdf_data)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        logging.debug(f"Extracted PDF text: {text[:100]}...")
        return text
    except Exception as e:
        logging.error(f"Failed to extract PDF text: {str(e)}")
        return None

def get_gemini_prompt(user_uid, file_type, gender):
    """根據文件類型和性別生成 Gemini 提示"""
    base_prompt = f"""
你是個專業的醫療數據分析師，請你從這份健檢報告中，精準地提取出重要的健康數據。
請你務必使用繁體中文，並以 JSON 格式回傳。

請你務必嘗試尋找並回傳以下所有欄位。如果報告中沒有某個數值，請將其值設定為 null。
報告日期請使用當前日期（格式：yyyy/mm/dd）。
請注意，這份報告的受測者性別為：{gender}。

注意：請特別關注每個欄位的別名，並將其數值對應到正確的標準欄位名稱。
例如：如果報告中出現 "SGPT"，請將其數值填入 "alt"。如果出現 "TG"，請填入 "triglycerides"。

{{
  "user_uid": "{user_uid}",
  "report_date": "{datetime.datetime.now().strftime('%Y/%m/%d')}",
  "vital_stats": {{
    "glucose": null,
    "hemoglobin_a1c": null,
    "total_cholesterol": null,
    "triglycerides": null,
    "ldl_cholesterol": null,
    "hdl_cholesterol": null,
    "bmi": null,
    "alt": null,
    "ast": null,
    "creatinine": null,
    "egfr": null,
    "uric_acid": null,
    "wbc": null,
    "rbc": null,
    "hemoglobin": null,
    "platelet": null,
    "urine_glucose": null,
    "urine_protein": null,
    "blood_pressure_systolic": null,
    "blood_pressure_diastolic": null,
    "HBsAg": null,
    "urine_ob": null
  }}
}}

請你只回傳 JSON 格式的內容，不要包含任何額外的文字或說明。
"""
    if file_type == "pdf":
        return f"{base_prompt}\n以下是健檢報告的文本內容："
    return base_prompt

def analyze_image_with_gemini(image_data, user_uid, gender):
    """分析圖片並返回健康數據"""
    logging.info("Sending image to Gemini for analysis...")
    prompt = get_gemini_prompt(user_uid, "image", gender)
    
    try:
        img = Image.open(BytesIO(image_data))
        if img.format not in ["JPEG", "PNG"]:
            logging.error(f"Unsupported image format: {img.format}")
            return None
        if img.size[0] < 100 or img.size[1] < 100:
            logging.error(f"Image resolution too low: {img.size}")
            return None

        image_buffer = BytesIO()
        img.save(image_buffer, format=img.format)
        image_part = genai_types.Part.from_bytes(
            data=image_buffer.getvalue(), mime_type=f"image/{img.format.lower()}"
        )

        response = _generate_gemini_content([
            prompt,
            image_part,
        ])

        logging.info("Gemini image analysis complete, processing returned data...")
        gemini_output_str = (
            response.text.strip().replace("```json", "").replace("```", "")
        )
        logging.debug(f"Gemini raw output: {gemini_output_str}")

        try:
            vital_stats_json = json.loads(gemini_output_str)
            if (
                not isinstance(vital_stats_json, dict)
                or "vital_stats" not in vital_stats_json
            ):
                logging.error("Invalid JSON structure from Gemini")
                return None
            return vital_stats_json
        except json.JSONDecodeError as json_e:
            logging.error(f"Failed to parse Gemini JSON output: {str(json_e)}")
            return None

    except Exception as e:
        logging.error(f"Failed to analyze image with Gemini: {str(e)}")
        return None

def analyze_pdf_with_gemini(pdf_data, user_uid, gender):
    """分析 PDF 並返回健康數據"""
    logging.info("Sending PDF text to Gemini for analysis...")
    text = extract_pdf_text(pdf_data)
    if not text:
        logging.error("No text extracted from PDF")
        return None

    prompt = get_gemini_prompt(user_uid, "pdf", gender)
    
    try:
        response = _generate_gemini_content([prompt, text])

        logging.info("Gemini PDF analysis complete, processing returned data...")
        gemini_output_str = (
            response.text.strip().replace("```json", "").replace("```", "")
        )
        logging.debug(f"Gemini raw output: {gemini_output_str}")

        try:
            vital_stats_json = json.loads(gemini_output_str)
            if (
                not isinstance(vital_stats_json, dict)
                or "vital_stats" not in vital_stats_json
            ):
                logging.error("Invalid JSON structure from Gemini")
                return None
            return vital_stats_json
        except json.JSONDecodeError as json_e:
            logging.error(f"Failed to parse Gemini JSON output: {str(json_e)}")
            return None

    except Exception as e:
        logging.error(f"Failed to analyze PDF with Gemini: {str(e)}")
        return None

def calculate_health_score(vital_stats, gender=None):
    """
    根據健檢數據與分級標準計算分數。
    A級扣5分、B級扣10分、C級扣15分，滿分100分最低1分。
    
    Args:
        vital_stats (dict): 從健檢報告提取的數據。
        gender (str, optional): 使用者的性別，'male' 或 'female'。
    """
    score = 100
    warnings = []
    
    gender_key = gender.lower() if gender in ['male', 'female'] else None

    # Helper function to get numeric value, also handles qualitative text
    def get_numeric_value(val):
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            val_lower = val.strip().lower()
            if val_lower in ["負", "negative", "(-)", "-"]: return 0
            if val_lower in ["+/-", "+"]: return 1
            if val_lower in ["++", "+++"]: return 2
            if val_lower == "++++": return 3
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    for key, value in vital_stats.items():
        standard_info = HEALTH_STANDARDS.get(key)
        
        if not standard_info or value is None or 'grades' not in standard_info:
            continue
        
        grade = None
        grades_to_check = None
        
        if gender_key and gender_key in standard_info['grades']:
            grades_to_check = standard_info['grades'][gender_key]
        elif 'general' in standard_info['grades']:
            grades_to_check = standard_info['grades']['general']
        else:
            continue

        numeric_value = get_numeric_value(value)

        # Handle quantitative values
        if numeric_value is not None:
            for grade_level, boundaries in grades_to_check.items():
                if isinstance(boundaries, list) and len(boundaries) == 2:
                    if boundaries[0] <= numeric_value <= boundaries[1]:
                        grade = grade_level
                        break
        # Handle qualitative values
        else:
            value_lower = str(value).strip().lower()
            for grade_level, thresholds in grades_to_check.items():
                if isinstance(thresholds, list):
                    if value_lower in [t.strip().lower() for t in thresholds]:
                        grade = grade_level
                        break
        
        if grade == "A":
            score -= 5
            warnings.append(f"{standard_info.get('name', key)} 數值為 A 級 ({value})，超過正常範圍")
        elif grade == "B":
            score -= 10
            warnings.append(f"{standard_info.get('name', key)} 數值為 B 級 ({value})，顯著超出正常範圍")
        elif grade == "C":
            score -= 15
            warnings.append(f"{standard_info.get('name', key)} 數值為 C 級 ({value})，嚴重超出正常範圍")

    # 綜合性三高判斷 - 修正為根據數值判斷，而非警告訊息
    three_high_count = 0
    glucose_val = vital_stats.get("glucose")
    if glucose_val is not None and get_numeric_value(glucose_val) is not None and get_numeric_value(glucose_val) >= 100:
        three_high_count += 1
    tcho_val = vital_stats.get("total_cholesterol")
    if tcho_val is not None and get_numeric_value(tcho_val) is not None and get_numeric_value(tcho_val) >= 200:
        three_high_count += 1
    sys_bp_val = vital_stats.get("blood_pressure_systolic")
    if sys_bp_val is not None and get_numeric_value(sys_bp_val) is not None and get_numeric_value(sys_bp_val) >= 130:
        three_high_count += 1
            
    if three_high_count == 1:
        score -= 5
        warnings.append("符合「一高」條件，額外扣 5 分。")
    elif three_high_count == 2:
        score -= 10
        warnings.append("符合「兩高」條件，額外扣 10 分。")
    elif three_high_count == 3:
        score -= 15
        warnings.append("符合「三高」條件，額外扣 15 分。")

    if score < 1:
        score = 1
    
    logging.debug(f"Health score: {score}, Warnings: {warnings}")
    return score, warnings

# 新增生理性別參數
def analyze_health_report(file_data, user_id, file_type, gender):
    """
    執行完整的健檢報告分析流程，支援圖片和 PDF。
    """
    if file_type == "image":
        gemini_data = analyze_image_with_gemini(file_data, user_id, gender)
    elif file_type == "pdf":
        gemini_data = analyze_pdf_with_gemini(file_data, user_id, gender)
    else:
        logging.error(f"Unsupported file type: {file_type}")
        return None, 0, [], 0

    if not gemini_data:
        logging.warning("No data returned from Gemini analysis")
        return None, 0, [], 0

    # GALING 避免傳非健檢報告 10/5
    vital_stats = gemini_data.get("vital_stats", {})
    recognized_metric_count = _count_recognized_metrics(vital_stats)
    if recognized_metric_count <= 0:
        logging.warning("No recognizable health metrics found in report")
        return None, 0, [], 0

    # Example: In a real app, 'gender' would come from the user's profile
    health_score, health_warnings = calculate_health_score(
        vital_stats, gender=gender
    )
    logging.debug(
        f"Health score calculated: {health_score}, warnings: {health_warnings}, matched_metrics: {recognized_metric_count}"
    )

    return gemini_data, health_score, health_warnings, recognized_metric_count

