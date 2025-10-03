from flask import Flask, render_template
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

app = Flask(__name__)

# 初始化 Firebase
cred = credentials.Certificate("firebase_credentials/service_account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# 🔹 固定的圖片 mapping
cat_images = {
    "A1": "https://firebasestorage.googleapis.com/v0/b/gold-chassis-473807-j1.firebasestorage.app/o/cat_cards%2FA1.png?alt=media&token=58d97409-e570-444c-8ed7-e647b1ec182b",
    "A2": "https://firebasestorage.googleapis.com/v0/b/gold-chassis-473807-j1.firebasestorage.app/o/cat_cards%2FA2.png?alt=media&token=3f2095f2-80d7-48a9-97e5-5b42afc4cabc",
    "A3": "https://firebasestorage.googleapis.com/v0/b/gold-chassis-473807-j1.firebasestorage.app/o/cat_cards%2FA3.png?alt=media&token=beaa5879-4ff1-41d2-b4d6-e35748f0f6b5",
    "B1": "https://firebasestorage.googleapis.com/v0/b/gold-chassis-473807-j1.firebasestorage.app/o/cat_cards%2FB1.png?alt=media&token=6083faac-6e23-45c2-b8d6-bac3e7a95b3b",
    "B2": "https://firebasestorage.googleapis.com/v0/b/gold-chassis-473807-j1.firebasestorage.app/o/cat_cards%2FB2.png?alt=media&token=807c6e80-c75e-4ded-bb63-3792fcb6cff4",
    "B3": "https://firebasestorage.googleapis.com/v0/b/gold-chassis-473807-j1.firebasestorage.app/o/cat_cards%2FB3.png?alt=media&token=d39495b1-67ef-4b6d-8fd3-8b193ee434aa",
    "C1": "https://firebasestorage.googleapis.com/v0/b/gold-chassis-473807-j1.firebasestorage.app/o/cat_cards%2FC1.png?alt=media&token=146a237e-4d49-4dd2-bafc-2d2b84347d0d",
    "C2": "https://firebasestorage.googleapis.com/v0/b/gold-chassis-473807-j1.firebasestorage.app/o/cat_cards%2FC2.png?alt=media&token=1ba9e8ec-6de0-4579-8a32-d0b7fff6bf3a",
    "C3": "https://firebasestorage.googleapis.com/v0/b/gold-chassis-473807-j1.firebasestorage.app/o/cat_cards%2FC3.png?alt=media&token=22d41015-dae2-4c19-a753-b9a4b4957843"
}


DEFAULT_CAT = "https://images.unsplash.com/photo-1526336024174-e58f5cdd8e13?crop=entropy&cs=tinysrgb&fit=max&fm=jpg"

def parse_submit_time(ts_str):
    # 例如: "2025年9月28日 中午12:46:03 [UTC+8]"
    ts_str = ts_str.replace("上午", "AM").replace("中午", "PM").replace("下午", "PM").replace("凌晨", "AM")
    return datetime.strptime(ts_str, "%Y年%m月%d日 %p%I:%M:%S [UTC+8]")

def get_latest_mind_score(user_uid):
    latest_score = None
    latest_time = None
    psy_docs = db.collection("users").document(user_uid).collection("psychology_tests").stream()
    
    for doc in psy_docs:
        data = doc.to_dict()
        doc_score = data.get("mind_score")
        doc_time = data.get("submit_time")
        #print("Doc ID:", doc.id, "mind_score:", doc_score, "submit_time:", doc_time) #如果要看log再把註解打開
        
        if doc_score is not None:
            if latest_time is None or doc_time > latest_time:
                latest_time = doc_time
                latest_score = doc_score
    
    print("Latest mind_score selected:", latest_score)
    return latest_score

def get_latest_health_score(user_uid):
    health_docs = db.collection("health_reports").where("user_uid", "==", user_uid).stream()
    latest_score = None
    latest_time = None
    for doc in health_docs:
        data = doc.to_dict()
        score = data.get("health_score")
        created_at = data.get("created_at")
        if created_at and score is not None:
            # 假設 created_at 是 timestamp
            dt = created_at
            if latest_time is None or dt > latest_time:
                latest_time = dt
                latest_score = score
    return latest_score

def get_cat_image(health_score, mind_score):
    # 區間判斷
    def score_to_interval(score):
        if 1 <= score <= 33:
            return 1
        elif 34 <= score <= 66:
            return 2
        elif 67 <= score <= 99:
            return 3
        else:
            return None

    p = score_to_interval(health_score)
    m = score_to_interval(mind_score)

    if p is None or m is None:
        return None

    prefix = "C" if m == 1 else ("B" if m == 2 else "A")
    key = f"{prefix}{p}"
    return cat_images.get(key)

@app.route("/result/<user_uid>")
def result(user_uid):
    # --- 抓分數 ---
    health_score = get_latest_health_score(user_uid)
    mind_score = get_latest_mind_score(user_uid)

    if health_score is None or mind_score is None:
        return "User scores not found"

    # --- 計算對應貓咪 key ---
    cat_key = get_cat_image(health_score, mind_score)

    # --- 只印這三行 ---
    print("Fetched health_score:", health_score)
    print("Fetched mind_score:", mind_score)
    print("Selected cat image key:", cat_key)

    return render_template("result.html", card_url=cat_key)

if __name__ == "__main__":
    app.run(debug=True)
