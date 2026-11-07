"""安全まわりのルール。Gemini には任せず、すべてコードで決める。

- 危機的な言葉の検知
- 有名な場所の判定（混雑を避けたい人向け）
- 開いているか・暗くないか・私有地っぽくないか
- アクセス数の制限
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

from .osm_data import category, haversine

JST = ZoneInfo("Asia/Tokyo")

# ---------- 危機的な言葉 ----------
CRISIS_WORDS = [
    "死にたい", "しにたい", "消えたい", "きえたい", "自殺", "じさつ", "死のう", "死ぬ方法",
    "生きていたくない", "生きたくない", "いなくなりたい", "飛び降り", "首吊", "首をつ",
    "リスカ", "自傷", "楽になりたい",
]

BLOCK_MESSAGE = (
    "いまはお散歩のルートを作るより、誰かと話すことを大切にしてほしいです。"
    "つらい気持ちを聞いてくれる窓口があります。厚生労働省「まもろうよ こころ」で、"
    "電話やSNSの相談先を探せます。https://www.mhlw.go.jp/mamorouyokokoro/"
)


def is_crisis(text: str) -> bool:
    t = text.replace(" ", "").replace("　", "")
    return any(w in t for w in CRISIS_WORDS)


# ---------- 有名な場所 ----------
FAMOUS_SPOTS = {
    "清水寺": (34.9949, 135.7850),
    "八坂神社": (35.0037, 135.7785),
    "高台寺": (34.9997, 135.7809),
    "知恩院": (35.0055, 135.7832),
    "伏見稲荷大社": (34.9671, 135.7727),
    "金閣寺": (35.0394, 135.7292),
    "鹿苑寺": (35.0394, 135.7292),
    "銀閣寺": (35.0270, 135.7982),
    "慈照寺": (35.0270, 135.7982),
    "二条城": (35.0142, 135.7481),
    "三十三間堂": (34.9879, 135.7718),
    "平安神宮": (35.0160, 135.7824),
    "南禅寺": (35.0113, 135.7944),
    "東福寺": (34.9765, 135.7737),
    "錦市場": (35.0050, 135.7650),
    "天龍寺": (35.0157, 135.6738),
    "渡月橋": (35.0128, 135.6778),
    "竹林の小径": (35.0170, 135.6717),
    "花見小路": (35.0025, 135.7752),
}
FAMOUS_RADIUS_M = 200


def near_famous(lat: float, lon: float) -> str | None:
    for name, (flat, flon) in FAMOUS_SPOTS.items():
        if haversine(lat, lon, flat, flon) <= FAMOUS_RADIUS_M:
            return name
    return None


def famous_names() -> list[str]:
    return list(FAMOUS_SPOTS)


# ---------- 日の入り（京都・月ごとのおおよその時刻、時） ----------
SUNSET = {1: 17.2, 2: 17.6, 3: 18.0, 4: 18.4, 5: 18.8, 6: 19.1,
          7: 19.1, 8: 18.7, 9: 18.1, 10: 17.4, 11: 16.9, 12: 16.8}


def is_dark(now: datetime) -> bool:
    h = now.hour + now.minute / 60
    return h >= SUNSET[now.month] or h < 6.0


# ---------- 私有地っぽい場所 ----------
PRIVATE_TYPES = {"school", "primary_school", "secondary_school", "university", "hospital",
                 "lodging", "hotel", "apartment_building", "housing_complex", "doctor"}
PRIVATE_WORDS = ["マンション", "アパート", "社宅", "寮", "工場", "ハイツ", "レジデンス"]
WATER_WORDS = ["川", "池", "滝", "沼"]


def decide(place: dict, google: dict | None, crowd: str, now: datetime) -> tuple[bool, str | None]:
    """場所を残すかどうかと、注意書き（なければ None）を返す。

    place  : OSM の場所（name, key, value, lat, lon）
    google : Places API の結果（見つからなければ None）。中身は判定にだけ使い、Gemini には渡さない
    """
    cat = category(place)
    name = place["name"]
    dark = is_dark(now)

    if any(w in name for w in PRIVATE_WORDS):
        return False, None

    if google:
        status = google.get("businessStatus")
        if status and status != "OPERATIONAL":
            return False, None  # CLOSED_TEMPORARILY / CLOSED_PERMANENTLY
        if PRIVATE_TYPES & set(google.get("types", [])):
            return False, None
        hours = google.get("currentOpeningHours")
        if hours is not None and hours.get("openNow") is False:
            return False, None
        if hours is None:
            # 営業時間がわからない（お店以外は料金節約のため取っていない）
            if cat == "shop":
                return False, None
            if dark:
                return False, None
    else:
        # Google で見つからなかった：店は閉店の可能性があるので外す。それ以外は明るいうちだけ
        if cat == "shop" or dark:
            return False, None

    if crowd == "avoid":
        if any(f in name for f in FAMOUS_SPOTS) or near_famous(place["lat"], place["lon"]):
            return False, None

    if dark and cat in ("outdoor", "worship", "historic"):
        return False, None

    warning = None
    if any(w in name for w in WATER_WORDS) or place["value"] == "water":
        warning = "水辺です。足元に気をつけて"
    elif cat == "outdoor" or place["value"] in ("viewpoint",):
        warning = "屋外です。暗くなる前に"
    elif not google:
        warning = "地図で確認できなかった場所です。入れなさそうなら次へ"
    return True, warning


# ---------- アクセス数の制限（Cloud Run は max-instances 1 なのでメモリで数える） ----------
class RateLimiter:
    def __init__(self, per_ip_per_hour: int = 15, global_per_day: int = 300):
        self.per_ip = per_ip_per_hour
        self.per_day = global_per_day
        self.hits: dict[str, deque] = defaultdict(deque)
        self.day = ""
        self.day_count = 0
        self.lock = threading.Lock()

    def allow(self, ip: str) -> tuple[bool, str | None]:
        now = time.time()
        today = datetime.now(JST).strftime("%Y-%m-%d")
        with self.lock:
            if today != self.day:
                self.day, self.day_count = today, 0
            if self.day_count >= self.per_day:
                return False, "今日はたくさん使われたので、また明日お願いします"
            q = self.hits[ip]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= self.per_ip:
                return False, "少し時間をおいてからもう一度お願いします"
            q.append(now)
            self.day_count += 1
            return True, None
