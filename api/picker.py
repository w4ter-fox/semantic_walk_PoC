"""Grounding with Google Maps で立ち寄り先を選ぶ（本番でも使う中核処理）

規約上の線引き:
  - Gemini に渡すのは、ユーザーの希望と自分たちで用意したデータ（有名スポットの名前など）だけ
  - Places API の詳細は、アプリでの絞り込みと画面表示にだけ使い、Gemini には渡さない
  - Gemini の生成した理由は、質問したユーザー本人にだけ表示する（共有・保存しない）

改善点（実験で見えた問題への対策）:
  1. 場所は Gemini が書いた名前ではなく、Maps の参照情報の場所IDで扱う
  2. Maps を参照しなかった・数が足りないときは、指示を強めて聞き直す（最大2回）
  3. Places API の詳細で、閉業・徒歩圏外・有名すぎる場所・有名スポットの一部を除外する
  4. 多めに選ばせてから絞り込む

使う環境変数（.env）:
  GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION … Vertex AI
  MAPS_API_KEY … Places API (New)
  PICKER_MODEL（なければ GEMINI_MODEL、なければ gemini-3.5-flash）
  FAMOUS_THRESHOLD（既定 3000）
"""

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass

import requests
from google import genai
from google.genai import errors, types

MODEL = os.environ.get("PICKER_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash"
MAPS_KEY = os.environ.get("MAPS_API_KEY")
FAMOUS = int(os.environ.get("FAMOUS_THRESHOLD", "3000"))
MIN_PLACES = 4      # これ未満なら聞き直す
MAX_ATTEMPTS = 2    # 聞き直しを含めた最大回数
ASK_COUNT = 8       # 多めに選ばせてから絞り込む

# 「代わり」を探したい有名スポット。自分たちで用意したリスト（Maps のデータではない）
FAMOUS_SPOTS = [
    "清水寺", "八坂神社", "高台寺", "知恩院", "伏見稲荷大社", "金閣寺", "鹿苑寺",
    "銀閣寺", "慈照寺", "竹林の小径", "渡月橋", "天龍寺", "二条城", "花見小路",
    "三十三間堂", "平安神宮", "南禅寺", "東福寺", "錦市場",
]

_client = None


def client():
    """.env に GOOGLE_GENAI_USE_VERTEXAI=true があれば Vertex AI 経由になる"""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


@dataclass
class Place:
    place_id: str
    name: str
    reason: str          # Gemini の生成した理由（質問したユーザー本人にだけ表示）
    lat: float
    lon: float
    dist: float
    count: int
    maps_uri: str


@dataclass
class PickResult:
    interpretation: str
    places: list
    dropped: dict        # 除外した理由ごとの件数
    attempts: int
    grounded_chunks: int
    seconds: float


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _norm(s: str) -> str:
    return re.sub(r"[\s・()（）\-]", "", s or "").lower()


def _clean_title(t: str) -> str:
    t = re.sub(r"\s*-\s*Google Maps$", "", t or "")
    return re.sub(r"^Review of\s+", "", t).strip()


def _parse_json(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


# ---------- Gemini + Grounding with Google Maps ----------

PROMPT = """あなたは散歩の案内人です。必ず Google マップのツールで検索し、検索で見つかった場所だけから選んでください。
{retry_note}
出発地から徒歩{walk}分以内（直線でおよそ{radius}m以内）で、次の希望を満たせる立ち寄り先を{n}か所選んでください。
希望: {concept}
{preference}
条件:
- 次の有名な観光地と、その境内・敷地の中にある場所は選ばない: {famous}
- 寺社だけに偏らず、庭、路地、商店街、喫茶店、公園なども含めて種類を混ぜる
- name には Google マップでの名称をそのまま書く

次の形式の JSON だけを返してください（コードブロックや説明文は付けない）:
{{"interpretation": "この場所での希望の解釈（2文程度）",
 "picks": [{{"name": "場所の名称", "reason": "なぜ希望に合うかを一言"}}]}}
"""


def _retrieval_config(lat, lon):
    latlng = types.LatLng(latitude=lat, longitude=lon)
    try:  # 参照情報の名称を日本語で返させる（SDK が対応していなければ外す）
        return types.RetrievalConfig(lat_lng=latlng, language_code="ja")
    except Exception:
        return types.RetrievalConfig(lat_lng=latlng)


def _ask(concept, preference, lat, lon, radius, retry):
    prompt = PROMPT.format(
        retry_note=("前回は検索せずに回答したか、条件に合う場所が足りませんでした。"
                    "今回は必ず Google マップで検索してから選んでください。") if retry else "",
        walk=max(5, round(radius / 80)), radius=radius, n=ASK_COUNT, concept=concept,
        preference=f"補足: {preference}" if preference else "",
        famous="、".join(FAMOUS_SPOTS),
    )
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_maps=types.GoogleMaps())],
        tool_config=types.ToolConfig(retrieval_config=_retrieval_config(lat, lon)),
        temperature=0.3,
    )
    for i in range(3):
        try:
            return client().models.generate_content(model=MODEL, contents=prompt, config=config)
        except (errors.ServerError, errors.ClientError) as e:
            if e.code not in (429, 503) or i == 2:
                raise
            time.sleep(4 * (i + 1))


def _chunks(res):
    out = []
    gm = res.candidates[0].grounding_metadata if res.candidates else None
    for ch in (gm.grounding_chunks or []) if gm else []:
        maps = getattr(ch, "maps", None)
        if not maps:
            continue
        pid = getattr(maps, "place_id", None)
        out.append({"pid": pid.split("/")[-1] if pid else None, "title": _clean_title(maps.title)})
    return out


# ---------- Places API（絞り込みと表示のためだけに使う） ----------

_details_cache: dict = {}
_search_cache: dict = {}


def details(pid):
    if pid in _details_cache:
        return _details_cache[pid]
    try:
        r = requests.get(
            f"https://places.googleapis.com/v1/places/{pid}",
            params={"languageCode": "ja"},
            headers={"X-Goog-Api-Key": MAPS_KEY,
                     "X-Goog-FieldMask": "id,displayName,location,userRatingCount,"
                                         "businessStatus,googleMapsUri"},
            timeout=20,
        )
        r.raise_for_status()
        j = r.json()
        d = {"name": (j.get("displayName") or {}).get("text", ""),
             "lat": j["location"]["latitude"], "lon": j["location"]["longitude"],
             "count": j.get("userRatingCount", 0), "status": j.get("businessStatus", ""),
             "uri": j.get("googleMapsUri", "")}
    except (requests.RequestException, KeyError):
        d = None
    _details_cache[pid] = d
    return d


def search_id(name, lat, lon):
    """Gemini が書いた名称（生成した出力）で検索し、場所IDを得る"""
    key = (name, round(lat, 3), round(lon, 3))
    if key in _search_cache:
        return _search_cache[key]
    pid = None
    try:
        r = requests.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={"X-Goog-Api-Key": MAPS_KEY, "X-Goog-FieldMask": "places.id"},
            json={"textQuery": name, "languageCode": "ja", "pageSize": 1,
                  "locationBias": {"circle": {"center": {"latitude": lat, "longitude": lon},
                                              "radius": 2000.0}}},
            timeout=20,
        )
        r.raise_for_status()
        places = r.json().get("places", [])
        pid = places[0]["id"] if places else None
    except requests.RequestException:
        pass
    _search_cache[key] = pid
    return pid


# ---------- 本体 ----------

def pick_places(concept, lat, lon, radius=1000, preference="") -> PickResult:
    t0 = time.time()
    dropped, places = Counter(), {}
    interpretation, attempts, total_chunks = "", 0, 0

    while attempts < MAX_ATTEMPTS and len(places) < MIN_PLACES:
        res = _ask(concept, preference, lat, lon, radius, retry=attempts > 0)
        attempts += 1
        data = _parse_json(res.text)
        chunks = _chunks(res)
        total_chunks += len(chunks)
        interpretation = interpretation or data.get("interpretation", "")
        picks = data.get("picks", [])
        if not chunks:
            dropped["Mapsを参照しなかった"] += len(picks)
            continue
        chunk_ids = {c["pid"] for c in chunks if c["pid"]}

        for p in picks:
            name = p.get("name", "")
            # 1. 参照情報の名称と照合 → 2. 名前で検索し、参照情報に含まれるIDか確認
            pid = next((c["pid"] for c in chunks if c["pid"] and c["title"] and
                        (_norm(c["title"]) in _norm(name) or _norm(name) in _norm(c["title"]))), None)
            if not pid:
                found = search_id(name, lat, lon)
                if found and (found in chunk_ids or not chunk_ids):
                    pid = found
            if not pid:
                dropped["参照情報にない"] += 1
                continue
            if pid in places:
                continue

            d = details(pid)
            if not d:
                dropped["詳細を取得できない"] += 1
                continue
            if d["status"] and d["status"] != "OPERATIONAL":
                dropped["閉業・休業"] += 1
                continue
            dist = haversine_m(lat, lon, d["lat"], d["lon"])
            if dist > radius * 1.2:
                dropped["徒歩圏外"] += 1
                continue
            if d["count"] >= FAMOUS:
                dropped["有名すぎる"] += 1
                continue
            if any(_norm(f) in _norm(d["name"]) for f in FAMOUS_SPOTS):
                dropped["有名スポットの一部"] += 1
                continue
            places[pid] = Place(pid, d["name"], p.get("reason", ""), d["lat"], d["lon"],
                                dist, d["count"], d["uri"])

    return PickResult(interpretation, list(places.values()), dict(dropped),
                      attempts, total_chunks, round(time.time() - t0, 1))