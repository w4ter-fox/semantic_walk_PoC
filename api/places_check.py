"""Places API (New) で「本当にあるか・いま開いているか」を確かめる部分。

ここで取った情報は除外ルール（safety.decide）と地図リンクにだけ使い、Gemini には渡さない。
"""
from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import requests

from .osm_data import haversine

log = logging.getLogger("walk")
URL = "https://places.googleapis.com/v1/places:searchText"
# 料金はフィールドで決まる。営業時間を取ると Enterprise（無料枠 1,000回/月）、
# 取らなければ Pro（無料枠 5,000回/月）。営業時間が要るのはお店だけなので分ける。
_BASE = ["places.id", "places.displayName", "places.location", "places.businessStatus",
         "places.googleMapsUri", "places.types"]
FIELDS_PRO = ",".join(_BASE)
FIELDS_ENTERPRISE = ",".join(_BASE + ["places.currentOpeningHours.openNow"])
MATCH_DIST_M = 300
CACHE_TTL_SEC = 3600  # 規約上 Places の内容は長く保存しない。同じ場所の連続確認を防ぐ程度

_cache: dict[tuple[str, bool], tuple[float, dict | None]] = {}


def enabled() -> bool:
    """PLACES_CHECK=off にすると Places を呼ばない（開発中の節約用）。"""
    return os.environ.get("PLACES_CHECK", "on").lower() != "off"


def _key() -> str:
    k = os.environ.get("MAPS_API_KEY")
    if not k:
        raise RuntimeError("MAPS_API_KEY が設定されていません")
    return k


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[（(].*?[)）]", "", s)
    return re.sub(r"[\s・\-‐ー〜~]", "", s)


def _same_name(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    return bool(a and b) and (a in b or b in a)


def lookup(place: dict, need_hours: bool = False) -> dict | None:
    """OSM の場所を Places で探す。見つからなければ None。"""
    if not enabled():
        return None
    ck = (place["id"], need_hours)
    hit = _cache.get(ck)
    if hit and time.time() - hit[0] < CACHE_TTL_SEC:
        return hit[1]
    body = {
        "textQuery": place["name"],
        "languageCode": "ja",
        "maxResultCount": 3,
        "locationBias": {"circle": {
            "center": {"latitude": place["lat"], "longitude": place["lon"]},
            "radius": float(MATCH_DIST_M),
        }},
    }
    headers = {"X-Goog-Api-Key": _key(),
               "X-Goog-FieldMask": FIELDS_ENTERPRISE if need_hours else FIELDS_PRO}
    found = None
    try:
        r = requests.post(URL, json=body, headers=headers, timeout=8)
        if r.status_code != 200:
            log.warning("Places API %s: %s", r.status_code, r.text[:300])
            return None  # 失敗はキャッシュしない
        for g in r.json().get("places", []):
            loc = g.get("location", {})
            name = g.get("displayName", {}).get("text", "")
            if not loc or not _same_name(name, place["name"]):
                continue
            if haversine(place["lat"], place["lon"], loc["latitude"], loc["longitude"]) <= MATCH_DIST_M:
                found = g
                break
    except requests.RequestException as e:
        log.warning("Places API 通信失敗: %s", e)
        return None  # 通信失敗はキャッシュしない
    _cache[ck] = (time.time(), found)
    return found


def lookup_many(places: list[dict], need_hours: list[bool]) -> list[dict | None]:
    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(lookup, places, need_hours))
