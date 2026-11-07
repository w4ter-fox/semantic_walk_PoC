"""時間内に回れる順番を決めて、Routes API で歩くルートを1回だけ取る部分。"""
from __future__ import annotations

import os

import requests

from .osm_data import haversine

WALK_M_PER_MIN = 70
DETOUR = 1.3  # 直線距離 → 道のりのおおよその倍率
STAY_MIN = 10
URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELDS = "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"


def search_radius(minutes: int) -> float:
    """候補を集める半径（m）。往復するので歩ける距離の1/3程度。"""
    return min(max(minutes * WALK_M_PER_MIN / 3, 400), 1500)


def _walk_min(a: tuple[float, float], b: tuple[float, float]) -> float:
    return haversine(a[0], a[1], b[0], b[1]) * DETOUR / WALK_M_PER_MIN


def order_and_fit(start: tuple[float, float], places: list[dict], minutes: int, must: set[str] | None = None) -> list[dict]:
    """近い順にたどり、出発地へ戻る時間も含めて minutes に収まるところまで入れる。
    must に入っている id（ユーザーが入れたい場所）は優先して入れる。"""
    must = must or set()
    rest = sorted(places, key=lambda p: p["id"] not in must)  # must を先頭に
    route: list[dict] = []
    here = start
    used = 0.0
    while rest:
        pool = [p for p in rest if p["id"] in must] or rest
        nxt = min(pool, key=lambda p: _walk_min(here, (p["lat"], p["lon"])))
        pos = (nxt["lat"], nxt["lon"])
        cost = _walk_min(here, pos) + STAY_MIN
        if used + cost + _walk_min(pos, start) > minutes and route:
            if nxt["id"] in must:
                must.discard(nxt["id"])  # 時間的に無理なものは諦める
            rest.remove(nxt)
            continue
        route.append(nxt)
        used += cost
        here = pos
        rest.remove(nxt)
    return route


def _ll(lat: float, lon: float) -> dict:
    return {"location": {"latLng": {"latitude": lat, "longitude": lon}}}


def compute(start: tuple[float, float], ordered: list[dict]) -> dict:
    """出発地 → 場所たち → 出発地 の歩くルート。polyline と所要時間（分、滞在込み）を返す。"""
    key = os.environ.get("MAPS_API_KEY")
    if not key:
        raise RuntimeError("MAPS_API_KEY が設定されていません")
    body = {
        "origin": _ll(*start),
        "destination": _ll(*start),
        "intermediates": [_ll(p["lat"], p["lon"]) for p in ordered],
        "travelMode": "WALK",
        "languageCode": "ja-JP",
        "units": "METRIC",
    }
    r = requests.post(URL, json=body, headers={"X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELDS}, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Routes API {r.status_code}: {r.text[:300]}")
    routes = r.json().get("routes") or []
    if not routes:
        raise RuntimeError("歩くルートが見つかりませんでした")
    rt = routes[0]
    walk_sec = int(str(rt.get("duration", "0s")).rstrip("s") or 0)
    return {
        "polyline": rt.get("polyline", {}).get("encodedPolyline", ""),
        "distance_m": rt.get("distanceMeters", 0),
        "total_minutes": round(walk_sec / 60 + STAY_MIN * len(ordered)),
    }
