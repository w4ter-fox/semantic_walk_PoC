"""OSM から作った場所一覧（api/data/places.json）を読む部分。

places.json は scripts/build_osm_json.py で作る。リポジトリには入れない（.gitignore）が、
Cloud Run へのデプロイ時には .gcloudignore の設定でアップロードされる。
© OpenStreetMap contributors（ODbL）
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

DATA = Path(os.environ.get("PLACES_JSON", Path(__file__).parent / "data" / "places.json"))

# 種類ごとに上限を設けて、カフェや店ばかりにならないようにする
PER_KIND = 6
MAX_ITEMS = 150

KIND_KEYS = ["historic", "tourism", "amenity", "leisure", "man_made", "railway", "natural", "craft", "shop"]

LABELS = {
    ("amenity", "cafe"): "カフェ",
    ("amenity", "place_of_worship"): "寺社・祠",
    ("amenity", "public_bath"): "銭湯",
    ("amenity", "library"): "図書館",
    ("amenity", "theatre"): "劇場",
    ("amenity", "arts_centre"): "アートスペース",
    ("amenity", "marketplace"): "市場",
    ("amenity", "fountain"): "噴水",
    ("amenity", "bar"): "バー",
    ("amenity", "pub"): "パブ",
    ("amenity", "restaurant"): "飲食店",
    ("amenity", "fast_food"): "軽食",
    ("amenity", "clock"): "時計",
    ("leisure", "park"): "公園",
    ("leisure", "garden"): "庭園",
    ("leisure", "playground"): "遊び場",
    ("tourism", "museum"): "博物館",
    ("tourism", "gallery"): "ギャラリー",
    ("tourism", "artwork"): "アート作品",
    ("tourism", "viewpoint"): "眺めのいい場所",
    ("tourism", "attraction"): "見どころ",
    ("man_made", "bridge"): "橋",
    ("man_made", "tower"): "塔",
    ("railway", "station"): "駅",
    ("natural", "water"): "水辺",
    ("natural", "tree_row"): "並木",
    ("natural", "wood"): "林",
}
KEY_LABELS = {"historic": "史跡・碑", "craft": "工房", "shop": "お店", "natural": "自然"}

SHOP_AMENITIES = {"cafe", "public_bath", "bar", "pub", "restaurant", "fast_food", "marketplace", "theatre"}


@lru_cache(maxsize=1)
def all_places() -> list[dict]:
    if not DATA.exists():
        raise FileNotFoundError(
            f"{DATA} がありません。scripts/build_osm_json.py で作ってください。"
        )
    with DATA.open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _index() -> dict[str, dict]:
    return {p["id"]: p for p in all_places()}


def by_id(pid: str) -> dict | None:
    return _index().get(pid)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2点間の距離（メートル）"""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def label(p: dict) -> str:
    return LABELS.get((p["key"], p["value"])) or KEY_LABELS.get(p["key"]) or p["value"]


def category(p: dict) -> str:
    """除外ルールで使う大まかな分類: shop / worship / historic / outdoor / other"""
    k, v = p["key"], p["value"]
    if k in ("shop", "craft") or (k == "amenity" and v in SHOP_AMENITIES):
        return "shop"
    if k == "amenity" and v == "place_of_worship":
        return "worship"
    if k == "historic" or (k == "tourism" and v in ("artwork",)):
        return "historic"
    if k in ("leisure", "natural", "man_made") or (k == "tourism" and v == "viewpoint") or (
        k == "amenity" and v in ("fountain", "clock")
    ):
        return "outdoor"
    return "other"


def inventory(lat: float, lon: float, radius: float, exclude: set[str] | None = None) -> list[dict]:
    """出発地から radius m 以内の候補を、種類ごとに近い順で PER_KIND 件ずつ集める。"""
    exclude = exclude or set()
    near = []
    for p in all_places():
        if p["id"] in exclude or not p.get("name"):
            continue
        d = haversine(lat, lon, p["lat"], p["lon"])
        if d <= radius:
            near.append({**p, "dist": round(d)})
    near.sort(key=lambda p: p["dist"])

    counts: dict[tuple[str, str], int] = {}
    out: list[dict] = []
    for p in near:
        kind = (p["key"], p["value"] if p["key"] != "shop" else "*")
        if counts.get(kind, 0) >= PER_KIND:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        out.append(p)
        if len(out) >= MAX_ITEMS:
            break
    return out
