"""OSM の XML から、サーバーが読む api/data/places.json を作る。

使い方:
    uv run --env-file .env python scripts/build_osm_json.py
    （OSM_FILE 未設定なら data/area_poi.osm を読む）

できたファイルは .gitignore 済み。© OpenStreetMap contributors（ODbL）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import osmnx as ox

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("OSM_FILE", ROOT / "data" / "area_poi.osm"))
OUT = ROOT / "api" / "data" / "places.json"

TAGS = {
    "historic": True,
    "tourism": ["attraction", "museum", "gallery", "artwork", "viewpoint"],
    "amenity": ["cafe", "place_of_worship", "public_bath", "library", "theatre", "arts_centre",
                "marketplace", "fountain", "bar", "pub", "restaurant", "fast_food", "clock"],
    "leisure": ["park", "garden", "playground"],
    "man_made": ["bridge", "tower", "water_tower", "lighthouse"],
    "railway": ["station"],
    "natural": ["water", "tree_row", "wood"],
    "craft": True,
    "shop": True,
}
KIND_KEYS = list(TAGS)


def main() -> None:
    if not SRC.is_absolute():
        src = ROOT / SRC
    else:
        src = SRC
    if not src.exists():
        sys.exit(f"{src} がありません（get_osm.sh で作ってください）")

    gdf = ox.features_from_xml(src, tags=TAGS)
    out, seen = [], set()
    for (element, osmid), row in gdf.iterrows():
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        key = value = None
        for k in KIND_KEYS:
            v = row.get(k)
            if isinstance(v, str) and v:
                if TAGS[k] is True or v in TAGS[k]:
                    key, value = k, v
                    break
        if key is None:
            continue
        pt = row.geometry.representative_point() if row.geometry.geom_type != "Point" else row.geometry
        pid = f"osm:{element}/{osmid}"
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"id": pid, "name": name.strip(), "key": key, "value": value,
                    "lat": round(pt.y, 6), "lon": round(pt.x, 6)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} 件 → {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
