"""散歩アプリのサーバー。

起動（ローカル）:
    uv run --env-file .env uvicorn api.main:app --reload --port 8080

流れ（/api/route）:
    OSM の候補 → Gemini が番号で選ぶ（一言ヒント付き）
    → Places API で実在・営業を確認（結果は Gemini に渡さない）
    → ルールで除外 → 時間に収まる順番を決める → Routes API で歩くルートを1回取る
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import osm_data, places_check, route, selector
from .llm import LLMError
from .safety import BLOCK_MESSAGE, JST, RateLimiter, decide, is_crisis

log = logging.getLogger("walk")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="walk api")
# /api/route は Places（有料）を呼ぶので厳しめ。上限は環境変数で変えられる
clarify_limiter = RateLimiter(per_ip_per_hour=20, global_per_day=200)
route_limiter = RateLimiter(per_ip_per_hour=int(os.environ.get("ROUTE_PER_IP_HOUR", "6")),
                            global_per_day=int(os.environ.get("ROUTE_DAILY_LIMIT", "30")))
MAX_VERIFY = 6  # Places で確認する場所の上限（1ルートあたりの呼び出し回数の上限）

ATTRIBUTIONS = ["© OpenStreetMap contributors"]
SOURCES = [{"title": "OpenStreetMap", "uri": "https://www.openstreetmap.org/copyright"}]


# ---------- 共通 ----------
def client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return req.client.host if req.client else "unknown"


def err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


def check_limit(req: Request, limiter: RateLimiter) -> JSONResponse | None:
    ok, msg = limiter.allow(client_ip(req))
    return None if ok else err(429, msg or "混み合っています")


# ---------- モデル ----------
class ClarifyReq(BaseModel):
    mood: str = Field(min_length=1, max_length=100)


class RouteReq(BaseModel):
    mood: str = Field(min_length=1, max_length=100)
    preference: str | None = Field(default=None, max_length=60)
    crowd: Literal["avoid", "neutral", "ok"] = "neutral"
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    minutes: int = Field(default=60, ge=15, le=120)
    exclude_ids: list[str] = Field(default_factory=list, max_length=50)
    include_ids: list[str] = Field(default_factory=list, max_length=5)


# ---------- エンドポイント ----------
@app.get("/api/health")
def health():
    return {"ok": True, "places": len(osm_data.all_places())}


@app.post("/api/clarify")
def clarify(body: ClarifyReq, req: Request):
    if is_crisis(body.mood):
        return {"question": "", "options": [], "blocked": True, "message": BLOCK_MESSAGE}
    if (r := check_limit(req, clarify_limiter)) is not None:
        return r
    try:
        c = selector.clarify(body.mood)
    except LLMError as e:
        log.warning("clarify failed: %s", e)
        return err(503, "いま考えるのに時間がかかっています。少ししてからもう一度お願いします")
    return {"question": c.question, "options": [o.model_dump() for o in c.options], "blocked": False}


@app.post("/api/route")
def make_route(body: RouteReq, req: Request):
    if is_crisis(body.mood) or (body.preference and is_crisis(body.preference)):
        return {"blocked": True, "message": BLOCK_MESSAGE, "places": []}
    if (r := check_limit(req, route_limiter)) is not None:
        return r

    t0 = time.time()
    start = (body.lat, body.lon)
    radius = route.search_radius(body.minutes)
    exclude = set(body.exclude_ids)

    candidates = osm_data.inventory(body.lat, body.lon, radius, exclude | set(body.include_ids))
    if len(candidates) < 3:
        return err(404, "この近くには、まだ案内できる場所のデータがありません")

    # 1) Gemini が OSM の候補から選ぶ
    try:
        sel = selector.select(body.mood, body.preference, body.crowd, candidates)
    except LLMError as e:
        log.warning("select failed: %s", e)
        return err(503, "いま考えるのに時間がかかっています。少ししてからもう一度お願いします")
    picked = [{**candidates[p.index], "hint": p.hint} for p in sel.picks]

    # ユーザーが「入れたい」と言った場所
    for pid in body.include_ids:
        p = osm_data.by_id(pid)
        if p:
            d = osm_data.haversine(body.lat, body.lon, p["lat"], p["lon"])
            picked.insert(0, {**p, "dist": round(d), "hint": "あなたが選んだ場所", "_must": True})
    t1 = time.time()

    # 2) 先に OSM の座標で時間に収まる順番を決め、確認するのは実際に回る場所だけ（Places の回数を減らす）
    start_must = {p["id"] for p in picked if p.get("_must")}
    ordered = route.order_and_fit(start, picked, body.minutes, set(start_must))[:MAX_VERIFY]

    # 3) Places で確認（営業時間はお店だけ取る）→ ルールで除外
    now = datetime.now(JST)
    googles = places_check.lookup_many(ordered, [osm_data.category(p) == "shop" for p in ordered])
    kept = []
    for p, g in zip(ordered, googles):
        if p.get("_must"):
            keep, warning = True, None
        else:
            keep, warning = decide(p, g, body.crowd, now)
        if keep:
            kept.append({**p, "_google": g, "warning": warning})
    t2 = time.time()
    if not kept:
        return err(404, "いま開いていて行ける場所が見つかりませんでした。時間や気分を変えてみてください")

    # 4) Routes API（1回）
    ordered = kept
    try:
        rt = route.compute(start, ordered)
    except Exception as e:  # noqa: BLE001
        log.warning("routes failed: %s", e)
        return err(502, "歩くルートを作れませんでした。もう一度お願いします")
    t3 = time.time()
    log.info("route: select %.1fs / places %.1fs / routes %.1fs, %d→%d→%d places",
             t1 - t0, t2 - t1, t3 - t2, len(picked), len(kept), len(ordered))

    places = []
    for p in ordered:
        g = p["_google"] or {}
        places.append({
            "id": p["id"],
            "place_id": g.get("id"),
            "name": p["name"],
            "kind": osm_data.label(p),
            "lat": p["lat"],
            "lon": p["lon"],
            "hint": p["hint"],
            "warning": p["warning"],
            "maps_uri": g.get("googleMapsUri"),
        })
    return {
        "interpretation": sel.interpretation,
        "places": places,
        "polyline": rt["polyline"],
        "total_minutes": rt["total_minutes"],
        "distance_m": rt["distance_m"],
        "sources": SOURCES,
        "attributions": ATTRIBUTIONS,
        "blocked": False,
    }


# ---------- 画面（web/ があれば同じサーバーで配る） ----------
WEB = Path(__file__).resolve().parent.parent / "web"
if WEB.is_dir():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
