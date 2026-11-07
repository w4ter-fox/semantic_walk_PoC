"""Gemini に「問いかけ」を作らせる部分と、OSM の候補から場所を選ばせる部分。

Gemini に渡すのは OSM の情報（名前・種類・距離）だけ。Google Maps / Places の結果は渡さない。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .llm import ask_json
from .osm_data import label
from .safety import famous_names

Crowd = Literal["avoid", "neutral", "ok"]


# ---------- /api/clarify ----------
class Option(BaseModel):
    label: str = Field(description="選択肢。20文字以内。気分の質だけを言い換える")
    crowd: Crowd = Field(description="この選択肢を選んだ人が人混みを避けたいか: avoid / neutral / ok")


class Clarify(BaseModel):
    question: str = Field(description="問いかけ。30文字以内")
    options: list[Option] = Field(min_length=3, max_length=4)


def clarify(mood: str) -> Clarify:
    prompt = f"""あなたは散歩ルートを作るアプリの聞き役です。
ユーザーが入力した気分は、まだぼんやりしています。どんな質の気分なのかを確かめるための問いかけを1つ作り、
選択肢を3〜4個出してください。

# ルール
- 選択肢は「気分の質」だけを言い換える（例: 「泣きたい」→「思いきり泣ける静かな所」「泣いたあとすっきりしたい」）
- 具体的な場所の名前・ジャンル（カフェ、神社など）を選択肢に出さない
- 診断や助言はしない。やさしく短く
- 各選択肢に、その気分なら人混みをどう感じそうか crowd を付ける（avoid=避けたい / neutral=どちらでも / ok=人がいてもいい）

# ユーザーの気分
{mood}
"""
    return ask_json(prompt, Clarify, temperature=0.7)


# ---------- /api/route の場所選び ----------
class Pick(BaseModel):
    index: int = Field(description="候補一覧の番号")
    hint: str = Field(description="その場所で何を感じられそうかの一言。30文字以内。事実や歴史は書かない")


class Selection(BaseModel):
    interpretation: str = Field(description="気分をどう受け取ったか。60文字以内")
    picks: list[Pick] = Field(min_length=1, max_length=10)


def select(mood: str, preference: str | None, crowd: str, candidates: list[dict], n: int = 8) -> Selection:
    lines = [f"{i}. {p['name']}（{label(p)}・{p['dist']}m）" for i, p in enumerate(candidates)]
    crowd_rule = {
        "avoid": "人混みを避けたい。次の有名な場所やその境内・周辺の一部は選ばない: " + "、".join(famous_names()),
        "neutral": "人の多さは気にしない。",
        "ok": "人がいる賑やかな場所でもかまわない。",
    }[crowd]
    prompt = f"""あなたは散歩ルートを作るアプリです。ユーザーの気分に合う場所を、下の候補一覧から {n} 個前後選んでください。

# ユーザーの気分
{mood}
{f"（さらに選んだ方向: {preference}）" if preference else ""}

# ルール
- 候補一覧にある場所だけを番号で選ぶ。一覧にない場所を作らない
- 一般に公開されていない場所（非公開の寺院の塔頭、私有地、学校、住宅など）は選ばない
- {crowd_rule}
- 同じ種類ばかりにしない（寺社だけ、カフェだけ、にしない）
- hint は「ここで何を感じられそうか」を30文字以内の一言で。名前の由来・歴史・営業時間などの事実は書かない（不確かなことを書かないため）
- interpretation は気分をどう受け取ったかを60文字以内で

# 候補一覧（OpenStreetMap より）
{chr(10).join(lines)}
"""
    sel = ask_json(prompt, Selection)
    seen: set[int] = set()
    sel.picks = [p for p in sel.picks if 0 <= p.index < len(candidates) and not (p.index in seen or seen.add(p.index))]
    for p in sel.picks:
        p.hint = p.hint[:40]
    return sel
