"""Gemini（Vertex AI 経由）を呼ぶ部分。

- 環境変数 GOOGLE_GENAI_USE_VERTEXAI=true / GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION で Vertex を使う
- 入れ子の response_schema で 503 が出たことがあるので、スキーマはプロンプトに書き、
  返ってきた JSON を pydantic で検証する
- 429 / 503 のときはモデルを切り替えて再試行する
"""
from __future__ import annotations

import json
import os
import time
from typing import Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

MODELS = [
    os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
    "gemini-3.1-flash-lite",
]

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


class LLMError(RuntimeError):
    pass


def _retryable(e: Exception) -> bool:
    s = str(e)
    return any(code in s for code in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))


def ask_json(prompt: str, schema: Type[T], temperature: float = 0.4) -> T:
    """prompt を投げて、schema の形の JSON を受け取る。"""
    full = (
        prompt
        + "\n\n# 出力形式\n次の JSON Schema に従う JSON だけを出力すること。説明文やコードブロックは付けない。\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=temperature,
    )
    last: Exception | None = None
    for model in dict.fromkeys(MODELS):  # 重複を除いて順番どおり
        for attempt in range(2):
            try:
                res = client().models.generate_content(model=model, contents=full, config=cfg)
                text = (res.text or "").strip()
                if text.startswith("```"):
                    text = text.strip("`").removeprefix("json").strip()
                return schema.model_validate_json(text)
            except ValidationError as e:
                last = e  # 形が崩れた → 同じモデルでもう一度
                continue
            except Exception as e:  # noqa: BLE001
                last = e
                if _retryable(e):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise LLMError(str(e)) from e
    raise LLMError(f"Gemini の呼び出しに失敗しました: {last}")
