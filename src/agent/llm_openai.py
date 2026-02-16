from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from openai import OpenAI

# Load local .env automatically
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")


def llm_json(model: str, system: str, user: str, schema_wrapper: Dict[str, Any]) -> Dict[str, Any]:
    """
    schema_wrapper must look like:
      {
        "name": "rag_plan",
        "schema": { ... JSON Schema object ... }
      }
    Responses API requires: text.format.{type,name,schema,strict}.  [oai_citation:1‡OpenAI Developers](https://developers.openai.com/api/docs/guides/migrate-to-responses/?utm_source=chatgpt.com)
    """
    name = schema_wrapper["name"]
    schema = schema_wrapper["schema"]

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": name,
                "schema": schema,
                "strict": True,
            }
        },
    )

    return json.loads(resp.output_text)