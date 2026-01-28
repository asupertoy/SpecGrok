#!/usr/bin/env python3
"""
Quick LLM connectivity test.
- Imports `get_llm()` and attempts a short completion call.
- Prints success or detailed error to help diagnose credential/network issues.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ensure project src is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.llm import get_llm


def main():
    try:
        llm = get_llm()
    except Exception as e:
        print(json.dumps({"status": "init_failed", "error": str(e)}))
        return 2

    prompt = "Say 'pong' if you can reach the model. Reply with just the word 'pong'."
    try:
        # Try common APIs
        if hasattr(llm, 'complete'):
            resp = llm.complete(prompt=prompt)
            text = resp if isinstance(resp, str) else getattr(resp, 'choices', [{}])[0].get('text', str(resp))
        elif hasattr(llm, 'chat'):
            resp = llm.chat(prompt)
            text = str(resp)
        elif hasattr(llm, 'generate'):
            resp = llm.generate(prompt)
            text = str(resp)
        else:
            text = "__NO_CALLABLE_METHOD__"

        text = (text or "").strip()
        ok = 'pong' in text.lower()
        print(json.dumps({"status": "ok" if ok else "unexpected_response", "response": text}))
        return 0 if ok else 3
    except Exception as e:
        print(json.dumps({"status": "call_failed", "error": str(e)}))
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
