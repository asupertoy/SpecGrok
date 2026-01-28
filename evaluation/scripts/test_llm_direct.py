#!/usr/bin/env python3
"""
Direct LLM connectivity test using OpenAI-compatible API.
Based on the official DashScope example.
"""
import os
import sys
from pathlib import Path
from openai import OpenAI

# ensure project src is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import settings

def main():
    try:
        client = OpenAI(
            api_key=settings.ALI_API_KEY,
            base_url=settings.ALI_API_URL,
        )
    except Exception as e:
        print(f"Failed to create OpenAI client: {e}")
        return 1

    messages = [{"role": "user", "content": "你是谁"}]
    try:
        completion = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            extra_body={"enable_thinking": True},
            stream=True
        )
        is_answering = False  # 是否进入回复阶段
        print("\n" + "=" * 20 + "思考过程" + "=" * 20)
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    print(delta.reasoning_content, end="", flush=True)
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                    is_answering = True
                print(delta.content, end="", flush=True)
        print("\nTest successful!")
        return 0
    except Exception as e:
        print(f"Call failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())