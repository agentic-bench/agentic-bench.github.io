#!/usr/bin/env python3
"""
Generic LLM Gateway using litellm - Drop-in replacement for direct_gateway.

Supports 100+ LLM providers through a unified interface.
API-compatible with direct_gateway for easy switching.

Providers: OpenAI, Anthropic, AWS Bedrock, Google Vertex, Azure, Cohere, Mistral, etc.
"""

import json
import time
from typing import Union, Tuple, Dict, Optional
from dotenv import load_dotenv

try:
    from litellm import completion
except ImportError:
    raise ImportError("litellm not installed. Install with: pip install litellm")

import tiktoken

load_dotenv()

# Configuration
RETRY = 3
INTERVAL = 10


def _estimate_tokens(text: str) -> int:
    """Fallback token estimation using tiktoken."""
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def single_request(
    model: str = "gpt-4.1-2025-04-14",
    system_prompt: str = "",
    user_prompt: str = "",
    json_output: bool = False,
    return_usage: bool = False,
    temperature: float = 0.0,
    **kwargs,
) -> Union[str, Tuple[str, Dict[str, int]]]:
    """
    Make a single LLM request - API-compatible with direct_gateway.

    Drop-in replacement: just change the import!

    Args:
        model: Model name (gpt-4o, claude-3-5-sonnet-20241022, etc.)
        system_prompt: System instruction (optional)
        user_prompt: User prompt/question
        json_output: Request JSON response format
        return_usage: Return (content, usage) tuple
        temperature: Sampling temperature (0.0 to 2.0)
        **kwargs: Additional provider-specific parameters

    Returns:
        str or (str, dict): Response content, optionally with usage stats

    Examples:
        # Before (direct_gateway):
        from utils.direct_gateway import single_request

        # After (generic_llm_gateway):
        from utils.generic_llm_gateway import single_request
        # Rest of code stays the same!

        content = single_request(
            model="gpt-4o",
            system_prompt="You are helpful",
            user_prompt="Explain recursion",
            return_usage=True
        )
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    if not messages:
        return None

    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **kwargs,
    }

    # Handle json_output (compatible with direct_gateway)
    if json_output:
        params["response_format"] = {"type": "json_object"}

    request_count = 0
    while request_count < RETRY:
        try:
            response = completion(**params)
            content = response.choices[0].message.content

            # Get usage stats
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            if return_usage:
                return content, usage
            return content

        except Exception as ex:
            request_count += 1
            print(f"Request failed: {str(ex)}")
            if request_count < RETRY:
                print(f"Retrying in {INTERVAL} seconds...")
                time.sleep(INTERVAL)

    print(f"Failed after {RETRY} attempts")
    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python generic_llm_gateway.py <model> <prompt>")
        print("Example: python generic_llm_gateway.py gpt-4o 'Hello!'")
        sys.exit(1)

    model = sys.argv[1]
    prompt = sys.argv[2]

    print(f"Calling {model}...")
    content, usage = single_request(model=model, user_prompt=prompt, return_usage=True)

    if content:
        print(f"\n{content}")
        print(f"\nUsage: {usage}")
    else:
        print("Request failed")
