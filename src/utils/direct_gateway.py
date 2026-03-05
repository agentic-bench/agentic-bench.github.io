import json
import re
import time
import os

from dotenv import load_dotenv
import requests
import tiktoken

load_dotenv()


# Load configuration from local env file (src/utils/.env)
LANYARD_CONFIG_ID = os.getenv("LANYARD_CONFIG_ID")
USE_CASE_ID = os.getenv("USE_CASE_ID")
CLOUD_ID = os.getenv("CLOUD_ID")
BASE_URL = os.getenv("BASE_URL")

# fix API endpoint, through LANYARD proxy
ENDPOINT = f"{BASE_URL}/v2/chat"

# retries and cool down time
RETRY = 3
INTERVAL = 10


def get_ai_gateway_headers() -> str:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Atlassian-CloudId": CLOUD_ID,
        "X-Atlassian-UseCaseId": USE_CASE_ID,
        "X-Atlassian-UserId": os.environ.get("USER", ""),
        "Lanyard-Config": LANYARD_CONFIG_ID,
    }

    return headers


def _estimate_tokens(text: str) -> int:
    """Fallback token estimation using tiktoken (cl100k_base)."""
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to simple heuristic if tiktoken fails
        return max(1, len(text) // 4)


def _send_request(payload: str) -> dict:
    response = requests.request(
        "POST", ENDPOINT, data=payload, headers=get_ai_gateway_headers()
    )

    full_response = json.loads(response.text)
    response_payload = full_response.get("response_payload", {})

    # extract response content
    content = response_payload["choices"][0]["message"]["content"][0]["text"]
    usage = response_payload.get("usage", {})

    return {"content": content, "usage": usage}


def single_request(
    model: str = "gpt-4.1-2025-04-14",
    system_prompt: str = "",
    user_prompt: str = "",
    json_output: bool = False,
    return_usage: bool = False,
) -> any:

    # format payload
    payload = json.dumps(
        {
            "platform_attributes": {"model": model},
            "request_payload": {
                "temperature": 0,
                "messages": [
                    {"content": system_prompt, "role": "system"},
                    {"role": "user", "content": user_prompt},
                ],
            },
        }
    )

    # Check payload size to prevent 413 or context window errors
    # Using 400k chars as a conservative limit for safety
    if len(payload) > 400000:
        print(f"Payload too large ({len(payload)} chars), skipping request.")
        return None

    request_count = 0
    while request_count < RETRY:

        try:
            response_dict = _send_request(payload=payload)
            content = response_dict["content"]
            usage = response_dict.get("usage") or {}

            # Fallback estimation if usage is missing or zero
            if not usage or usage.get("prompt_tokens", 0) == 0:
                usage["prompt_tokens"] = _estimate_tokens(system_prompt + user_prompt)
                usage["completion_tokens"] = _estimate_tokens(content)
                usage["total_tokens"] = (
                    usage["prompt_tokens"] + usage["completion_tokens"]
                )
                usage["is_estimated"] = True

            if return_usage:
                return content, usage
            return content

        except Exception as ex:
            request_count += 1

            print(payload)

            print("Request failed: ", str(ex))
            print(f"Retrying in {INTERVAL} seconds...")
            time.sleep(INTERVAL)

    return None
