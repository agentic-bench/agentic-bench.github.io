import json
import time
import os
import uuid
from urllib.parse import unquote

from httpx import URL, AsyncClient, Request, Response
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

# AI Gateway headers and request preparation
# API Reference: https://developer.atlassian.com/platform/ai-gateway/rest/v2/api-group-v-/

# Original source:
# - https://bitbucket.org/atlassian/acra-python/src/main/packages/code-nemo/src/nemo/providers/unified.py
# - https://bitbucket.org/atlassian/acra-python/src/main/packages/code-nemo/src/nemo/utils/ai_gateway.py

load_dotenv()


# Load configuration from local env file (src/utils/.env)
LANYARD_CONFIG_ID = os.getenv("LANYARD_CONFIG_ID")
USE_CASE_ID = os.getenv("USE_CASE_ID")
CLOUD_ID = os.getenv("CLOUD_ID")
BASE_URL = os.getenv("BASE_URL")


def get_ai_gateway_headers() -> str:
    """Get the LLM client configuration for the given model ID.

    Returns:
        AI Gateway headers.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Atlassian-CloudId": CLOUD_ID,
        "X-Atlassian-UserId": os.environ.get("USER", ""),
        "X-Atlassian-UseCaseId": USE_CASE_ID,
        "Lanyard-Config": LANYARD_CONFIG_ID or "",
    }
    return headers


class UnifiedProviderHttpClient(AsyncClient):

    def __init__(self, *args, **kwargs):
        super().__init__(
            headers=get_ai_gateway_headers(), base_url=BASE_URL, *args, **kwargs
        )
        self.model = None

    def _prep_request(self, req: Request) -> Request:
        """Prepare modified request with unified gateway format"""
        payload = json.loads(req.content)
        self.model = payload.pop("model", None)
        headers = get_ai_gateway_headers() | {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        assert self.model is not None, "Model is required"
        self.platform_attrs = {
            "model": self.model,
        }

        return Request(
            method=req.method,
            url=BASE_URL + "/v2/beta/chat",
            headers=headers,
            json={
                "request_payload": payload,
                "platform_attributes": self.platform_attrs,
            },
        )

    def _process_content(self, content: list) -> str | list:
        """Process response content list into string if it contains text type content"""
        if not content:
            return ""  # Return empty string instead of empty list
        if not isinstance(content[0], dict):
            return content
        if content[0].get("type") != "text":
            raise ValueError(f"Unsupported content type: {content[0].get('type')}")
        return "\n\n".join(c["text"] for c in content)

    def _process_usage(self, platform_attrs: dict) -> dict:
        """Gets usage and returns as an OpenAI response compatible dict"""
        usage_dict = platform_attrs.get("metrics", {}).get("usage", {})
        return dict(
            usage=dict(
                total_tokens=usage_dict.get("total_tokens", 0),
                prompt_tokens=usage_dict.get("input_tokens", 0),
                completion_tokens=usage_dict.get("output_tokens", 0),
            )
        )

    def _prep_response(self, resp: Response, mod_req: Request) -> Response:
        """Prepare modified response in OpenAI format"""
        resp_data = json.loads(resp.content)
        payload = resp_data["response_payload"]

        if self.model:
            if "gpt" in self.model.lower():
                for choice in payload["choices"]:
                    content = choice["message"].get("content", None)
                    if content is not None and isinstance(content, list):
                        choice["message"]["content"] = self._process_content(content)
                    elif content is None:
                        choice["message"]["content"] = ""

            elif "claude" in self.model.lower():
                # print("Processing Claude model response")
                valid_finish_reasons = {
                    "stop",
                    "length",
                    "tool_calls",
                    "content_filter",
                    "function_call",
                }
                for idx, choice in enumerate(payload["choices"]):
                    content = choice["message"].get("content", None)
                    if content is not None and isinstance(content, list):
                        choice["message"]["content"] = self._process_content(content)
                    elif content is None:
                        choice["message"]["content"] = ""
                    
                    # Fix tool_calls format for Claude
                    if "tool_calls" in choice["message"]:
                        tool_calls = choice["message"]["tool_calls"]
                        if tool_calls:
                            for tool_call in tool_calls:
                                # Ensure arguments is a string, not None
                                if tool_call.get("function", {}).get("arguments") is None:
                                    tool_call["function"]["arguments"] = "{}"
                                # Ensure arguments is a string if it's a dict
                                elif isinstance(tool_call["function"]["arguments"], dict):
                                    tool_call["function"]["arguments"] = json.dumps(tool_call["function"]["arguments"])
                    
                    # Ensure index
                    choice["index"] = idx
                    # Fix finish_reason if needed
                    finish_reason = choice.get("finish_reason", "stop")
                    if (
                        finish_reason == "end_turn"
                        or finish_reason not in valid_finish_reasons
                    ):
                        finish_reason = "stop"
                    choice["finish_reason"] = finish_reason

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": payload["choices"],
            **self._process_usage(resp_data.get("platform_attributes", {})),
        }

        return Response(
            status_code=resp.status_code,
            json=response,
            headers=resp.headers,
            extensions=resp.extensions,
            request=mod_req,
        )

    async def send(self, request: Request, *args, **kwargs) -> Response:
        """Send request through unified gateway and process response"""
        mod_req = self._prep_request(request)
        resp = await super().send(mod_req, *args, **kwargs)
        assert (
            resp.status_code == 200
        ), f"Unified AI Gateway Error, with status code {resp.status_code}, and content {resp.content!r}"
        return self._prep_response(resp, mod_req)


class UnifiedProvider(OpenAIProvider):
    """Unified AI Gateway Provider"""

    @property
    def name(self) -> str:
        return "unified"

    def __init__(self, *args, **kwargs):
        super().__init__(
            api_key="NA",
            base_url="NA",
            http_client=UnifiedProviderHttpClient(timeout=900),
            *args,
            **kwargs,
        )
