# utils/llm_gateway.py

import os
from typing import Union, Tuple, Dict

from utils.direct_gateway import single_request as _direct_single_request
from utils.generic_llm_gateway import single_request as _generic_single_request

# How to choose backend: env var, config file, feature flag, etc.
BACKEND = os.getenv("LLM_GATEWAY_BACKEND", "generic")  # "direct" or "generic"


def single_request(
    model: str = "gpt-4o",
    system_prompt: str = "",
    user_prompt: str = "",
    json_output: bool = False,
    return_usage: bool = False,
    **kwargs,
) -> Union[str, Tuple[str, Dict[str, int]]]:
    """
    Stable public API for LLM calls.

    All code should import from here:
        from utils.llm_gateway import single_request

    Implementation is selected at runtime via BACKEND.
    """
    if BACKEND == "direct":
        # Preserve the old direct_gateway behavior/signature
        return _direct_single_request(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_output=json_output,
            return_usage=return_usage,
            # kwargs ignored because direct_gateway doesn't support them
        )

    elif BACKEND == "generic":
        # Use new generic gateway
        return _generic_single_request(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_output=json_output,
            return_usage=return_usage,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown LLM_GATEWAY_BACKEND: {BACKEND}")
