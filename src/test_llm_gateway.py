#!/usr/bin/env python3
"""
Test script for utils.llm_gateway

Usage examples:
    # OpenAI
    python test_llm_gateway.py --provider openai --model gpt-4o

    # Anthropic
    python test_llm_gateway.py --provider anthropic --model claude-3-5-sonnet-20241022

    # AWS Bedrock
    python test_llm_gateway.py --provider bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0 --region us-east-1

    # Google Vertex AI
    python test_llm_gateway.py --provider vertex --model gemini-1.5-pro --project my-project --location us-central1
"""

from utils.llm_gateway import single_request


def main():

    # Simple test prompts
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of France? Answer in one word."

    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}\n")

    # Common kwargs for llm_gateway.single_request
    kwargs = {
        "model": "gpt-4.1-2025-04-14",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "return_usage": True,
    }

    # Make request via the facade
    result = single_request(**kwargs)

    if result:
        content, usage = result
        print(f"✅ Response: {content}")
        print("\n📊 Usage:")
        print(f"  Input tokens:  {usage.get('prompt_tokens', 0)}")
        print(f"  Output tokens: {usage.get('completion_tokens', 0)}")
        print(f"  Total tokens:  {usage.get('total_tokens', 0)}")
        if usage.get("is_estimated"):
            print("  ⚠️  (estimated)")
    else:
        print("❌ Request failed")


if __name__ == "__main__":
    main()
