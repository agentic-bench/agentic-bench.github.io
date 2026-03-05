# TrajectoryTokenMetrics.py
#
# Description: Calculates trajectory-level cost metrics from token usage.
#              Reads trajectory.input_tokens, trajectory.output_tokens, and
#              trajectory.total_tokens from the trajectory data, then uses
#              genai_price to calculate costs based on the model name.
#
#              This is a trajectory-level evaluator that returns multiple metrics:
#                - trajectory_input_costs
#                - trajectory_output_costs
#                - trajectory_total_costs
#
#              Returns None if model is missing (skip calculation).
#              Raises exception if model name is unknown.

from ..base import BaseEvaluator

try:
    from genai_prices import calc_price, Usage
except ImportError:
    raise ImportError(
        "genai_prices library is required. Install with: pip install genai-prices"
    )


class TrajectoryTokenMetrics(BaseEvaluator):
    evaluation_name = "metric/ops/trajectory_token_metrics"

    # Provider inference mapping based on model name patterns
    PROVIDER_MAP = {
        "gpt": "openai",
        "o1": "openai",
        "o3": "openai",
        "chatgpt": "openai",
        "claude": "anthropic",
        "gemini": "google",
        "mistral": "mistral",
        "codestral": "mistral",
        "deepseek": "deepseek",
        "llama": "meta",
        "qwen": "alibaba",
    }

    @staticmethod
    def infer_provider(model_name: str) -> str:
        """
        Infer the provider from the model name.

        Args:
            model_name: The model name string

        Returns:
            Provider name (e.g., 'openai', 'anthropic')

        Raises:
            ValueError: If model name is not recognized
        """
        model_lower = model_name.lower()

        for pattern, provider in TrajectoryTokenMetrics.PROVIDER_MAP.items():
            if pattern in model_lower:
                return provider

        raise ValueError(
            f"Unknown model '{model_name}'. Cannot infer provider. "
            f"Supported patterns: {list(TrajectoryTokenMetrics.PROVIDER_MAP.keys())}"
        )

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> "dict | None":
        """
        Calculate cost metrics from trajectory token usage.

        Args:
            comment_dict: Dictionary containing trajectory and submission data
            ground_truth: List of ground truth dicts (not used for this evaluator)

        Returns:
            Dict with keys: trajectory_input_costs, trajectory_output_costs, trajectory_total_costs
            Returns None if model is missing from submission
        """
        # Extract submission metadata
        submission = comment_dict.get("submission", {})
        model = submission.get("model")

        # If model is missing, skip calculation (return None for all metrics)
        if not model:
            return {
                "trajectory_input_costs": None,
                "trajectory_output_costs": None,
                "trajectory_total_costs": None,
            }

        # Extract trajectory data
        trajectory = comment_dict.get("trajectory", {})
        input_tokens = trajectory.get("input_tokens")
        output_tokens = trajectory.get("output_tokens")
        total_tokens = trajectory.get("total_tokens")

        # If no token data available, return None for all costs
        if input_tokens is None and output_tokens is None:
            return {
                "trajectory_input_costs": None,
                "trajectory_output_costs": None,
                "trajectory_total_costs": None,
            }

        # Infer provider from model name (raises ValueError if unknown)
        provider = TrajectoryTokenMetrics.infer_provider(model)

        # Calculate costs using genai_prices
        try:
            # Create usage object with token counts
            usage = Usage(
                input_tokens=int(input_tokens) if input_tokens else None,
                output_tokens=int(output_tokens) if output_tokens else None,
            )

            # Calculate price using genai_prices
            price_calc = calc_price(usage=usage, model_ref=model, provider_id=provider)

            # Extract individual costs from the price calculation
            # Note: genai_prices returns Decimal objects, convert to float
            input_cost = (
                float(price_calc.input_price)
                if price_calc.input_price is not None
                else None
            )
            output_cost = (
                float(price_calc.output_price)
                if price_calc.output_price is not None
                else None
            )
            total_cost = (
                float(price_calc.total_price)
                if price_calc.total_price is not None
                else None
            )

            return {
                "trajectory_input_costs": input_cost,
                "trajectory_output_costs": output_cost,
                "trajectory_total_costs": total_cost,
            }

        except Exception as e:
            # Log error and return None for all metrics
            logger = BaseEvaluator.get_json_logger()
            logger.error(
                "Error calculating costs",
                extra={
                    "evaluator": "TrajectoryTokenMetrics",
                    "model": model,
                    "provider": provider,
                    "error": str(e),
                },
            )
            return {
                "trajectory_input_costs": None,
                "trajectory_output_costs": None,
                "trajectory_total_costs": None,
            }
