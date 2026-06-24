"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.core.errors import StudentTodoError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


from openai import OpenAI
import logging
from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Provider-agnostic LLM client implementation using OpenAI/DeepSeek."""

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.compatible_api_key or settings.openai_api_key or "mock-key"
        base_url = settings.compatible_base_url
        self.model = settings.compatible_model_name or settings.openai_model

        logger.info(f"Initializing LLMClient with model: {self.model}, base_url: {base_url}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion and track input/output tokens and cost."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content or ""

            input_tokens = 0
            output_tokens = 0
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

            # Calculate cost
            # DeepSeek V4 Flash or GPT-4o-mini pricing:
            # Let's assume standard API cost: input: $0.15 / 1M, output: $0.60 / 1M.
            input_cost = (input_tokens * 0.15) / 1_000_000
            output_cost = (output_tokens * 0.60) / 1_000_000
            cost_usd = input_cost + output_cost

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        except Exception as e:
            logger.error(f"LLM Client completion error: {e}")
            raise
