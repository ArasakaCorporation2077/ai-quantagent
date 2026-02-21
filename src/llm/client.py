"""Unified LLM client supporting OpenAI and Anthropic."""

import logging
import time

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.model = model

        if provider == 'openai':
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == 'anthropic':
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f'Unknown LLM provider: {provider}')

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> str:
        """Generate a response from the LLM. Returns raw text."""
        for attempt in range(max_retries):
            try:
                if self.provider == 'openai':
                    return self._generate_openai(system_prompt, user_prompt, temperature, max_tokens)
                else:
                    return self._generate_anthropic(system_prompt, user_prompt, temperature, max_tokens)
            except Exception as e:
                logger.warning(f'LLM call failed (attempt {attempt+1}): {e}')
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def _generate_openai(self, system_prompt: str, user_prompt: str,
                         temperature: float, max_tokens: int) -> str:
        # GPT-5 series only supports temperature=1, so omit it
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            max_completion_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content if content is not None else ''

    def _generate_anthropic(self, system_prompt: str, user_prompt: str,
                            temperature: float, max_tokens: int) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=temperature,
        )
        return response.content[0].text
