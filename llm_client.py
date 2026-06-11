import json
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from config import get_config

T = TypeVar('T', bound=BaseModel)

class LLMClient:
    def __init__(self):
        config = get_config()
        self.model = config.vllm_model
        # Use vLLM OpenAI-compatible endpoint with a mock API key
        self.client = AsyncOpenAI(
            base_url=config.vllm_base_url,
            api_key="vllm-mock-key"
        )
        logger.debug(f"LLMClient initialized with base_url={config.vllm_base_url} and model={self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True
    )
    async def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 1000,
        response_format: Optional[dict] = None
    ) -> str:
        """Generates a text completion using system and user prompts."""
        logger.debug("Sending completion request to vLLM.")
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=max_tokens,
                temperature=0.0,  # Greedy decoding for consistency in financial contexts
                response_format=response_format
            )
            
            # Log token usage
            usage = response.usage
            if usage:
                logger.info(
                    f"LLM Generation Complete: Prompt Tokens={usage.prompt_tokens}, "
                    f"Completion Tokens={usage.completion_tokens}, Total Tokens={usage.total_tokens}"
                )
            else:
                logger.info("LLM Generation Complete: Token usage not available in response.")
                
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM Generation failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True
    )
    async def generate_json(
        self,
        system: str,
        user: str,
        schema: Type[T],
        max_tokens: int = 1000
    ) -> T:
        """Generates a structured JSON output conforming to a Pydantic model."""
        logger.debug(f"Sending JSON completion request matching schema: {schema.__name__}")
        
        # In newer OpenAI/vLLM versions, response_format={"type": "json_object"}
        # is supported, but to enforce schema structure, we instruct the model
        # in the system prompt to return JSON and parse the response text.
        # Alternatively, we can pass response_format = {"type": "json_object", "schema": schema.model_json_schema()} if supported.
        # Here we'll pass standard json_object format and parse it to the schema.
        # Let's guide the LLM to output the exact schema format.
        
        enriched_system = (
            f"{system}\n\nYour output must be a valid JSON object matching the following JSON schema:\n"
            f"{json.dumps(schema.model_json_schema())}\n"
            "Return ONLY the raw JSON object, without markdown formatting or code blocks."
        )
        
        response_text = await self.generate(
            system=enriched_system,
            user=user,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        
        # Clean response if LLM returned markdown codeblock format anyway
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            return schema.model_validate_json(cleaned_text)
        except Exception as e:
            logger.error(f"Failed to validate JSON response against schema {schema.__name__}: {str(e)}. Raw text: {response_text}")
            raise ValueError(f"Failed to parse LLM response into schema {schema.__name__}: {str(e)}") from e
