import os
from typing import Optional

from openai import OpenAI
from app.config import load_config


OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg = load_config()
        api_key = cfg.openrouter.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _client


def _candidate_models() -> list[str]:
    cfg = load_config()
    override_model = (cfg.openrouter.model or os.environ.get("OPENROUTER_MODEL", "")).strip()
    if override_model:
        # поддержка одного значения или списка через запятую
        return [m.strip() for m in override_model.split(",") if m.strip()]
    # Набор популярных моделей с тегом :free (наличие может меняться у OpenRouter)
    return [
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemma-2-9b-it:free",
        "qwen/qwen-2-7b-instruct:free",
        "openrouter/auto",
    ]


def generate_reply(user_text: str) -> str:
    client = get_client()
    last_error_text = None
    cfg = load_config()
    system_prompt = (
        cfg.guide_system_prompt
        or os.environ.get("GUIDE_SYSTEM_PROMPT")
        or (
            "Ты — Путешественник-гид. Отвечай кратко, практично, по делу. "
            "Помогай с маршрутами, локациями, логистикой, бюджетом и советами путешественника. "
            "Пиши по-русски, структурируй списками, избегай воды."
        )
    )

    for model_name in _candidate_models():
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.7,
                max_tokens=400,
            )
            choice = response.choices[0].message
            content = getattr(choice, "content", None)
            if content:
                return content
            last_error_text = "Пустой ответ от модели"
        except Exception as e:
            text = str(e)
            last_error_text = text
            # Если недостаточно кредитов — пробуем следующую модель
            if "Insufficient credits" in text or "402" in text:
                continue
            # Прочие ошибки — пробуем следующую, но не прерываемся
            continue

    if last_error_text:
        if "Insufficient credits" in last_error_text or "402" in last_error_text:
            return (
                "Ошибка LLM: на доступных бесплатных моделях недостаточно кредитов. "
                "Попробуйте позже или задайте переменную `OPENROUTER_MODEL` с другой моделью."
            )
        return f"Ошибка LLM: {last_error_text}"
    return "Извините, не удалось получить ответ."


