from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore


CONFIG_PATH = Path("config.toml")


@dataclass(frozen=True)
class TelegramConfig:
    token: str | None


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str | None
    model: str | None


@dataclass(frozen=True)
class AppConfig:
    telegram: TelegramConfig
    openrouter: OpenRouterConfig
    guide_system_prompt: str | None


def _read_toml(path: Path) -> dict:
    if not path.exists() or tomllib is None:
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config() -> AppConfig:
    data = _read_toml(CONFIG_PATH)

    tg = data.get("telegram", {}) if isinstance(data, dict) else {}
    orc = data.get("openrouter", {}) if isinstance(data, dict) else {}
    guide = data.get("guide", {}) if isinstance(data, dict) else {}

    telegram_token = tg.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN")
    openrouter_key = orc.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    openrouter_model = orc.get("model") or os.environ.get("OPENROUTER_MODEL")
    guide_system_prompt = guide.get("system_prompt") or os.environ.get("GUIDE_SYSTEM_PROMPT")

    return AppConfig(
        telegram=TelegramConfig(token=telegram_token),
        openrouter=OpenRouterConfig(api_key=openrouter_key, model=openrouter_model),
        guide_system_prompt=guide_system_prompt,
    )


