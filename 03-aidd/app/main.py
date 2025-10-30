import asyncio

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import CommandStart, Command

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
from app.llm_client import generate_reply
from app.config import load_config

BOT_TOKEN = "7854146507:AAHe76po58T-gyAh-y9HsqQOx3ahR2KCDIM"
BOT_NAME = "Testbot_forAI"


async def main() -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    # file logging (logs/app.log, 1MB, keep 3 files)
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(logs_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level, logging.INFO))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    # Avoid duplicating handlers if re-run in same process
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        root_logger.addHandler(file_handler)
    cfg = load_config()
    token = cfg.telegram.token or BOT_TOKEN
    session = AiohttpSession(timeout=120)
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML), session=session)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def on_start(message: types.Message) -> None:
        logging.info("/start from %s", message.from_user.id if message.from_user else "unknown")
        await message.answer(f"Привет! Я бот {BOT_NAME}.")

    @dp.message(Command("ping"))
    async def on_ping(message: types.Message) -> None:
        logging.info("/ping from %s", message.from_user.id if message.from_user else "unknown")
        await message.answer("pong")

    async def _safe_answer(message: types.Message, text: str) -> None:
        try:
            await message.answer(text, disable_web_page_preview=True, request_timeout=120)
        except Exception:
            logging.exception("Send message failed; retrying once")
            await asyncio.sleep(2)
            try:
                await message.answer(text, disable_web_page_preview=True, request_timeout=120)
            except Exception:
                logging.exception("Send message failed on retry")

    @dp.message(F.text)
    async def on_message(message: types.Message) -> None:
        logging.info("text from %s: %s", message.from_user.id if message.from_user else "unknown", message.text)
        reply_text = await asyncio.to_thread(generate_reply, message.text or "")
        await _safe_answer(message, reply_text)

    # Убедимся, что вебхук снят, иначе polling будет останавливаться
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        logging.exception("Failed to delete webhook")

    while True:
        try:
            await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
            break
        except Exception:
            logging.exception("Polling failed; retrying in 3s")
            # Простая повторная попытка при сетевых сбоях
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())


