import asyncio
import logging
import os
import time
from pathlib import Path
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from handlers import router
from config import config
import indexer
import rag

# Создаем директорию для логов
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Настройка логирования в консоль и файл
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler(log_dir / "bot.log", encoding='utf-8')  # Запись в файл
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("=" * 50)
    logger.info("Bot starting...")
    
    # Инициализация LangSmith трейсинга
    if config.LANGSMITH_TRACING_V2:
        if config.LANGSMITH_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT
            logger.info(f"LangSmith tracing enabled for project: {config.LANGSMITH_PROJECT}")
        else:
            logger.warning("LANGSMITH_TRACING_V2=true but LANGSMITH_API_KEY not set. Tracing disabled.")
    else:
        logger.info("LangSmith tracing disabled (set LANGSMITH_TRACING_V2=true to enable)")
    
    # Индексация при старте
    logger.info("Starting indexing...")
    rag.vector_store = await indexer.reindex_all()
    if rag.vector_store:
        # Инициализируем retriever
        rag.initialize_retriever()
        stats = rag.get_vector_store_stats()
        logger.info(f"Indexing completed successfully: {stats['count']} documents indexed")
    else:
        logger.warning("Indexing completed with no documents - bot will run but cannot answer questions")
    
    # Создаем Bot с увеличенными таймаутами для стабильности
    bot = Bot(
        token=config.TELEGRAM_TOKEN,
        session_timeout=60,  # Увеличенный таймаут сессии
    )
    dp = Dispatcher()
    dp.include_router(router)
    
    logger.info("Starting bot polling...")
    try:
        await dp.start_polling(
            bot,
            allowed_updates=["message", "callback_query"],  # Ограничиваем типы обновлений
            close_bot_session=True
        )
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except TelegramNetworkError as e:
        logger.warning(f"Telegram network error: {e}")
        logger.info("This is usually a temporary network issue. Try restarting the bot.")
    except Exception as e:
        logger.error(f"Bot stopped with error: {e}", exc_info=True)
    finally:
        logger.info("Bot shutdown complete")
        logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())

