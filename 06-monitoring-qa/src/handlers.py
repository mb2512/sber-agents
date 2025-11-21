import logging
import asyncio
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import config
import indexer
import rag
# evaluation импортируется лениво только при использовании команды /evaluate_dataset

logger = logging.getLogger(__name__)
router = Router()

# Глобальный словарь для хранения историй диалогов в формате LangChain Messages
chat_conversations: dict[int, list] = {}

async def send_message_with_retry(message: Message, text: str, max_retries: int = 3, **kwargs):
    """
    Отправка сообщения с повторными попытками при сетевых ошибках
    
    Args:
        message: Объект сообщения Telegram
        text: Текст для отправки
        max_retries: Максимальное количество попыток
        **kwargs: Дополнительные параметры для message.answer (например, parse_mode)
    """
    for attempt in range(1, max_retries + 1):
        try:
            await message.answer(text, **kwargs)
            return
        except TelegramRetryAfter as e:
            # Telegram просит подождать
            wait_time = e.retry_after
            logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            continue
        except TelegramNetworkError as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Экспоненциальная задержка: 2, 4, 8 секунд
                logger.warning(
                    f"Telegram network error (attempt {attempt}/{max_retries}): {e}. "
                    f"Retrying in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to send message after {max_retries} attempts: {e}")
                # Не бросаем исключение, чтобы не прерывать работу бота
                return
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            return

@router.message(Command("start"))
async def cmd_start(message: Message):
    logger.info(f"User {message.chat.id} started the bot")
    
    # Инициализируем историю с системным промптом в LangChain формате
    chat_conversations[message.chat.id] = [
        SystemMessage(content=config.SYSTEM_PROMPT)
    ]
    
    await send_message_with_retry(
        message,
        "Привет! Я RAG-ассистент Сбербанка.\n\n"
        "Я могу:\n"
        "• Отвечать на вопросы по документам\n"
        "• Помогать с информацией о кредитах и вкладах\n"
        "• Поддерживать диалог с учетом контекста\n\n"
        "Используйте /help для просмотра всех команд."
    )

@router.message(Command("help"))
async def cmd_help(message: Message):
    logger.info(f"User {message.chat.id} requested help")
    help_text = (
        "🤖 *RAG-ассистент Сбербанка*\n\n"
        "Я помогаю отвечать на вопросы по документам о кредитах и вкладах.\n\n"
        "*Доступные команды:*\n"
        "/start - Начать новый диалог (сбросить историю)\n"
        "/help - Показать эту справку\n"
        "/index - Переиндексировать документы\n"
        "/index\\_status - Проверить статус индексации\n"
        "/evaluate\\_dataset - Оценить качество RAG системы\n\n"
        "*Возможности:*\n"
        "• Ответы на вопросы по документам\n"
        "• Понимание уточняющих вопросов\n"
        "• Сохранение контекста диалога\n"
        "• Оценка качества через RAGAS метрики\n\n"
        "*Примеры вопросов:*\n"
        "• Какие условия потребительского кредита?\n"
        "• Какие проценты по вкладам?\n"
        "• Можно ли досрочно погасить кредит?\n\n"
        "_Если вопрос выходит за рамки документов, я сообщу об этом\\._"
    )
    await send_message_with_retry(message, help_text, parse_mode="Markdown")

@router.message(Command("index"))
async def cmd_index(message: Message):
    logger.info(f"User {message.chat.id} requested reindexing")
    await send_message_with_retry(message, "Начинаю переиндексацию документов...")
    
    try:
        rag.vector_store = await indexer.reindex_all()
        if rag.vector_store:
            rag.initialize_retriever()
            stats = rag.get_vector_store_stats()
            await send_message_with_retry(
                message,
                f"✅ Переиндексация завершена!\n"
                f"Проиндексировано документов: {stats['count']}"
            )
        else:
            await send_message_with_retry(message, "⚠️ Не найдено документов для индексации")
    except Exception as e:
        logger.error(f"Error during reindexing: {e}")
        await send_message_with_retry(message, f"❌ Ошибка при переиндексации: {str(e)}")

@router.message(Command("index_status"))
async def cmd_index_status(message: Message):
    logger.info(f"User {message.chat.id} requested index status")
    stats = rag.get_vector_store_stats()
    
    if stats["status"] == "not initialized":
        await send_message_with_retry(message, "⚠️ Векторное хранилище не инициализировано")
    else:
        await send_message_with_retry(
            message,
            f"📊 Статус индексации:\n"
            f"Статус: {stats['status']}\n"
            f"Количество документов: {stats['count']}"
        )

@router.message(Command("evaluate_dataset"))
async def cmd_evaluate_dataset(message: Message):
    logger.info(f"User {message.chat.id} requested dataset evaluation")
    
    # Проверка API ключа
    if not config.LANGSMITH_API_KEY:
        await send_message_with_retry(
            message,
            "⚠️ LangSmith API key не настроен.\n"
            "Установите LANGSMITH_API_KEY в .env файле для использования evaluation."
        )
        return
    
    # Проверка векторного хранилища
    if rag.vector_store is None or rag.retriever is None:
        await send_message_with_retry(
            message,
            "⚠️ Векторное хранилище не инициализировано.\n"
            "Используйте /index для индексации документов."
        )
        return
    
    # Извлекаем название датасета из команды (опционально)
    command_parts = message.text.split(maxsplit=1)
    dataset_name = command_parts[1] if len(command_parts) > 1 else None
    
    if dataset_name is None:
        dataset_name = config.LANGSMITH_DATASET
        await send_message_with_retry(
            message,
            f"🔍 Начинаю evaluation датасета: {dataset_name}\n\n"
            f"Это может занять несколько минут...\n"
            f"Шаг 1/3: Запуск эксперимента в LangSmith..."
        )
    else:
        await send_message_with_retry(
            message,
            f"🔍 Начинаю evaluation датасета: {dataset_name}\n\n"
            f"Это может занять несколько минут..."
        )
    
    try:
        # Ленивый импорт evaluation (только когда нужен)
        try:
            import evaluation
        except ImportError as e:
            error_msg = f"Не удалось импортировать модуль evaluation: {e}"
            if "git" in str(e).lower():
                error_msg += "\n\n⚠️ Проблема с git: библиотека ragas требует git в системе.\n"
                error_msg += "Установите git и добавьте его в PATH, или установите переменную окружения:\n"
                error_msg += "GIT_PYTHON_REFRESH=quiet"
            logger.error(error_msg)
            await send_message_with_retry(message, f"❌ {error_msg}")
            return
        
        # Запускаем evaluation
        result = evaluation.evaluate_dataset(dataset_name)
        
        # Формируем отчет
        metrics = result["metrics"]
        num_examples = result["num_examples"]
        
        report = (
            f"✅ Evaluation завершен!\n\n"
            f"📊 Датасет: {dataset_name}\n"
            f"📝 Примеров обработано: {num_examples}\n\n"
        )
        
        # Разделяем метрики на Generation и Retrieval
        generation_metrics = {
            "faithfulness": "Обоснованность (нет галлюцинаций)",
            "answer_relevancy": "Релевантность ответа",
            "answer_correctness": "Правильность ответа",
            "answer_similarity": "Похожесть на эталон",
        }
        
        retrieval_metrics = {
            "context_recall": "Полнота контекста (все ли релевантные документы найдены)",
            "context_precision": "Точность поиска (насколько документы релевантны)",
        }
        
        # Добавляем Generation метрики
        report += "🎯 Generation метрики (качество ответов):\n"
        for metric_name, desc in generation_metrics.items():
            if metric_name in metrics:
                score = metrics[metric_name]
                # Эмодзи в зависимости от оценки
                if score >= 0.8:
                    emoji = "🟢"
                elif score >= 0.6:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                report += f"{emoji} {desc}: {score:.3f}\n"
        
        # Добавляем Retrieval метрики
        report += "\n🔍 Retrieval метрики (качество поиска документов):\n"
        for metric_name, desc in retrieval_metrics.items():
            if metric_name in metrics:
                score = metrics[metric_name]
                # Эмодзи в зависимости от оценки
                if score >= 0.8:
                    emoji = "🟢"
                elif score >= 0.6:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                report += f"{emoji} {desc}: {score:.3f}\n"
        
        # Добавляем другие метрики, если есть
        other_metrics = {k: v for k, v in metrics.items() 
                        if k not in generation_metrics and k not in retrieval_metrics}
        if other_metrics:
            report += "\n📈 Другие метрики:\n"
            for metric_name, score in other_metrics.items():
                if score >= 0.8:
                    emoji = "🟢"
                elif score >= 0.6:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                report += f"{emoji} {metric_name}: {score:.3f}\n"
        
        report += f"\n💡 Результаты загружены в LangSmith как feedback"
        
        await send_message_with_retry(message, report)
        logger.info(f"Evaluation completed for user {message.chat.id}")
        
    except ValueError as e:
        logger.error(f"ValueError in evaluation: {e}")
        await send_message_with_retry(message, f"❌ Ошибка: {str(e)}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        error_msg = f"❌ Ошибка: {str(e)}"
        if "git" in str(e).lower():
            error_msg += "\n\n💡 Подсказка: Установите git или установите переменную окружения GIT_PYTHON_REFRESH=quiet"
        await send_message_with_retry(message, error_msg)

@router.message()
async def handle_message(message: Message):
    # Игнорируем сообщения без текста (стикеры, фото и т.д.)
    if not message.text:
        await send_message_with_retry(message, "Извините, я работаю только с текстовыми сообщениями.")
        return
    
    logger.info(f"Message from {message.chat.id}: {message.text[:100]}...")
    
    # Инициализируем историю если её нет
    if message.chat.id not in chat_conversations:
        chat_conversations[message.chat.id] = [
            SystemMessage(content=config.SYSTEM_PROMPT)
        ]
    
    # Добавляем сообщение пользователя в историю
    chat_conversations[message.chat.id].append(
        HumanMessage(content=message.text)
    )
    
    try:
        # Проверка инициализации векторного хранилища
        if rag.vector_store is None or rag.retriever is None:
            logger.warning(f"Vector store not initialized for chat {message.chat.id}")
            await send_message_with_retry(
                message,
                "⚠️ Векторное хранилище не инициализировано. "
                "Пожалуйста, подождите или используйте /index для индексации."
            )
            # Удаляем последнее сообщение из истории
            chat_conversations[message.chat.id].pop()
            return
        
        # Получаем ответ через RAG (передаем историю без system message)
        # Теперь возвращает dict с answer и documents
        result = await rag.rag_answer(chat_conversations[message.chat.id][1:])
        answer = result["answer"]
        documents = result["documents"]
        
        # Добавляем ответ в историю
        chat_conversations[message.chat.id].append(
            AIMessage(content=answer)
        )
        
        # Формируем итоговый ответ с источниками если включено
        final_response = answer
        if config.SHOW_SOURCES and documents:
            sources = rag.format_sources(documents)
            if sources:
                final_response = f"{answer}\n\n{sources}"
        
        # Отправка ответа с обработкой сетевых ошибок
        await send_message_with_retry(message, final_response)
        
    except ValueError as e:
        logger.error(f"ValueError in handle_message for chat {message.chat.id}: {e}")
        # Удаляем последнее сообщение из истории
        chat_conversations[message.chat.id].pop()
        await send_message_with_retry(
            message,
            "⚠️ Векторное хранилище не готово. "
            "Используйте /index для индексации документов."
        )
    except Exception as e:
        logger.error(f"Error in handle_message for chat {message.chat.id}: {e}", exc_info=True)
        # Удаляем последнее сообщение из истории
        chat_conversations[message.chat.id].pop()
        await send_message_with_retry(
            message,
            "Произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )

