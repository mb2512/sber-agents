import logging
import asyncio
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import config
from indexer_with_json import reindex_all
import rag

logger = logging.getLogger(__name__)
router = Router()

# Глобальный словарь для хранения историй диалогов в формате LangChain Messages
chat_conversations: dict[int, list] = {}

@router.message(Command("start"))
async def cmd_start(message: Message):
    logger.info(f"User {message.chat.id} started the bot")
    
    # Инициализируем историю с системным промптом в LangChain формате
    chat_conversations[message.chat.id] = [
        SystemMessage(content=config.SYSTEM_PROMPT)
    ]
    
    await message.answer(
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
        "/index\\_status - Проверить статус индексации\n\n"
        "*Возможности:*\n"
        "• Ответы на вопросы по документам\n"
        "• Понимание уточняющих вопросов\n"
        "• Сохранение контекста диалога\n\n"
        "*Примеры вопросов:*\n"
        "• Какие условия потребительского кредита?\n"
        "• Какие проценты по вкладам?\n"
        "• Можно ли досрочно погасить кредит?\n\n"
        "_Если вопрос выходит за рамки документов, я сообщу об этом\\._"
    )
    await message.answer(help_text, parse_mode="Markdown")

@router.message(Command("index"))
async def cmd_index(message: Message):
    logger.info(f"User {message.chat.id} requested reindexing")
    await message.answer("Начинаю переиндексацию документов...")
    
    try:
        rag.vector_store = await reindex_all()
        if rag.vector_store:
            rag.initialize_retriever()
            stats = rag.get_vector_store_stats()
            await message.answer(
                f"✅ Переиндексация завершена!\n"
                f"Проиндексировано документов: {stats['count']}"
            )
        else:
            await message.answer("⚠️ Не найдено документов для индексации")
    except Exception as e:
        logger.error(f"Error during reindexing: {e}")
        await message.answer(f"❌ Ошибка при переиндексации: {str(e)}")

@router.message(Command("index_status"))
async def cmd_index_status(message: Message):
    logger.info(f"User {message.chat.id} requested index status")
    stats = rag.get_vector_store_stats()
    
    if stats["status"] == "not initialized":
        await message.answer("⚠️ Векторное хранилище не инициализировано")
    else:
        await message.answer(
            f"📊 Статус индексации:\n"
            f"Статус: {stats['status']}\n"
            f"Количество документов: {stats['count']}"
        )

@router.message()
async def handle_message(message: Message):
    # Игнорируем сообщения без текста (стикеры, фото и т.д.)
    if not message.text:
        await message.answer("Извините, я работаю только с текстовыми сообщениями.")
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
            await message.answer(
                "⚠️ Векторное хранилище не инициализировано. "
                "Пожалуйста, подождите или используйте /index для индексации."
            )
            # Удаляем последнее сообщение из истории
            chat_conversations[message.chat.id].pop()
            return
        
        # Отправляем индикатор "печатает..." (не критично, если не отправится)
        try:
            await asyncio.wait_for(
                message.bot.send_chat_action(message.chat.id, "typing"),
                timeout=5  # Короткий таймаут для chat action
            )
        except Exception as e:
            logger.debug(f"Could not send chat action (non-critical): {e}")
            # Продолжаем работу, даже если chat action не отправился
        
        # Получаем ответ через RAG с таймаутом (передаем историю без system message)
        # Увеличиваем таймаут, так как RAG включает несколько запросов к API
        try:
            response = await asyncio.wait_for(
                rag.rag_answer(chat_conversations[message.chat.id][1:]),
                timeout=config.REQUEST_TIMEOUT * 3 + 20  # Увеличенный таймаут для RAG (query transform + retrieval + generation)
            )
            
            # Добавляем ответ в историю
            chat_conversations[message.chat.id].append(
                AIMessage(content=response)
            )
            
            # Отправляем ответ с обработкой ошибок сети и повторными попытками
            max_retries = 3
            retry_delay = 2  # Увеличиваем задержку между попытками
            sent = False
            
            for attempt in range(max_retries):
                try:
                    await message.answer(response)
                    sent = True
                    logger.info(f"✅ Message sent successfully (attempt {attempt + 1})")
                    break  # Успешно отправлено
                except Exception as send_error:
                    error_msg = str(send_error)
                    logger.warning(f"⚠️ Error sending message (attempt {attempt + 1}/{max_retries}): {error_msg[:200]}")
                    
                    if attempt < max_retries - 1:
                        # Ждем перед повторной попыткой (увеличиваем задержку с каждой попыткой)
                        wait_time = retry_delay * (attempt + 1)
                        logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        # Последняя попытка - отправляем сообщение об ошибке
                        logger.error(f"❌ Failed to send message after {max_retries} attempts")
                        try:
                            # Пытаемся отправить короткое сообщение об ошибке
                            await asyncio.wait_for(
                                message.answer(
                                    "⚠️ Произошла ошибка при отправке ответа. "
                                    "Ответ был сгенерирован, но не удалось его отправить. "
                                    "Попробуйте задать вопрос еще раз."
                                ),
                                timeout=10
                            )
                        except Exception as e:
                            logger.error(f"Failed to send error message to user: {e}")
                        # Не поднимаем исключение, чтобы не прерывать обработку
            
            if not sent:
                logger.error(f"❌ Could not send message to user {message.chat.id} after all retries")
        except asyncio.TimeoutError:
            logger.error(f"Timeout error in handle_message for chat {message.chat.id}")
            chat_conversations[message.chat.id].pop()
            await message.answer(
                "⏱️ Превышено время ожидания ответа. "
                "Попробуйте задать вопрос еще раз или упростить запрос."
            )
        
    except ValueError as e:
        logger.error(f"ValueError in handle_message for chat {message.chat.id}: {e}")
        # Удаляем последнее сообщение из истории
        if message.chat.id in chat_conversations:
            chat_conversations[message.chat.id].pop()
        await message.answer(
            "⚠️ Векторное хранилище не готово. "
            "Используйте /index для индексации документов."
        )
    except Exception as e:
        logger.error(f"Error in handle_message for chat {message.chat.id}: {e}", exc_info=True)
        # Удаляем последнее сообщение из истории
        if message.chat.id in chat_conversations:
            chat_conversations[message.chat.id].pop()
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            await message.answer(
                "⏱️ Превышено время ожидания ответа от API. "
                "Попробуйте еще раз через несколько секунд."
            )
        else:
            await message.answer(
                "Произошла ошибка при обработке вашего сообщения. "
                "Попробуйте еще раз или используйте /start для начала нового диалога."
            )

