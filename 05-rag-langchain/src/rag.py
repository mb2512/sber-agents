import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from config import config

logger = logging.getLogger(__name__)

# Глобальное векторное хранилище
vector_store = None
retriever = None

# Кеши для промптов и LLM клиентов
_conversational_answering_prompt = None
_retrieval_query_transform_prompt = None
_llm_query_transform = None
_llm = None

def initialize_retriever():
    """Инициализация retriever из векторного хранилища"""
    global retriever
    if vector_store is None:
        logger.error("Cannot initialize retriever: vector_store is None")
        return False
    
    retriever = vector_store.as_retriever(search_kwargs={'k': config.RETRIEVER_K})
    logger.info(f"Retriever initialized with k={config.RETRIEVER_K}")
    return True

def format_chunks(chunks):
    """
    Форматирование чанков с метаданными для лучшей прозрачности
    """
    if not chunks:
        return "Нет доступной информации"
    
    formatted_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Получаем метаданные
        source = chunk.metadata.get('source', 'Unknown')
        page = chunk.metadata.get('page', 'N/A')
        
        # Извлекаем имя файла из пути
        source_name = source.split('/')[-1] if '/' in source else source
        
        # Форматируем чанк
        formatted_parts.append(
            f"[Источник {i}: {source_name}, стр. {page}]\n{chunk.page_content}"
        )
    
    return "\n\n---\n\n".join(formatted_parts)

def _load_prompts():
    """Ленивая загрузка промптов с обработкой ошибок"""
    global _conversational_answering_prompt, _retrieval_query_transform_prompt
    
    if _conversational_answering_prompt is not None:
        return _conversational_answering_prompt, _retrieval_query_transform_prompt
    
    try:
        conversation_system_text = config.load_prompt(config.CONVERSATION_SYSTEM_PROMPT_FILE)
        query_transform_text = config.load_prompt(config.QUERY_TRANSFORM_PROMPT_FILE)
        
        # Создаем промпт с поддержкой переменной context
        # Используем ChatPromptTemplate напрямую со списком кортежей (как в примере из notebook)
        # Это должно правильно обработать переменную {context} из RunnablePassthrough.assign
        _conversational_answering_prompt = ChatPromptTemplate(
            [
                ("system", conversation_system_text),  # Строка с {context} будет обработана
                ("placeholder", "{messages}")  # Используем placeholder вместо MessagesPlaceholder
            ]
        )
        
        _retrieval_query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                ("user", query_transform_text),
            ]
        )
        
        logger.info("Prompts loaded successfully")
        return _conversational_answering_prompt, _retrieval_query_transform_prompt
        
    except FileNotFoundError as e:
        logger.error(f"Prompt file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompts: {e}", exc_info=True)
        raise

def _get_llm_query_transform():
    """Ленивая инициализация LLM для query transformation с кешированием"""
    global _llm_query_transform
    if _llm_query_transform is None:
        _llm_query_transform = ChatOpenAI(
            model=config.MODEL_QUERY_TRANSFORM,
            temperature=0.4,
            openai_api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=2
        )
        logger.info(f"Query transform LLM initialized: {config.MODEL_QUERY_TRANSFORM}")
    return _llm_query_transform

def _get_llm():
    """Ленивая инициализация основной LLM с кешированием"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=config.MODEL,
            temperature=0.9,
            openai_api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=2
        )
        logger.info(f"Main LLM initialized: {config.MODEL}")
    return _llm

def get_retrieval_query_transformation_chain():
    """Цепочка трансформации запроса"""
    _, retrieval_query_transform_prompt = _load_prompts()
    return (
        retrieval_query_transform_prompt
        | _get_llm_query_transform()
        | StrOutputParser()
    )

def get_rag_chain():
    """Финальная RAG-цепочка с query transformation"""
    if retriever is None:
        raise ValueError("Retriever not initialized")
    
    conversational_answering_prompt, _ = _load_prompts()
    
    def format_docs(docs):
        """Форматирование документов для контекста"""
        return format_chunks(docs)
    
    def log_prompt_input(input_dict):
        """Логирование входных данных для промпта (для отладки)"""
        context = input_dict.get("context", "NO CONTEXT!")
        messages_count = len(input_dict.get("messages", []))
        # Используем INFO для важных сообщений отладки
        logger.info(f"🔍 Prompt input - context length: {len(context) if context != 'NO CONTEXT!' else 0}, messages: {messages_count}")
        if context == "NO CONTEXT!":
            logger.error("❌ CONTEXT IS MISSING! This is the problem!")
        else:
            logger.info(f"✅ Context preview: {context[:500]}...")
        return input_dict
    
    # Создаем цепочку: трансформация запроса -> поиск -> форматирование -> ответ
    retrieval_chain = (
        get_retrieval_query_transformation_chain() 
        | retriever 
        | format_docs
    )
    
    def log_formatted_messages(formatted_messages):
        """Логируем отформатированные сообщения перед отправкой в LLM"""
        if hasattr(formatted_messages, 'messages'):
            for i, msg in enumerate(formatted_messages.messages):
                if hasattr(msg, 'content'):
                    content_preview = str(msg.content)[:500] if msg.content else "Empty"
                    logger.info(f"📤 Message {i} to LLM ({type(msg).__name__}): {content_preview}...")
                    if "context" in content_preview.lower() and "контекст" not in content_preview.lower():
                        logger.warning("⚠️ Context variable might not be substituted!")
        return formatted_messages
    
    return (
        RunnablePassthrough.assign(context=retrieval_chain)
        | log_prompt_input  # Логируем, что передается в промпт
        | conversational_answering_prompt
        | log_formatted_messages  # Логируем отформатированные сообщения
        | _get_llm()
        | StrOutputParser()
    )

async def rag_answer(messages):
    """
    Получить ответ от RAG с учетом истории диалога
    
    Args:
        messages: список LangChain messages (HumanMessage, AIMessage)
    
    Returns:
        str: ответ от RAG
    """
    if vector_store is None or retriever is None:
        logger.error("Vector store or retriever not initialized")
        raise ValueError("Векторное хранилище не инициализировано. Запустите индексацию.")
    
    # Логируем последний вопрос для отладки
    last_message = messages[-1].content if messages else "No messages"
    logger.info(f"Processing RAG query: {last_message[:100]}...")
    
    try:
        # Для отладки: проверяем, что retriever работает
        transform_chain = get_retrieval_query_transformation_chain()
        transformed_query = await transform_chain.ainvoke({"messages": messages})
        logger.info(f"Transformed query: {transformed_query[:200]}...")
        
        # Сначала пытаемся найти точное или частичное совпадение по question в метаданных
        # Это быстрее и точнее для вопросов из JSON
        exact_match_chunk = None
        query_lower = last_message.lower().strip()
        query_words = set(query_lower.split())
        
        if vector_store:
            try:
                # Ищем точное совпадение по оригинальному запросу (не transformed_query!)
                # Используем большой k, чтобы найти все возможные совпадения
                logger.info(f"🔍 Searching for exact match by question metadata (query: '{query_lower}')...")
                
                # Способ 1: ищем через similarity_search с оригинальным запросом и большим k
                search_chunks = await vector_store.asimilarity_search(query_lower, k=200)
                logger.info(f"🔍 Got {len(search_chunks)} chunks from similarity search with original query")
                
                for chunk in search_chunks:
                    question = chunk.metadata.get('question', '').lower().strip()
                    if question:
                        # Точное совпадение (с учетом знаков препинания)
                        question_normalized = question.rstrip('?').rstrip('!').rstrip('.')
                        query_normalized = query_lower.rstrip('?').rstrip('!').rstrip('.')
                        if question_normalized == query_normalized or question == query_lower:
                            exact_match_chunk = chunk
                            logger.info(f"✅ Found exact match by question: '{question}'")
                            break
                
                # Способ 2: если не нашли точное, ищем частичное совпадение
                if not exact_match_chunk:
                    logger.info(f"🔍 Searching for partial match...")
                    for chunk in search_chunks:
                        question = chunk.metadata.get('question', '').lower().strip()
                        if question:
                            # Частичное совпадение - проверяем, содержит ли question ключевые слова запроса
                            # Исключаем служебные слова
                            stop_words = {'как', 'что', 'где', 'когда', 'кто', 'почему', 'зачем', 'для', 'нужны', 'нужен', 'нужна', 'нужно', 'чтобы', 'можно', 'можно', 'ли'}
                            query_words_filtered = {w for w in query_words if w not in stop_words and len(w) > 2}
                            question_words = set(question.split())
                            question_words_filtered = {w for w in question_words if w not in stop_words and len(w) > 2}
                            
                            # Если больше 70% значимых слов запроса есть в question
                            if len(query_words_filtered) > 0:
                                common_words = query_words_filtered & question_words_filtered
                                match_ratio = len(common_words) / len(query_words_filtered)
                                if match_ratio >= 0.7:
                                    exact_match_chunk = chunk
                                    logger.info(f"✅ Found partial match by question: '{question}' (match ratio: {match_ratio:.2f}, common words: {common_words})")
                                    break
                
                # Способ 3: если все еще не нашли, пробуем с transformed_query
                if not exact_match_chunk:
                    logger.info(f"🔍 Trying with transformed query...")
                    all_chunks = await vector_store.asimilarity_search(transformed_query, k=100)
                    logger.info(f"🔍 Got {len(all_chunks)} chunks from similarity search with transformed query")
                    for chunk in all_chunks:
                        question = chunk.metadata.get('question', '').lower().strip()
                        if question:
                            question_normalized = question.rstrip('?').rstrip('!').rstrip('.')
                            query_normalized = query_lower.rstrip('?').rstrip('!').rstrip('.')
                            if question_normalized == query_normalized or question == query_lower:
                                exact_match_chunk = chunk
                                logger.info(f"✅ Found exact match via transformed query: '{question}'")
                                break
            except Exception as e:
                logger.warning(f"Could not search for exact match: {e}", exc_info=True)
        
        # Получаем релевантные чанки через векторный поиск
        retrieved_chunks = await retriever.ainvoke(transformed_query)
        
        # Если нашли точное совпадение, ставим его первым
        if exact_match_chunk:
            # Удаляем точное совпадение из списка, если оно там есть
            retrieved_chunks = [c for c in retrieved_chunks if c != exact_match_chunk]
            # Ставим точное совпадение первым
            retrieved_chunks = [exact_match_chunk] + retrieved_chunks[:config.RETRIEVER_K - 1]
            logger.info(f"✅ Using exact match as first chunk, total chunks: {len(retrieved_chunks)}")
        else:
            # Если точное совпадение не найдено, фильтруем результаты - оставляем только чанки с метаданными question
            # Это поможет убрать нерелевантные чанки из PDF
            filtered_chunks = []
            for chunk in retrieved_chunks:
                if chunk.metadata.get('question'):
                    filtered_chunks.append(chunk)
            
            if filtered_chunks:
                logger.info(f"✅ Filtered to {len(filtered_chunks)} chunks with question metadata (from {len(retrieved_chunks)} total)")
                retrieved_chunks = filtered_chunks[:config.RETRIEVER_K]
            else:
                logger.warning(f"⚠️ No chunks with question metadata found, using all {len(retrieved_chunks)} chunks")
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        if retrieved_chunks:
            # Логируем все найденные чанки для отладки
            for i, chunk in enumerate(retrieved_chunks):
                question = chunk.metadata.get('question', 'N/A')
                category = chunk.metadata.get('category', 'N/A')
                preview = chunk.page_content[:150].replace('\n', ' ')
                logger.info(f"Chunk {i+1}: Q='{question}', Category='{category}', Preview='{preview}...'")
            
            # Проверяем, есть ли в чанках информация о картах
            has_card_info = any("карт" in chunk.page_content.lower() or "card" in chunk.page_content.lower() 
                              for chunk in retrieved_chunks)
            logger.info(f"Chunks contain card-related info: {has_card_info}")
            
            # Проверяем, есть ли точное совпадение вопроса
            query_lower = last_message.lower()
            exact_match = any(query_lower in chunk.page_content.lower() or 
                            chunk.metadata.get('question', '').lower() in query_lower
                            for chunk in retrieved_chunks)
            logger.info(f"Exact question match found: {exact_match}")
        else:
            logger.warning("⚠️ No chunks retrieved! This might be the problem.")
        
        # Форматируем контекст из модифицированных чанков
        formatted_context = format_chunks(retrieved_chunks)
        logger.info(f"📝 Formatted context length: {len(formatted_context)} chars")
        logger.info(f"📝 Formatted context preview (first 800 chars): {formatted_context[:800]}...")
        
        # Проверяем, содержит ли контекст ключевые слова из запроса
        context_lower = formatted_context.lower()
        # Убираем знаки препинания из ключевых слов
        import string
        stop_words = {'как', 'что', 'где', 'когда', 'кто', 'почему', 'зачем', 'для', 'нужны', 'нужен', 'нужна', 'нужно', 'чтобы', 'можно', 'ли'}
        query_keywords = [w.rstrip('?').rstrip('!').rstrip('.').rstrip(',') 
                         for w in query_lower.split() 
                         if len(w.rstrip('?').rstrip('!').rstrip('.').rstrip(',')) > 2 
                         and w.rstrip('?').rstrip('!').rstrip('.').rstrip(',') not in stop_words]
        found_keywords = [kw for kw in query_keywords if kw in context_lower]
        logger.info(f"🔍 Query keywords: {query_keywords}, Found in context: {found_keywords} ({len(found_keywords)}/{len(query_keywords)})")
        
        # Если ключевые слова не найдены, это плохой знак - логируем предупреждение
        if len(found_keywords) == 0 and len(query_keywords) > 0:
            logger.warning(f"⚠️ None of the query keywords found in context! This suggests wrong chunks were retrieved.")
            logger.warning(f"⚠️ Query was: '{last_message}', but context contains: '{formatted_context[:500]}...'")
        
        # Создаем кастомную цепочку, которая использует уже найденные чанки
        # вместо повторного вызова retriever
        conversational_answering_prompt, _ = _load_prompts()
        
        # Создаем функцию, которая возвращает отформатированный контекст
        def get_context(input_dict):
            return formatted_context
        
        # Создаем цепочку с нашим контекстом
        custom_rag_chain = (
            RunnablePassthrough.assign(context=get_context)
            | conversational_answering_prompt
            | _get_llm()
            | StrOutputParser()
        )
        
        # Вызываем кастомную цепочку с уже найденными чанками
        input_data = {"messages": messages}
        logger.debug(f"Input to RAG chain: messages count={len(messages)}")
        
        result = await custom_rag_chain.ainvoke(input_data)
        
        logger.info(f"RAG response generated, length: {len(result)} chars")
        logger.debug(f"RAG response: {result[:200]}...")
        if not result or len(result.strip()) == 0:
            logger.warning("⚠️ Empty response from RAG chain!")
        if "не нашел" in result.lower() or "не нашёл" in result.lower():
            logger.warning("⚠️ LLM returned 'not found' response - context might not be passed correctly!")
        return result
        
    except Exception as e:
        logger.error(f"Error in rag_answer: {e}", exc_info=True)
        raise

def get_vector_store_stats():
    """Возвращает статистику векторного хранилища"""
    if vector_store is None:
        return {"status": "not initialized", "count": 0}
    
    doc_count = len(vector_store.store) if hasattr(vector_store, 'store') else 0
    return {"status": "initialized", "count": doc_count}

