import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from config import config

logger = logging.getLogger(__name__)

# Импорт OllamaEmbeddings (опционально)
try:
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("langchain-ollama not installed. Ollama embeddings will not be available.")

def load_pdf_documents(data_dir: str) -> list:
    """Загрузка всех PDF документов из директории"""
    pages = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Directory {data_dir} does not exist")
        return pages
    
    pdf_files = list(data_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages.extend(loader.load())
        logger.info(f"Loaded {pdf_file.name}")
    
    return pages

def load_json_documents(json_file_path: str) -> list:
    """
    Загрузка документов из JSON файла с вопросами-ответами
    Каждая пара Q&A становится отдельным чанком
    """
    from pathlib import Path
    import json
    
    json_path = Path(json_file_path)
    if not json_path.exists():
        logger.warning(f"JSON file {json_file_path} does not exist")
        return []
    
    try:
        # JSONLoader с jq_schema для извлечения full_text из каждого элемента массива
        loader = JSONLoader(
            file_path=str(json_path),
            jq_schema='.[].full_text',  # Извлекаем full_text из каждого элемента
            text_content=False
        )
        
        documents = loader.load()
        
        # Добавляем метаданные к документам для лучшего поиска
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Сопоставляем документы с данными по индексу
        for i, doc in enumerate(documents):
            if i < len(data):
                item = data[i]
                # Обновляем метаданные
                doc.metadata.update({
                    'question': item.get('question', ''),
                    'category': item.get('category', ''),
                    'url': item.get('url', '')
                })
                # Логируем первые несколько для проверки
                if i < 3:
                    logger.info(f"Sample document {i+1}: question='{item.get('question', '')}', category='{item.get('category', '')}'")
        
        logger.info(f"Loaded {len(documents)} Q&A pairs from JSON")
        return documents
    except ImportError as e:
        if "jq" in str(e).lower():
            logger.error("jq package is required for JSONLoader. Install it with: uv sync")
            logger.warning("Falling back to manual JSON parsing...")
            # Fallback: ручная загрузка JSON
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                from langchain_core.documents import Document
                documents = []
                for i, item in enumerate(data):
                    if 'full_text' in item:
                        # Добавляем метаданные для лучшего поиска
                        metadata = {
                            'source': str(json_path),
                            'type': 'json',
                            'question': item.get('question', ''),
                            'category': item.get('category', ''),
                            'url': item.get('url', '')
                        }
                        doc = Document(
                            page_content=item['full_text'],
                            metadata=metadata
                        )
                        documents.append(doc)
                        # Логируем первые несколько для проверки
                        if i < 3:
                            logger.info(f"Sample document {i+1}: question='{metadata['question']}', category='{metadata['category']}'")
                
                logger.info(f"Loaded {len(documents)} Q&A pairs from JSON (manual parsing)")
                return documents
            except Exception as fallback_error:
                logger.error(f"Error in fallback JSON parsing: {fallback_error}")
                return []
        else:
            raise
    except Exception as e:
        logger.error(f"Error loading JSON documents: {e}", exc_info=True)
        return []

def split_documents(pages: list) -> list:
    """Разбиение документов с учетом структуры"""
    # Сепараторы для банковских документов
    # Пробуем разбивать по: двойным переносам строк, одинарным, пробелам
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n\n\n",    # Тройной перенос - обычно разделы
            "\n\n",      # Двойной перенос - параграфы
            "\n",        # Одинарный перенос
            ". ",        # Конец предложения
            " ",         # Пробелы
            ""           # Символы
        ],
        keep_separator=True  # Сохраняем разделители для контекста
    )
    chunks = text_splitter.split_documents(pages)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks: list):
    """Создание векторного хранилища"""
    try:
        # Определяем тип эмбеддингов по имени модели
        # Если модель начинается с "aroxima/" или содержит "ollama", используем Ollama
        use_ollama = (
            OLLAMA_AVAILABLE and 
            (config.EMBEDDING_MODEL.startswith("aroxima/") or 
             "ollama" in config.EMBEDDING_MODEL.lower() or
             config.EMBEDDING_MODEL.endswith(":latest"))
        )
        
        if use_ollama:
            logger.info(f"Using Ollama embeddings with model: {config.EMBEDDING_MODEL}")
            embeddings = OllamaEmbeddings(
                model=config.EMBEDDING_MODEL
            )
        else:
            logger.info(f"Using OpenAI-compatible embeddings with model: {config.EMBEDDING_MODEL}")
            embeddings = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL,
                openai_api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL,
                timeout=config.REQUEST_TIMEOUT,
                max_retries=2
            )
        
        logger.info(f"Creating vector store with {len(chunks)} chunks using model {config.EMBEDDING_MODEL}")
        vector_store = InMemoryVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        logger.info(f"Created vector store with {len(chunks)} chunks")
        
        # Проверяем, что метаданные сохранились - используем similarity_search, так как это правильный способ
        try:
            # Ищем конкретный вопрос, который должен быть в JSON
            test_chunks = vector_store.similarity_search("Как заказать карту?", k=10)
            logger.info(f"🔍 Testing metadata preservation: found {len(test_chunks)} chunks for test query")
            found_question = False
            for i, chunk in enumerate(test_chunks[:5]):
                if hasattr(chunk, 'metadata'):
                    question = chunk.metadata.get('question', '')
                    if question:
                        logger.info(f"✅ Test chunk {i+1} metadata: question='{question}'")
                        if 'заказать' in question.lower() and 'карту' in question.lower():
                            found_question = True
                    else:
                        logger.warning(f"⚠️ Test chunk {i+1} has metadata but no 'question' field")
                else:
                    logger.warning(f"⚠️ Test chunk {i+1} has no metadata attribute")
            
            if found_question:
                logger.info("✅ Metadata preservation verified - found expected question in test search")
            else:
                logger.warning("⚠️ Could not find expected question 'Как заказать карту?' in test search - metadata might not be preserved correctly")
        except Exception as e:
            logger.warning(f"⚠️ Could not test metadata preservation: {e}")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        raise

async def reindex_all():
    """Полная переиндексация всех документов (PDF + JSON)"""
    logger.info("Starting full reindexing...")
    
    try:
        # 1. Загружаем и обрабатываем PDF документы
        pdf_pages = load_pdf_documents(config.DATA_DIR)
        if not pdf_pages:
            logger.warning("No PDF documents found to index")
        
        pdf_chunks = split_documents(pdf_pages) if pdf_pages else []
        
        # 2. Загружаем JSON с вопросами-ответами
        json_file = f"{config.DATA_DIR}/sberbank_help_documents.json"
        json_chunks = load_json_documents(json_file)
        
        # 3. Объединяем все чанки
        all_chunks = pdf_chunks + json_chunks
        
        if not all_chunks:
            logger.warning("No documents found to index")
            return None
        
        logger.info(f"Total chunks to index: {len(all_chunks)} (PDF: {len(pdf_chunks)}, JSON: {len(json_chunks)})")
            
        # 4. Создаём векторное хранилище
        logger.info("Creating vector store...")
        vector_store = create_vector_store(all_chunks)
        logger.info("Reindexing completed successfully")
        return vector_store
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except ValueError as e:
        logger.error(f"Configuration error: {e}. Check your .env file and API keys.")
        return None
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return None

