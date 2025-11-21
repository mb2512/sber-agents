import logging
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from config import config

logger = logging.getLogger(__name__)

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

def split_documents(pages: list) -> list:
    """Разбиение документов на чанки"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(pages)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def load_json_documents(json_file_path: str) -> list:
    """Загрузка Q&A пар из JSON, каждая пара - отдельный чанк"""
    json_path = Path(json_file_path)
    if not json_path.exists():
        logger.warning(f"JSON file {json_file_path} does not exist")
        return []
    
    try:
        loader = JSONLoader(
            file_path=str(json_path),
            jq_schema='.[].full_text',
            text_content=False
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} Q&A pairs from JSON")
        return documents
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return []

def create_vector_store(chunks: list):
    """Создание векторного хранилища"""
    
    # Валидация: проверяем соответствие провайдера и модели
    model_name = config.EMBEDDING_MODEL.lower()
    is_openai_model = model_name.startswith("openai/") or model_name.startswith("text-embedding")
    
    if config.EMBEDDING_PROVIDER == "huggingface" and is_openai_model:
        error_msg = (
            f"Ошибка конфигурации: EMBEDDING_PROVIDER=huggingface, но указана модель OpenAI: {config.EMBEDDING_MODEL}\n"
            f"Модель '{config.EMBEDDING_MODEL}' доступна только через OpenAI API, а не через HuggingFace.\n"
            f"Решения:\n"
            f"1. Используйте HuggingFace модель, например: EMBEDDING_MODEL=intfloat/multilingual-e5-base\n"
            f"2. Или измените провайдер: EMBEDDING_PROVIDER=openai"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Выбор embeddings на основе провайдера
    if config.EMBEDDING_PROVIDER == "huggingface":
        logger.info(f"Using HuggingFace embeddings: {config.EMBEDDING_MODEL}")
        logger.info("Loading embedding model (this may take a while on first run)...")
        
        # Попытки загрузки модели с повторными попытками
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(1, max_retries + 1):
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},  # или 'cuda' если есть GPU
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Embedding model loaded successfully")
                break
            except (ConnectionError, TimeoutError, Exception) as e:
                if attempt < max_retries:
                    logger.warning(f"Failed to load model (attempt {attempt}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Увеличиваем задержку с каждой попыткой
                else:
                    logger.error(f"Failed to load embedding model after {max_retries} attempts: {e}")
                    logger.error("This might be due to network issues or slow connection.")
                    logger.error("Possible solutions:")
                    logger.error("1. Check your internet connection")
                    logger.error("2. Try again later (model will be cached after first download)")
                    logger.error("3. Verify that the model name is correct for HuggingFace")
                    logger.error(f"4. Set EMBEDDING_PROVIDER=openai to use API embeddings instead")
                    raise
    else:
        logger.info(f"Using OpenAI embeddings: {config.EMBEDDING_MODEL}")
        embedding_kwargs = {
            "model": config.EMBEDDING_MODEL
        }
        
        # Если указан base_url (например, для OpenRouter), используем его
        if config.OPENAI_BASE_URL:
            embedding_kwargs["base_url"] = config.OPENAI_BASE_URL
            logger.info(f"Using base URL: {config.OPENAI_BASE_URL}")
        
        # Если указан API ключ, используем его
        if config.OPENAI_API_KEY:
            embedding_kwargs["api_key"] = config.OPENAI_API_KEY
        
        embeddings = OpenAIEmbeddings(**embedding_kwargs)
    
    logger.info(f"Creating vector store with {len(chunks)} chunks...")
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    logger.info(f"Created vector store with {len(chunks)} chunks")
    return vector_store

async def reindex_all():
    """Полная переиндексация всех документов (PDF + JSON)"""
    logger.info("Starting full reindexing...")
    
    try:
        # Загрузка PDF документов
        pages = load_pdf_documents(config.DATA_DIR)
        pdf_chunks = split_documents(pages) if pages else []
        logger.info(f"PDF: {len(pdf_chunks)} chunks")
        
        # Загрузка JSON Q&A пар
        json_file = Path(config.DATA_DIR) / "sberbank_help_documents.json"
        json_documents = load_json_documents(str(json_file))
        logger.info(f"JSON: {len(json_documents)} Q&A pairs")
        
        # Объединяем все чанки
        all_chunks = pdf_chunks + json_documents
        
        if not all_chunks:
            logger.warning("No documents found to index")
            return None
        
        logger.info(f"Total chunks to index: {len(all_chunks)} (PDF: {len(pdf_chunks)}, JSON: {len(json_documents)})")
        
        vector_store = create_vector_store(all_chunks)
        logger.info("Reindexing completed successfully")
        return vector_store
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return None

