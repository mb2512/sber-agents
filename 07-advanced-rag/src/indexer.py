import json
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
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
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Создаем Document для каждого элемента массива с полем full_text
        documents = []
        for item in data:
            if 'full_text' in item:
                doc = Document(
                    page_content=item['full_text'],
                    metadata={
                        'source': str(json_path),
                        'url': item.get('url', ''),
                        'question': item.get('question', ''),
                        'category': item.get('category', ''),
                        'type': item.get('type', '')
                    }
                )
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} Q&A pairs from JSON")
        return documents
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return []

def create_embeddings():
    """
    Фабрика для создания embeddings по провайдеру из конфига
    Поддерживает: openai (включая Fireworks), huggingface
    При ошибке загрузки HuggingFace автоматически переключается на OpenAI/Fireworks
    """
    provider = config.EMBEDDING_PROVIDER.lower()
    
    if provider == "openai":
        logger.info(f"Creating OpenAI embeddings: {config.EMBEDDING_MODEL}")
        # Поддержка Fireworks и других OpenAI-совместимых провайдеров через base_url
        kwargs = {"model": config.EMBEDDING_MODEL}
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        # api_key обязателен - используем из конфига или переменной окружения
        if config.OPENAI_API_KEY:
            kwargs["api_key"] = config.OPENAI_API_KEY
        else:
            # Если api_key не установлен в конфиге, проверяем переменную окружения
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key
            else:
                raise ValueError(
                    "OPENAI_API_KEY must be set either in config or environment variable. "
                    "Set OPENAI_API_KEY in your .env file or environment variables."
                )
        return OpenAIEmbeddings(**kwargs)
    
    elif provider == "huggingface":
        logger.info(f"Attempting to create HuggingFace embeddings: {config.HUGGINGFACE_EMBEDDING_MODEL} on {config.HUGGINGFACE_DEVICE}")
        try:
            # Пытаемся создать HuggingFace embeddings только из локального кэша
            # local_files_only=True предотвращает попытки скачивания
            embeddings = HuggingFaceEmbeddings(
                model_name=config.HUGGINGFACE_EMBEDDING_MODEL,
                model_kwargs={'device': config.HUGGINGFACE_DEVICE, 'local_files_only': True},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✓ HuggingFace embeddings created successfully from local cache")
            return embeddings
        except Exception as e:
            # Если модель не найдена локально или произошла другая ошибка
            logger.warning(f"Failed to load HuggingFace model from local cache: {e}")
            logger.warning("Falling back to OpenAI/Fireworks embeddings...")
            
            # Автоматически переключаемся на OpenAI/Fireworks
            kwargs = {"model": config.EMBEDDING_MODEL}
            if config.OPENAI_BASE_URL:
                kwargs["base_url"] = config.OPENAI_BASE_URL
            if config.OPENAI_API_KEY:
                kwargs["api_key"] = config.OPENAI_API_KEY
            
            logger.info(f"Creating OpenAI embeddings (fallback): {config.EMBEDDING_MODEL}")
            return OpenAIEmbeddings(**kwargs)
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Use 'openai' or 'huggingface'")

def create_vector_store(chunks: list):
    """Создание векторного хранилища"""
    embeddings = create_embeddings()
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    logger.info(f"Created vector store with {len(chunks)} chunks")
    return vector_store

async def reindex_all():
    """Полная переиндексация всех документов (PDF + JSON)
    
    Returns:
        tuple: (vector_store, chunks) для инициализации retriever
    """
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
            return None, []
        
        logger.info(f"Total chunks to index: {len(all_chunks)} (PDF: {len(pdf_chunks)}, JSON: {len(json_documents)})")
        
        vector_store = create_vector_store(all_chunks)
        logger.info("Reindexing completed successfully")
        
        # Возвращаем vector_store и chunks для BM25
        return vector_store, all_chunks
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None, []
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return None, []

