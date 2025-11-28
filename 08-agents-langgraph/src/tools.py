"""
Инструменты для ReAct агента

Инструменты - это функции, которые агент может вызывать для получения информации.
Декоратор @tool из LangChain автоматически создает описание для LLM.
"""
import json
import logging
from langchain_core.tools import tool
import rag

logger = logging.getLogger(__name__)

# Фиксированные курсы валют (для заглушки)
# В реальном проекте можно заменить на вызов API exchangerate-api.com или другого сервиса
CURRENCY_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "RUB": 92.0,
    "GBP": 0.79,
    "CNY": 7.2,
}

@tool
def rag_search(query: str) -> str:
    """
    Ищет информацию в документах Сбербанка (условия кредитов, вкладов и других банковских продуктов).
    
    Возвращает JSON со списком источников, где каждый источник содержит:
    - source: имя файла
    - page: номер страницы (только для PDF)
    - page_content: текст документа
    """
    try:
        # Получаем релевантные документы через RAG (retrieval + reranking)
        documents = rag.retrieve_documents(query)
        
        if not documents:
            return json.dumps({"sources": []}, ensure_ascii=False)
        
        # Формируем структурированный ответ для агента
        sources = []
        for doc in documents:
            source_data = {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content  # Полный текст документа
            }
            # page только для PDF (у JSON документов его нет)
            if "page" in doc.metadata:
                source_data["page"] = doc.metadata["page"]
            sources.append(source_data)
        
        # ensure_ascii=False для корректной кириллицы
        return json.dumps({"sources": sources}, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return json.dumps({"sources": []}, ensure_ascii=False)


@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Конвертирует сумму из одной валюты в другую.
    
    Поддерживаемые валюты: USD, EUR, RUB, GBP, CNY
    
    Args:
        amount: Сумма для конвертации (положительное число)
        from_currency: Исходная валюта (код из 3 букв: USD, EUR, RUB, GBP, CNY)
        to_currency: Целевая валюта (код из 3 букв: USD, EUR, RUB, GBP, CNY)
    
    Returns:
        Строка с результатом конвертации в читаемом формате
    
    Example:
        currency_converter(100, "USD", "RUB") -> "100.00 USD = 9200.00 RUB"
    """
    try:
        # Нормализуем коды валют (верхний регистр)
        from_currency = from_currency.upper().strip()
        to_currency = to_currency.upper().strip()
        
        # Проверяем валидность суммы
        if amount < 0:
            return f"Ошибка: сумма должна быть положительной, получено {amount}"
        
        # Проверяем поддержку валют
        if from_currency not in CURRENCY_RATES:
            return f"Ошибка: валюта '{from_currency}' не поддерживается. Доступные валюты: {', '.join(CURRENCY_RATES.keys())}"
        
        if to_currency not in CURRENCY_RATES:
            return f"Ошибка: валюта '{to_currency}' не поддерживается. Доступные валюты: {', '.join(CURRENCY_RATES.keys())}"
        
        # Если одинаковые валюты
        if from_currency == to_currency:
            return f"{amount:.2f} {from_currency} = {amount:.2f} {to_currency} (без конвертации)"
        
        # Конвертация через USD как базовую валюту
        # Сначала конвертируем из исходной валюты в USD
        amount_in_usd = amount / CURRENCY_RATES[from_currency]
        # Затем конвертируем из USD в целевую валюту
        result = amount_in_usd * CURRENCY_RATES[to_currency]
        
        logger.info(f"Currency conversion: {amount} {from_currency} -> {result:.2f} {to_currency}")
        
        return f"{amount:.2f} {from_currency} = {result:.2f} {to_currency}"
        
    except Exception as e:
        logger.error(f"Error in currency_converter: {e}", exc_info=True)
        return f"Ошибка при конвертации валют: {str(e)}"

