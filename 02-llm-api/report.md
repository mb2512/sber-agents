## 1. Настройка и запуск (Задание 1)

### ✅ Скриншот успешного запуска
### ✅ Пример простого диалога с метриками
![Скриншот запуска](screenshots/bot(role_default).png)

## 2. Системные промпты (Задание 2)

![Скриншот системного промта1](screenshots/bot(promt2).png)
![Скриншот запуска промта1](screenshots/bot(role_2).png)
![Скриншот системного промта2](screenshots/bot(promt3).png)
![Скриншот запуска промта2](screenshots/bot(role_3).png)
![Скриншот системного промта3](screenshots/bot(promt_mine).png)
![Скриншот запуска промта3](screenshots/bot(role_mine).png)

**Наблюдения:**
Если провести анализ системного промта, то выигрывает модель 2 эмпатичный ассистент. Минимум "воды".

## 3. Сравнение моделей (Задание 3)
Были применены модели:
openrouter/andromeda-alpha
meituan/longcat-flash-chat:free
deepseek/deepseek-r1-0528-qwen3-8b:free

На промте: Привет, как дела?
Результаты:
![Скриншот модель1](screenshots/bot(compare1).png)
![Скриншот модель2](screenshots/bot(compare2).png)
![Скриншот модель3](screenshots/bot(compare3).png)

**Наблюдения:**
Модель openrouter/andromeda-alpha оказалась менее творческой. Остальные модели соответствовали системному промту.

## 4. Управление историей (Задание 4)

### Описание реализованной стратегии

Реализована стратегия **ограничения по количеству сообщений**. Система сохраняет:
- Системный промпт (первое сообщение)
- Последние N-1 сообщений диалога

Где N = 10 (MAX_MESSAGES)

### Фрагмент кода

```python
def add_message(self, role: str, content: str):
    """Добавить сообщение в историю диалога."""
    self.conversation_history.append({
        "role": role,
        "content": content
    })
    MAX_MESSAGES = 10
    
    if len(self.conversation_history) > MAX_MESSAGES:
        # Сохраняем системный промпт + последние сообщения
        system_prompt = self.conversation_history[0]
        self.conversation_history = [system_prompt] + self.conversation_history[-(MAX_MESSAGES-1):]
```


![Скриншот статистики](screenshots/stats(limit10).png)
