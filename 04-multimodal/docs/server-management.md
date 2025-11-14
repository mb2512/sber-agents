# Управление ботом на сервере

## Информация о сервере

- **IP адрес**: 195.209.210.204
- **Пользователь**: ubuntu
- **Директория проекта**: `~/telegram-finance-bot`
- **SSH подключение**: `ssh finance-bot-server` (настроено в ~/.ssh/config)

## Управление сервисом

### Проверка статуса
```bash
ssh finance-bot-server "sudo systemctl status telegram-finance-bot"
```

### Просмотр логов
```bash
# Логи в реальном времени
ssh finance-bot-server "sudo journalctl -u telegram-finance-bot -f"

# Последние 50 строк
ssh finance-bot-server "sudo journalctl -u telegram-finance-bot -n 50"

# Логи за сегодня
ssh finance-bot-server "sudo journalctl -u telegram-finance-bot --since today"
```

### Управление сервисом
```bash
# Остановка
ssh finance-bot-server "sudo systemctl stop telegram-finance-bot"

# Запуск
ssh finance-bot-server "sudo systemctl start telegram-finance-bot"

# Перезапуск
ssh finance-bot-server "sudo systemctl restart telegram-finance-bot"

# Отключить автозапуск
ssh finance-bot-server "sudo systemctl disable telegram-finance-bot"

# Включить автозапуск
ssh finance-bot-server "sudo systemctl enable telegram-finance-bot"
```

## Переключение между провайдерами

### Переключение на Ollama
```bash
ssh finance-bot-server "cd ~/telegram-finance-bot && nano .env"
```

Закомментируйте OpenRouter и раскомментируйте Ollama:
```bash
# OpenRouter (закомментировано)
# OPENAI_API_KEY=sk-or-v1-...
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# MODEL_TEXT=openai/gpt-oss-20b:free
# MODEL_IMAGE=qwen/qwen2.5-vl-32b-instruct

# Ollama (активно)
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://195.209.214.27:11434/v1
MODEL_TEXT=gpt-oss:20b
MODEL_IMAGE=qwen3-vl:8b-instruct
```

Затем перезапустите:
```bash
ssh finance-bot-server "sudo systemctl restart telegram-finance-bot"
```

### Переключение на OpenRouter
Аналогично, но наоборот - закомментируйте Ollama и раскомментируйте OpenRouter.

## Обновление бота

```bash
# Остановите бота
ssh finance-bot-server "sudo systemctl stop telegram-finance-bot"

# Обновите код (если используете git)
ssh finance-bot-server "cd ~/telegram-finance-bot && git pull"

# Или скопируйте новые файлы с локальной машины
cd "/c/Users/mihae/OneDrive/Документы/GitHub/sber-agents/04-multimodal"
tar czf - --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.env' . | \
  ssh finance-bot-server "cd ~/telegram-finance-bot && tar xzf -"

# Обновите зависимости (если изменились)
ssh finance-bot-server "cd ~/telegram-finance-bot && export PATH=\"\$HOME/.local/bin:\$PATH\" && uv sync"

# Запустите бота
ssh finance-bot-server "sudo systemctl start telegram-finance-bot"

# Проверьте логи
ssh finance-bot-server "sudo journalctl -u telegram-finance-bot -f"
```

## Проверка работоспособности

### 1. Проверка статуса сервиса
```bash
ssh finance-bot-server "sudo systemctl status telegram-finance-bot"
```

Должно быть: `Active: active (running)`

### 2. Проверка в Telegram
- Откройте бота в Telegram
- Отправьте `/start`
- Должен прийти ответ

### 3. Тест функций
- Отправьте: "Купил продукты на 500 рублей"
- Отправьте: `/balance`
- Отправьте: `/transactions`
- Отправьте фото чека

## Устранение проблем

### Бот не отвечает
```bash
# Проверьте логи на ошибки
ssh finance-bot-server "sudo journalctl -u telegram-finance-bot -n 100 --no-pager | grep -i error"

# Проверьте статус
ssh finance-bot-server "sudo systemctl status telegram-finance-bot"
```

### Проблемы с Ollama
```bash
# Проверьте доступность Ollama
ssh finance-bot-server "curl http://195.209.214.27:11434/api/tags"

# Проверьте, что модели установлены
ssh finance-bot-server "curl http://195.209.214.27:11434/api/tags | grep -E 'gpt-oss|qwen3-vl'"
```

### Проблемы с конфигурацией
```bash
# Проверьте .env файл
ssh finance-bot-server "cd ~/telegram-finance-bot && cat .env"

# Проверьте загрузку конфигурации
ssh finance-bot-server "cd ~/telegram-finance-bot && export PATH=\"\$HOME/.local/bin:\$PATH\" && uv run python -c \"from src.config import config; print('Token:', config.TELEGRAM_TOKEN[:20] + '...' if config.TELEGRAM_TOKEN else 'НЕТ'); print('API:', config.OPENAI_API_KEY[:20] + '...' if config.OPENAI_API_KEY else 'НЕТ')\""
```

### Перезапуск с нуля
```bash
# Остановите бота
ssh finance-bot-server "sudo systemctl stop telegram-finance-bot"

# Удалите виртуальное окружение (если нужно)
ssh finance-bot-server "cd ~/telegram-finance-bot && rm -rf .venv"

# Переустановите зависимости
ssh finance-bot-server "cd ~/telegram-finance-bot && export PATH=\"\$HOME/.local/bin:\$PATH\" && uv sync"

# Запустите снова
ssh finance-bot-server "sudo systemctl start telegram-finance-bot"
```

## Полезные команды

```bash
# Быстрая проверка всех компонентов
ssh finance-bot-server "echo '=== Status ===' && sudo systemctl status telegram-finance-bot --no-pager | head -5 && echo '' && echo '=== Last logs ===' && sudo journalctl -u telegram-finance-bot -n 5 --no-pager"

# Проверка использования ресурсов
ssh finance-bot-server "ps aux | grep 'python src/bot.py'"

# Проверка сетевых подключений
ssh finance-bot-server "netstat -tuln | grep -E '11434|443'"
```

## Резервное копирование

```bash
# Создайте резервную копию конфигурации
ssh finance-bot-server "cd ~/telegram-finance-bot && tar czf ~/bot-backup-$(date +%Y%m%d).tar.gz .env prompts/ src/ pyproject.toml"

# Скачайте резервную копию
scp finance-bot-server:~/bot-backup-*.tar.gz ./
```

