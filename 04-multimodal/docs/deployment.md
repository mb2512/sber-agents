# Инструкция по развертыванию на новом сервере

## Информация о сервере

- **IP адрес**: 195.209.210.204
- **MAC адрес**: fa:16:3e:ab:21:fb
- **Наименование**: mb2512new1

## Предварительные требования

1. Подключение к серверу по SSH
2. Python 3.11+ установлен
3. Git установлен
4. uv установлен (менеджер зависимостей)

## Пошаговая инструкция

### 1. Подключение к серверу

```bash
ssh user@195.209.210.204
```

(Замените `user` на ваше имя пользователя)

### 2. Установка системных зависимостей

#### Для Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git curl build-essential
```

#### Для CentOS/RHEL:
```bash
sudo yum update -y
sudo yum install -y python3.11 python3-pip git curl gcc gcc-c++ make
```

### 3. Установка uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Или через pip:
```bash
pip install uv
```

Добавьте uv в PATH (если установлен через скрипт):
```bash
export PATH="$HOME/.cargo/bin:$PATH"
# Или добавьте в ~/.bashrc для постоянного использования
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Клонирование репозитория

```bash
# Перейдите в домашнюю директорию или выберите место для проекта
cd ~

# Клонируйте репозиторий
git clone <repository-url> telegram-finance-bot
cd telegram-finance-bot
```

(Замените `<repository-url>` на URL вашего репозитория)

### 5. Установка зависимостей проекта

```bash
# Установка зависимостей через uv
make install

# Или напрямую:
uv sync
```

### 6. Настройка переменных окружения

```bash
# Создайте файл .env из примера
cp .env.example .env

# Отредактируйте .env файл
nano .env
```

**Минимальная конфигурация .env:**

```bash
# Telegram Bot Token (обязательно)
TELEGRAM_TOKEN=ваш_токен_от_BotFather

# OpenAI/OpenRouter API (обязательно)
OPENAI_API_KEY=ваш_ключ_от_OpenRouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Модели для обработки
MODEL_TEXT=openai/gpt-oss-20b:free
MODEL_IMAGE=meta-llama/llama-3.2-11b-vision-instruct

# Пути к промптам (опционально, по умолчанию используются файлы)
SYSTEM_PROMPT_TEXT_PATH=prompts/system_prompt_text.txt
SYSTEM_PROMPT_IMAGE_PATH=prompts/system_prompt_image.txt
```

**Для использования Ollama (локальная модель):**

```bash
TELEGRAM_TOKEN=ваш_токен_от_BotFather
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_TEXT=llama3.2
MODEL_IMAGE=llama3.2-vision
```

### 7. Проверка установки

```bash
# Проверьте, что все зависимости установлены
uv run python -c "import aiogram, openai, pydantic; print('Все зависимости установлены')"

# Проверьте конфигурацию
uv run python -c "from src.config import config; print(f'Token: {config.TELEGRAM_TOKEN[:10]}...')"
```

### 8. Запуск бота (тестовый режим)

```bash
# Запуск через Makefile
make run

# Или напрямую:
uv run python src/bot.py
```

Если бот запустился успешно, вы увидите в логах:
```
INFO - Starting bot...
```

Нажмите `Ctrl+C` для остановки.

### 9. Настройка автозапуска через systemd

Создайте файл сервиса:

```bash
sudo nano /etc/systemd/system/telegram-finance-bot.service
```

Содержимое файла:

```ini
[Unit]
Description=Telegram Finance Bot
After=network.target

[Service]
Type=simple
User=ваш_пользователь
WorkingDirectory=/home/ваш_пользователь/telegram-finance-bot
Environment="PATH=/home/ваш_пользователь/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ваш_пользователь/.cargo/bin/uv run python src/bot.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Важно:** Замените:
- `ваш_пользователь` на ваше имя пользователя
- Путь к `uv` на актуальный (проверьте через `which uv`)

Активация и запуск сервиса:

```bash
# Перезагрузите systemd
sudo systemctl daemon-reload

# Включите автозапуск
sudo systemctl enable telegram-finance-bot

# Запустите сервис
sudo systemctl start telegram-finance-bot

# Проверьте статус
sudo systemctl status telegram-finance-bot
```

### 10. Просмотр логов

```bash
# Логи systemd
sudo journalctl -u telegram-finance-bot -f

# Последние 100 строк
sudo journalctl -u telegram-finance-bot -n 100

# Логи за сегодня
sudo journalctl -u telegram-finance-bot --since today
```

### 11. Управление сервисом

```bash
# Остановка
sudo systemctl stop telegram-finance-bot

# Запуск
sudo systemctl start telegram-finance-bot

# Перезапуск
sudo systemctl restart telegram-finance-bot

# Отключение автозапуска
sudo systemctl disable telegram-finance-bot

# Проверка статуса
sudo systemctl status telegram-finance-bot
```

## Обновление бота

При обновлении кода:

```bash
# Перейдите в директорию проекта
cd ~/telegram-finance-bot

# Остановите сервис
sudo systemctl stop telegram-finance-bot

# Получите последние изменения
git pull

# Обновите зависимости (если изменились)
uv sync

# Запустите сервис
sudo systemctl start telegram-finance-bot

# Проверьте логи
sudo journalctl -u telegram-finance-bot -f
```

## Проверка работоспособности

1. **Проверка подключения к Telegram:**
   - Откройте бота в Telegram
   - Отправьте команду `/start`
   - Должен прийти ответ от бота

2. **Проверка обработки текста:**
   - Отправьте сообщение: "Сегодня купил продукты на 1500 рублей"
   - Бот должен извлечь транзакцию и показать баланс

3. **Проверка обработки изображений:**
   - Отправьте фото чека
   - Бот должен обработать изображение и извлечь транзакции

4. **Проверка команд:**
   - `/balance` - показать баланс
   - `/transactions` - показать все транзакции

## Устранение проблем

### Бот не запускается

1. Проверьте логи:
   ```bash
   sudo journalctl -u telegram-finance-bot -n 50
   ```

2. Проверьте конфигурацию:
   ```bash
   cd ~/telegram-finance-bot
   uv run python -c "from src.config import config; print(config.TELEGRAM_TOKEN)"
   ```

3. Проверьте права доступа к файлам:
   ```bash
   ls -la ~/telegram-finance-bot
   ```

### Ошибки подключения к API

1. Проверьте интернет-соединение:
   ```bash
   ping openrouter.ai
   ```

2. Проверьте API ключ в `.env` файле

3. Проверьте доступность API:
   ```bash
   curl https://openrouter.ai/api/v1/models
   ```

### Проблемы с зависимостями

1. Переустановите зависимости:
   ```bash
   cd ~/telegram-finance-bot
   uv sync --reinstall
   ```

2. Проверьте версию Python:
   ```bash
   python3 --version  # Должно быть 3.11+
   ```

## Резервное копирование

Рекомендуется регулярно делать резервные копии:

1. **Код проекта:**
   - Код хранится в Git репозитории
   - Регулярно делайте коммиты и пуш

2. **Конфигурация:**
   ```bash
   # Сохраните .env файл в безопасное место
   cp ~/telegram-finance-bot/.env ~/backup/.env.$(date +%Y%m%d)
   ```

## Безопасность

1. **Защита .env файла:**
   ```bash
   chmod 600 ~/telegram-finance-bot/.env
   ```

2. **Firewall:**
   - Убедитесь, что открыты только необходимые порты
   - Бот не требует входящих соединений

3. **Обновления:**
   - Регулярно обновляйте систему и зависимости

## Контакты и поддержка

При возникновении проблем:
1. Проверьте логи: `sudo journalctl -u telegram-finance-bot -f`
2. Проверьте документацию в `README.md` и `docs/`
3. Проверьте статус сервиса: `sudo systemctl status telegram-finance-bot`

