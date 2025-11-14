#!/bin/bash

# Скрипт автоматического развертывания Telegram Finance Bot
# Использование: ./scripts/deploy.sh

set -e  # Остановка при ошибке

echo "🚀 Начало развертывания Telegram Finance Bot..."

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка, что скрипт запущен из корня проекта
if [ ! -f "pyproject.toml" ]; then
    error "Скрипт должен быть запущен из корня проекта (где находится pyproject.toml)"
    exit 1
fi

# 1. Проверка Python
info "Проверка Python..."
if ! command -v python3 &> /dev/null; then
    error "Python 3 не найден. Установите Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
info "Найден Python $PYTHON_VERSION"

# 2. Проверка/установка uv
info "Проверка uv..."
if ! command -v uv &> /dev/null; then
    warn "uv не найден. Устанавливаю..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        error "Не удалось установить uv"
        exit 1
    fi
    info "uv успешно установлен"
else
    info "uv уже установлен"
fi

# 3. Установка зависимостей
info "Установка зависимостей проекта..."
if [ -f "Makefile" ]; then
    make install
else
    uv sync
fi
info "Зависимости установлены"

# 4. Проверка .env файла
info "Проверка конфигурации..."
if [ ! -f ".env" ]; then
    warn ".env файл не найден"
    if [ -f ".env.example" ]; then
        info "Создаю .env из .env.example..."
        cp .env.example .env
        warn "⚠️  ВАЖНО: Отредактируйте файл .env и укажите ваши токены!"
        warn "   nano .env"
    else
        error ".env.example не найден. Создайте .env файл вручную"
        exit 1
    fi
else
    info ".env файл найден"
fi

# 5. Проверка обязательных переменных
info "Проверка обязательных переменных окружения..."
source .env 2>/dev/null || true

if [ -z "$TELEGRAM_TOKEN" ]; then
    error "TELEGRAM_TOKEN не установлен в .env файле"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    error "OPENAI_API_KEY не установлен в .env файле"
    exit 1
fi

info "Обязательные переменные проверены"

# 6. Проверка промптов
info "Проверка системных промптов..."
if [ ! -f "prompts/system_prompt_text.txt" ]; then
    error "prompts/system_prompt_text.txt не найден"
    exit 1
fi

if [ ! -f "prompts/system_prompt_image.txt" ]; then
    error "prompts/system_prompt_image.txt не найден"
    exit 1
fi

info "Промпты найдены"

# 7. Тестовая проверка импортов
info "Проверка импортов..."
if ! uv run python -c "from src.config import config; print('Config OK')" 2>/dev/null; then
    error "Ошибка при импорте конфигурации"
    exit 1
fi

info "Импорты работают корректно"

# 8. Создание systemd сервиса (опционально)
read -p "Создать systemd сервис для автозапуска? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Создание systemd сервиса..."
    
    SERVICE_NAME="telegram-finance-bot"
    SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
    CURRENT_USER=$(whoami)
    PROJECT_DIR=$(pwd)
    UV_PATH=$(which uv)
    
    if [ -z "$UV_PATH" ]; then
        UV_PATH="$HOME/.cargo/bin/uv"
    fi
    
    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Telegram Finance Bot
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$UV_PATH run python src/bot.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME"
    
    info "Сервис создан: $SERVICE_NAME"
    info "Для запуска: sudo systemctl start $SERVICE_NAME"
    info "Для просмотра логов: sudo journalctl -u $SERVICE_NAME -f"
fi

# 9. Финальная проверка
info "Финальная проверка..."
echo ""
echo "✅ Развертывание завершено!"
echo ""
echo "📋 Следующие шаги:"
echo "   1. Убедитесь, что .env файл настроен корректно"
echo "   2. Запустите бота:"
echo "      - Вручную: make run"
echo "      - Через systemd: sudo systemctl start telegram-finance-bot"
echo ""
echo "📖 Документация: docs/deployment.md"
echo ""

