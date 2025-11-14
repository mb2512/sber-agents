# Быстрый старт после развертывания

## Проверка работоспособности

### 1. Запуск бота в тестовом режиме

```bash
cd ~/telegram-finance-bot
make run
```

Или если используете systemd:
```bash
sudo systemctl start telegram-finance-bot
sudo systemctl status telegram-finance-bot
```

### 2. Проверка в Telegram

1. Найдите вашего бота в Telegram
2. Отправьте команду `/start`
3. Должен прийти ответ с приветствием

### 3. Тестирование функций

#### Тест 1: Обработка текстовых сообщений
```
Отправьте: "Сегодня купил продукты на 1500 рублей"
Ожидается: 
- Извлечение транзакции
- Сообщение о сохранении
- Показ баланса
```

#### Тест 2: Команда /balance
```
Отправьте: /balance
Ожидается: Отчет с балансом, доходами, расходами и статистикой
```

#### Тест 3: Команда /transactions
```
Отправьте: /transactions
Ожидается: Список всех сохраненных транзакций
```

#### Тест 4: Обработка изображений
```
Отправьте: Фото чека
Ожидается:
- Обработка изображения
- Извлечение транзакций из чека
- Сохранение и показ баланса
```

### 4. Просмотр логов

Если бот запущен через systemd:
```bash
# Логи в реальном времени
sudo journalctl -u telegram-finance-bot -f

# Последние 50 строк
sudo journalctl -u telegram-finance-bot -n 50
```

Если бот запущен вручную - логи выводятся в консоль.

### 5. Проверка ошибок

Если что-то не работает:

1. **Проверьте логи:**
   ```bash
   sudo journalctl -u telegram-finance-bot -n 100 --no-pager
   ```

2. **Проверьте конфигурацию:**
   ```bash
   cd ~/telegram-finance-bot
   cat .env | grep -E "TELEGRAM_TOKEN|OPENAI_API_KEY"
   ```

3. **Проверьте подключение к API:**
   ```bash
   curl -I https://openrouter.ai/api/v1/models
   ```

4. **Проверьте импорты:**
   ```bash
   uv run python -c "from src.config import config; print('OK')"
   ```

## Автозапуск при перезагрузке

Если используете systemd, автозапуск уже настроен:
```bash
# Проверка статуса
sudo systemctl status telegram-finance-bot

# Если не включен автозапуск:
sudo systemctl enable telegram-finance-bot
```

## Остановка/перезапуск

```bash
# Остановка
sudo systemctl stop telegram-finance-bot

# Запуск
sudo systemctl start telegram-finance-bot

# Перезапуск
sudo systemctl restart telegram-finance-bot
```

## Полезные команды

```bash
# Проверка статуса сервиса
sudo systemctl status telegram-finance-bot

# Просмотр логов за последний час
sudo journalctl -u telegram-finance-bot --since "1 hour ago"

# Просмотр логов за сегодня
sudo journalctl -u telegram-finance-bot --since today

# Перезагрузка конфигурации systemd
sudo systemctl daemon-reload
```

## Если что-то не работает

1. Проверьте, что все переменные в `.env` заполнены
2. Проверьте интернет-соединение
3. Проверьте логи на наличие ошибок
4. Убедитесь, что Python 3.11+ установлен
5. Убедитесь, что все зависимости установлены: `uv sync`

