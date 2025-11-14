# Подключение к серверу с существующим ключом провайдера

## Ситуация

На сервере уже настроен SSH-ключ, сгенерированный провайдером (OpenStack/Nova):
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDGRXjzuKmf8qQadZcho8AsqFFC+7YP0ZbWIS3Yn/QhE0YKTzpa158PEkKX/jxphQGksb4SumyEHKfQQdYlr4jJcQ4IcVsAIWGw0FvKnznM9Jv4UgVapU5wOOzH66HemkG1CLgYt9kqxSrJEwlyvU/+08d+zw5KH3Y+1BuspW4Ce6AmiKSPSWhEw1cWerbCWHMpCZQYnGxcsKNXXO/rPlICGjuJwFoQlHeaSl0deDabM5535ZgmVez8Tt3MYtCDGtu+mEKjviMpGG9H1A5uJCUYyuHwM7i6qhiGGmopEaztycg80aQgh3KKA/v6aPDZWkp9/gyPpV23QF5kg/rPX0VB Generated-by-Nova
```

## Решение 1: Скачать приватный ключ из панели управления

### Шаги:

1. **Войдите в панель управления OpenStack/провайдера**

2. **Найдите раздел:**
   - "Key Pairs" (OpenStack)
   - "SSH Keys"
   - "Access & Security"
   - "Compute" → "Key Pairs"

3. **Найдите ключ с именем**, которое соответствует серверу (или скачайте все доступные)

4. **Скачайте приватный ключ** (обычно кнопка "Download" или "Скачать")

5. **Сохраните ключ локально:**
   ```bash
   # Создайте директорию если её нет
   mkdir -p ~/.ssh
   
   # Сохраните ключ (например, как server_key)
   # Скопируйте содержимое скачанного файла в:
   nano ~/.ssh/server_key
   
   # Установите правильные права
   chmod 600 ~/.ssh/server_key
   ```

6. **Подключитесь используя этот ключ:**
   ```bash
   ssh -i ~/.ssh/server_key user@195.209.210.204
   ```
   
   Попробуйте разные имена пользователей:
   ```bash
   ssh -i ~/.ssh/server_key root@195.209.210.204
   ssh -i ~/.ssh/server_key ubuntu@195.209.210.204
   ssh -i ~/.ssh/server_key debian@195.209.210.204
   ssh -i ~/.ssh/server_key centos@195.209.210.204
   ```

## Решение 2: Добавить ваш ключ через веб-консоль

Если у вас есть доступ к веб-консоли сервера:

1. **Откройте веб-консоль** через панель управления

2. **Войдите в систему** (обычно `root` или другой пользователь)

3. **Добавьте ваш ключ:**
   ```bash
   # Определите текущего пользователя
   whoami
   
   # Создайте директорию для SSH
   mkdir -p ~/.ssh
   chmod 700 ~/.ssh
   
   # Добавьте ваш публичный ключ
   echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPOm0n2CNgwl1TTc1i+o4VNDukmmBvVlmip+9GMXzut+ finance-bot-server" >> ~/.ssh/authorized_keys
   
   # Установите права
   chmod 600 ~/.ssh/authorized_keys
   ```

4. **Теперь можно подключаться вашим ключом:**
   ```bash
   ssh user@195.209.210.204
   ```

## Решение 3: Настроить SSH config для удобства

Создайте файл `~/.ssh/config`:

```bash
nano ~/.ssh/config
```

Добавьте:

```
Host finance-bot
    HostName 195.209.210.204
    User root
    IdentityFile ~/.ssh/server_key
    IdentitiesOnly yes
```

Теперь можно подключаться просто:
```bash
ssh finance-bot
```

## Определение правильного пользователя

Если не знаете имя пользователя, попробуйте:

1. **Проверить в панели управления:**
   - Информация о сервере
   - Детали инстанса
   - Метаданные

2. **Попробовать стандартные имена:**
   - `root` (часто для CentOS/RHEL)
   - `ubuntu` (для Ubuntu)
   - `debian` (для Debian)
   - `admin` (иногда используется)
   - `cloud-user` (для некоторых облачных образов)

3. **Использовать веб-консоль:**
   - Войдите через веб-консоль
   - Выполните `whoami` чтобы узнать пользователя

## Проверка подключения

После настройки:

```bash
# С ключом провайдера
ssh -i ~/.ssh/server_key user@195.209.210.204

# Или с вашим ключом (если добавили)
ssh user@195.209.210.204
```

## Следующие шаги

После успешного подключения:
1. Обновите систему: `sudo apt update && sudo apt upgrade -y` (для Ubuntu/Debian)
2. Следуйте инструкции из `docs/deployment.md` для развертывания бота

