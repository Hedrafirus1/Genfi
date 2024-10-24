import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sqlite3
import aiosqlite
import asyncio
from telebot.async_telebot import AsyncTeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import logging
import json
from pathlib import Path
import os
import requests
import base64
from io import BytesIO
import subprocess
import aiohttp
import threading
import queue
from webuiapi import WebUIApi
from webuiapi import WebUIApi
from PIL import Image

#3.1

# Очередь для обработки задач
task_queue = asyncio.Queue()

# Здесь будет храниться текущая модель
current_model = None

# URL сервера Stable Diffusion
url_txt2txt = "http://127.0.0.1:7860/sdapi/v1/txt2img"
url_img2img = "http://127.0.0.1:7860/sdapi/v1/img2img"
url_progress = "http://127.0.0.1:7860/sdapi/v1/progress"
url_opt = "http://127.0.0.1:7860/sdapi/v1/options"

# Ollama параметры
#client = Client(host='http://localhost:11434')

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Получаем путь к директории, где находится main.py
project_dir = Path(__file__).parent

# Указываем пути к папкам
imgs_dir = project_dir / 'imgs'
tests_dir = project_dir / 'tests'
datasets_dir = project_dir / 'datasets'
db_dir = project_dir / 'db'
logs_dir = project_dir / 'logs'
model_gender_dir = project_dir / 'model_gender'
patch_dir = project_dir / 'patch'
backup_dir = project_dir / 'backup'

# Загрузка модели и токенизатора для модели определения гендера
model = load_model(model_gender_dir / 'gender_model.h5')
tokenizer = joblib.load(model_gender_dir / 'tokenizer.pkl')
label_encoder = joblib.load(model_gender_dir / 'label_encoder.pkl')

max_seq_length = 100

AUTHORIZED_USERS = [66666666666666666]
API_TOKEN = '6666666666666666666'

bot = AsyncTeleBot(API_TOKEN)

# Словарь для хранения состояния пользователей
user_states = {}
img_queue = [0]
user_state_gen = {}
msg2del = {}
user_images = {}
img2img = {}

# Функция для обработки задач из очереди
async def process_queue():
    while True:
        # Извлекаем следующую задачу из очереди
        task_data = await task_queue.get()

        # Обрабатываем задачу
        user_msg, user_id, chat_id, coins, back_msg_id = task_data

        # Генерируем изображение
        image_io = await asyncio.get_event_loop().run_in_executor(None, generate_image, user_msg)

        # Ставим кнопку для повторной генерации
        markup = InlineKeyboardMarkup()
        btn = InlineKeyboardButton(text='Повторить генерацию', callback_data='again')
        markup.add(btn)

        # Отправляем сгенерированное фото
        await bot.send_photo(user_id, image_io, caption=f"Осталось монет: {coins}\n{user_msg}", reply_markup=markup)

        # Удаляем предыдущее сообщение
        await bot.delete_message(user_id, message_id=back_msg_id)

        # Уменьшаем глобальный счётчик запросов
        img_queue[0] = img_queue[0] - 1

        # Помечаем задачу как завершенную
        task_queue.task_done()


# Синхронная функция для взаимодействия с API Stable Diffusion
def generate_image(prompt):
    user_tag = prompt.lower().strip().split(",")[0]
    prompt_xl = prompt.replace('!res,', '').strip()
    if user_tag == "!res":
        option_payload = {
            "sd_model_checkpoint": "dreamshaperXL_lightningDPMSDE.safetensors [fdbe56354b]"
        }
        payload = {
            "prompt": prompt_xl,
            "negative_prompt": "(deformed, distorted, disfigured:1.1), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
            "steps": 4,
            "seed": -1,
            "subseed": -1,
            "sampler_name": "DPM++ SDE",
            "scheduler": "Karras",
            "width": 1024,
            "height": 1024,
            "cfg_scale": 2
        }
    else:
        option_payload = {
            "sd_model_checkpoint": "dreamshaper_8.safetensors [879db523c3]"
        }
        payload = {
            "prompt": prompt,
            "negative_prompt": "(deformed, distorted, disfigured:1.1), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
            "steps": 20,
            "seed": -1,
            "subseed": -1,
            "enable_hr": "true",
            "sampler_name": "Euler a",
            "width": 512,
            "height": 512,
            "cfg_scale": 7,
            "hr_scale": 2,
            "hr_upscaler": "4x_NMKD-Siax_200k",
            "denoising_strength": 0.3
        }

    # Синхронный запрос к API
    requests.post(url_opt, json=option_payload)
    response = requests.post(url_txt2txt, json=payload)
    if response.status_code == 200:
        r = response.json()

        # Декодируем изображение из base64
        image_data = base64.b64decode(r['images'][0])
        image_io = BytesIO(image_data)
        image_io.name = 'output.png'
        return image_io
    else:
        raise Exception("Ошибка при запросе")


# Функция для добавления задачи в очередь
async def add_task_to_queue(user_msg, user_id, chat_id, coins, back_msg_id):
    await task_queue.put((user_msg, user_id, chat_id, coins, back_msg_id))


# Функция для предсказания гендера по новому сообщению
def predict_gender(message):
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_seq_length)
    pred = model.predict(padded)
    gender = label_encoder.inverse_transform([pred.argmax(axis=-1)[0]])
    if gender == 'мужчина':
        return 'мужчина'
    elif gender == 'женщина':
        return 'женщина'
    else:
        return '❓'


# Функция для отправки вопроса
async def send_question(chat_id):
    # Проверяем наличие активного теста у пользователя
    if chat_id not in user_states or 'questions' not in user_states[chat_id]:
        await bot.send_message(chat_id, "Пожалуйста, выберите тест с помощью команды /tests.")
        return

    state = user_states[chat_id]
    questions = state['questions']
    current_index = state['current_question_index']

    # Проверяем, есть ли еще вопросы
    if current_index >= len(questions):
        await bot.send_message(chat_id,
                               f"Тест '{state['test_name']}' завершен! Ваш результат: {state['score']} из {len(questions)}")
        del user_states[chat_id]
        return

    question_data = questions[current_index]
    question_text = question_data['question']
    options = question_data['options']
    correct_option = question_data['correct_option']

    # Сохраняем правильный ответ в состоянии пользователя
    state['correct_option'] = correct_option

    # Создание инлайн-клавиатуры с вариантами ответов и кнопкой завершения
    markup = InlineKeyboardMarkup()

    # Кнопки для вариантов ответов
    buttons = []
    for i in range(len(options)):
        button = InlineKeyboardButton(text=str(i + 1), callback_data=f"quiz_answer_{i}")
        buttons.append(button)

    # Располагаем кнопки по две в ряд
    for i in range(0, len(buttons), 2):
        markup.row(*buttons[i:i + 2])

    # Кнопка для принудительного завершения теста
    terminate_button = InlineKeyboardButton(text="Завершить тест", callback_data="quiz_terminate")
    markup.add(terminate_button)

    # Формируем текст вопроса с вариантами ответов
    question_number = current_index + 1
    question_text_with_options = f"Вопрос {question_number}:\n{question_text}\n\n"
    for idx, option in enumerate(options, 1):
        question_text_with_options += f"{idx}. {option}\n"

    if 'message_id' in state:
        # Если message_id уже есть, обновляем сообщение
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=state['message_id'],
            text=question_text_with_options,
            reply_markup=markup
        )
    else:
        # Отправляем новое сообщение и сохраняем message_id
        sent_message = await bot.send_message(chat_id, question_text_with_options, reply_markup=markup,
                                              disable_notification=True)
        state['message_id'] = sent_message.message_id


# Функция для добавления сообщения в базу данных
async def add_message(user_id, role, content):
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_data.db') as db:
        await db.execute('''
            INSERT INTO messages (user_id, role, content)
            VALUES (?, ?, ?)
        ''', (user_id, role, content))
        await db.commit()


# Функция для извлечения истории сообщений пользователя
async def get_message_history(user_id, limit=10):
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_data.db') as db:
        async with db.execute('''
            SELECT role, content FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        ''', (user_id, limit)) as cursor:
            messages = await cursor.fetchall()

    # Возвращаем список сообщений в нужном формате
    return [{'role': role, 'content': content} for role, content in reversed(messages)]


# Обработка команды /start
@bot.message_handler(commands=['start'])
async def handle_start(message):
    start_coin = 15
    chat_id = message.chat.id
    user_id = message.from_user.id
    username = message.from_user.username
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_users.db') as db:
        await db.execute('''
                INSERT OR IGNORE INTO users (user_id, username, coins)
                VALUES (?, ?, ?)
            ''', (user_id, username, start_coin))
        await db.commit()
        try:
            # Получение данных из таблицы
            async with db.execute('''
                    SELECT user_id, username, coins FROM users WHERE user_id = ?
                ''', (user_id,)) as cursor:
                user_row = await cursor.fetchone()
                await bot.send_message(chat_id=chat_id, text=f"{user_row}")
        except:
            pass
    await bot.delete_message(chat_id=chat_id, message_id=message.id)
    await bot.send_message(chat_id,
                           text="Ну, начнём!\nМеня зовут Генфи. Я могу рисовать любые картинки. Могу определить гендер человека по написанию текста.\nВсе команды ты найдёшь в меню!\nДерзай! Я надеюсь мы подружимся. Я всегда стараюсь обновляться и приносить новые фичи.\nОбычно я работаю с утра до вечера (ночью я сплю)",
                           disable_notification=True)


# Обработка команды /help
@bot.message_handler(commands=['help'])
async def handle_help(message):
    chat_id = message.chat.id
    await bot.delete_message(chat_id=chat_id, message_id=message.id)
    await bot.send_message(chat_id,
                           text="Требуется помощь)?\n/gender - определение гендера по тексту\n/tests - разные тестики\nА вообще скоро будет ещё больше фишек, с которыми я помогу!")


# Обработка команды /tests
@bot.message_handler(commands=['tests'])
async def handle_tests_command(message):
    chat_id = message.chat.id

    # Получаем список доступных тестовых файлов
    test_files = [f for f in tests_dir.iterdir() if f.suffix == '.json']

    if not test_files:
        await bot.send_message(chat_id, "Нет доступных тестов.")
        return

    # Создаем инлайн-клавиатуру с кнопками для каждого теста
    markup = InlineKeyboardMarkup()
    for test_file in test_files:
        test_name = Path(test_file).stem  # Имя теста без расширения
        button = InlineKeyboardButton(text=test_name, callback_data=f"test_select_{test_name}")
        markup.add(button)

    await bot.delete_message(chat_id=chat_id, message_id=message.id)
    await bot.send_message(chat_id, "Выберите тест:", reply_markup=markup, disable_notification=True)


# Обработка команды /gender
@bot.message_handler(commands=['gender'])
async def handle_gender(message):
    chat_id = message.chat.id
    user_info = message.from_user
    user_id = user_info.id
    username = user_info.username
    first_name = user_info.first_name
    last_name = user_info.last_name
    user_msg = message.text
    await bot.delete_message(chat_id=chat_id, message_id=message.id)
    await bot.send_message(chat_id, text="Отправьте мне текст для распознавания гендера)).")
    user_state_gen[chat_id] = "waiting_for_text"
    # Подключение к базе данных и сохранение сообщения
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_data.db') as db:
        await db.execute('''
                    INSERT INTO users (user_id, user_name, user_surname, username, user_message)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, first_name, last_name, username, user_msg))
        await db.commit()


@bot.message_handler(commands=['id'])
async def handle_id(message):
    chat_id = message.chat.id
    await bot.delete_message(chat_id=chat_id, message_id=message.id)
    await bot.send_message(chat_id=message.from_user.id, text=f"{message.from_user.id}", disable_notification=True)


@bot.message_handler(commands=['image'])
async def handle_id(message):
    chat_id = message.chat.id
    await bot.delete_message(chat_id=chat_id, message_id=message.id)
    msg = await bot.send_message(chat_id,
                                 text="Опишите, что хотите увидеть!)\nПример: yellow cat, (fluffy:1.2), dark background, real photo")
    msg2del[chat_id] = msg.message_id
    user_state_gen[chat_id] = "txt2img"


@bot.message_handler(commands=['coins'])
async def handle_coins(message):
    user_id = message.from_user.id
    await bot.delete_message(chat_id=user_id, message_id=message.id)
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_users.db') as db:
        async with db.execute("SELECT coins FROM users WHERE user_id = ?", (user_id,)) as cursor:
            result = await cursor.fetchone()  # Получаем первую запись из результата
    await bot.send_message(chat_id=user_id, text=f"Твой счёт монет для генерации картиночек: {result[0]}")


@bot.message_handler(commands=['giveDB'])
async def handle_give(message):
    user_id = message.from_user.id
    if user_id not in AUTHORIZED_USERS:
        return
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_users.db') as db:
        async with db.execute("SELECT * FROM users") as cursor:
            rows = await cursor.fetchall()
        result = ""
        for row in rows:
            result += " | ".join([str(item) for item in row]) + "\n"
        await bot.send_message(user_id, f"Table:\n{result}")


@bot.message_handler(commands=['progress'])
async def get_progress(message):
    user_id = message.from_user.id
    await bot.delete_message(chat_id=user_id, message_id=message.id)
    await bot.send_message(user_id, text=f"Картинок в очереде: {img_queue[0]}")


# Обработчик команды /sendPN
@bot.message_handler(commands=['sendPN'])
async def send_personal_notification(message):
    with open(patch_dir / "patchnote.json", 'r', encoding='utf-8') as file:
        text = json.load(file)
    PATCHNOTE = text['patch']
    user_id = message.from_user.id

    # Проверка авторизации
    if user_id not in AUTHORIZED_USERS:
        await bot.reply_to(message, "Ух, милашка. Не нужно этого делать)")
        return

    # Получение списка пользователей из базы данных
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_users.db') as db:
        cursor = await db.execute('SELECT user_id FROM users')
        users = await cursor.fetchall()

    # Отправка сообщения каждому пользователю
    for (user_id,) in users:
        try:
            await bot.send_message(user_id, PATCHNOTE)
            # Задержка для соблюдения лимитов Telegram
            await asyncio.sleep(0.2)  # Вы можете настроить задержку
        except Exception as e:
            print(f"Не удалось отправить сообщение пользователю {user_id}: {e}")

    await bot.reply_to(message, f"Патчноут разослала!")


# Обработчик для получения файла '.json'
@bot.message_handler(content_types=['document'])
async def handle_document(message):
    chat_id = message.chat.id
    user_id = message.from_user.id

    if chat_id not in AUTHORIZED_USERS:
        await bot.send_message(chat_id, text="Пока мне не нужны твои файлы.")
        return

    # Проверка авторизации пользователя
    if user_id not in AUTHORIZED_USERS:
        await bot.reply_to(message, "Нет, детка! Не нужно.")
        return

    # Получение информации о файле
    file_info = await bot.get_file(message.document.file_id)
    file_name = message.document.file_name

    # Проверка имени файла
    if file_name in (
            'Photonic1.json', 'Photonic2.json', 'Photonic3.json', 'Photonic4.json', 'Photonic5.json', 'Photonic6.json',
            'Photonic7.json', 'Photonic8.json', 'Photonic9.json', 'Photonic10.json', 'Genfi.json'):
        # Загрузка файла
        downloaded_file = await bot.download_file(file_info.file_path)

        # Сохранение файла в папку
        with open(tests_dir / f"{file_name}", 'wb') as new_file:
            new_file.write(downloaded_file)

        await bot.reply_to(message, "Файл теста обновлен и перезагружен.")
    elif file_name == 'patchnote.json':
        # Загрузка файла
        downloaded_file = await bot.download_file(file_info.file_path)

        # Сохранение файла patchnote в папку
        with open(patch_dir / f"{file_name}", 'wb') as new_file:
            new_file.write(downloaded_file)

        await bot.reply_to(message, "Файл патча обновлен и перезагружен.")
    elif file_name == 'main_t.py':
        # Загрузка файла
        downloaded_file = await bot.download_file(file_info.file_path)

        # Сохранение файла patchnote в папку
        with open(project_dir / f"{file_name}", 'wb') as new_file:
            new_file.write(downloaded_file)

        # Уведомляем, что файл загружен
        await bot.reply_to(message, "Мой геном загружен. Я перезапускаюсь ...")

        # Запуск .bat файла
        subprocess.Popen(['start', project_dir / 'restart_bot.bat'], shell=True)

        # Завершаем текущий процесс
        os._exit(0)
    else:
        await bot.reply_to(message, "Пожалуйста, отправьте нужный файл.")


# Обработчик нажатий на инлайн-кнопки выбора тестов
@bot.callback_query_handler(func=lambda call: call.data.startswith('test_select_'))
async def handle_test_selection(call):
    chat_id = call.message.chat.id
    test_name = call.data[len('test_select_'):]  # Извлекаем имя теста из callback_data

    # Путь к выбранному файлу теста
    test_file_path = tests_dir / f'{test_name}.json'

    if not test_file_path.exists():
        await bot.send_message(chat_id, "Выбранный тест не найден.")
        return

    # Загрузка вопросов из выбранного теста
    with open(test_file_path, 'r', encoding='utf-8') as file:
        test_questions = json.load(file)

    # Инициализируем состояние пользователя для выбранного теста
    user_states[chat_id] = {
        'test_name': test_name,
        'questions': test_questions,
        'current_question_index': 0,
        'score': 0
    }

    # Отправляем первый вопрос
    await send_question(chat_id)
    await bot.delete_message(chat_id=chat_id, message_id=call.message.id)


# Обработчик нажатий на инлайн-кнопки выбора ответов
@bot.callback_query_handler(func=lambda call: call.data.startswith('quiz_answer_') or call.data == 'quiz_terminate')
async def handle_quiz_callback(call):
    chat_id = call.message.chat.id
    message_id = call.message.message_id

    if chat_id not in user_states:
        await bot.send_message(chat_id, "Пожалуйста, начните тест с помощью команды /tests.")
        return

    state = user_states[chat_id]

    if call.data == 'quiz_terminate':
        # Принудительное завершение теста
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=state['message_id'],
            text=f"Тест '{state['test_name']}' был завершен досрочно. Ваш результат: {state['score']} из {len(state['questions'])}",
            reply_markup=None
        )
        del user_states[chat_id]
        return

    # Обработка выбранного ответа
    user_answer = int(call.data[len('quiz_answer_'):])

    correct_option = state['correct_option']
    current_index = state['current_question_index']
    questions = state['questions']

    # Получаем текущий вопрос и варианты
    question_data = questions[current_index]
    question_text = question_data['question']
    options = question_data['options']

    # Проверка ответа и формирование символа
    if user_answer == correct_option:
        response_symbol = "✅"
        state['score'] += 1
    else:
        response_symbol = "❌"

    # Формируем обновленный текст вопроса с галочкой или крестиком
    question_number = current_index + 1
    updated_question_text = f"Вопрос {question_number}:\n{question_text}\n\n"
    for idx, option in enumerate(options):
        if idx == user_answer:
            # Добавляем символ (✅ или ❌) рядом с выбранным вариантом
            updated_question_text += f"{idx + 1}. {option} {response_symbol}\n"
        else:
            updated_question_text += f"{idx + 1}. {option}\n"

    # Удаляем клавиатуру из сообщения
    await bot.edit_message_text(
        chat_id=chat_id,
        message_id=state['message_id'],
        text=updated_question_text,
        reply_markup=None
    )

    # Пауза перед показом следующего вопроса (опционально)
    await asyncio.sleep(1)

    # Переходим к следующему вопросу
    state['current_question_index'] += 1

    # Проверяем, есть ли еще вопросы
    if state['current_question_index'] < len(questions):
        # Отправляем следующий вопрос, редактируя то же сообщение
        await send_question(chat_id)
    else:
        # Тест завершен, обновляем сообщение с результатом
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=state['message_id'],
            text=f"Тест '{state['test_name']}' завершен! Ваш результат: {state['score']} из {len(questions)}",
            reply_markup=None
        )
        # Удаляем состояние пользователя
        del user_states[chat_id]


# Обработчик команды /image
@bot.message_handler(func=lambda message: user_state_gen.get(message.chat.id) == "txt2img")
async def get_image(message):
    img_queue[0] = img_queue[0] + 1
    user_id = message.from_user.id
    # Очистка состояния пользователя
    user_state_gen.pop(user_id, None)
    await bot.delete_message(chat_id=user_id, message_id=msg2del[user_id])
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_users.db') as db:
        async with db.execute("SELECT coins FROM users WHERE user_id = ?", (user_id,)) as cursor:
            result = await cursor.fetchone()  # Получаем первую запись из результата
            if result[0] > 0:
                coins = result[0] - 1  # Возвращаем значение coins

                # Обновляем значение coins в базе данных, уменьшаем на 1
                await db.execute("UPDATE users SET coins = coins - 1 WHERE user_id = ?", (user_id,))
                await db.commit()  # Не забываем сделать commit, чтобы изменения сохранились

                """Добавляем запрос в очередь для генерации изображения"""
                user_msg = message.text
                try:
                    # Используем run_in_executor для асинхронного выполнения функции generate_image
                    #loop = asyncio.get_event_loop()
                    await bot.delete_message(user_id, message.id)
                    msg = await bot.send_message(user_id, text=f"Твоя картинка в очереди: {img_queue[0]}")
                    back_msg_id = msg.message_id
                    #image_io = await loop.run_in_executor(None, generate_image, user_msg)

                    #markup = InlineKeyboardMarkup()
                    #btn = InlineKeyboardButton(text='Повторить генерацию', callback_data='again')
                    #markup.add(btn)

                    # Отправляем изображение пользователю
                    await add_task_to_queue(user_msg, user_id, user_id, coins, back_msg_id)
                    #await bot.send_photo(user_id, image_io, caption=f"Осталось монет: {coins}\n{user_msg}",
                    #reply_markup=markup)
                except Exception as e:
                    await bot.send_message(user_id, e)
            else:
                await bot.send_message(chat_id=user_id,
                                       text="У тебя нет монеток(\nСвяжись с @Hedrafirus, чтобы он добавил немного.")


@bot.callback_query_handler(func=lambda call: call.data == 'again')
async def handle_again_img(call):
    user_id = call.message.chat.id
    message_prompt = call.message.caption.split("\n")[1]
    img_queue[0] = img_queue[0] + 1
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_users.db') as db:
        async with db.execute("SELECT coins FROM users WHERE user_id = ?", (user_id,)) as cursor:
            result = await cursor.fetchone()  # Получаем первую запись из результата
            if result[0] > 0:
                coins = result[0] - 1  # Возвращаем значение coins

                # Обновляем значение coins в базе данных, уменьшаем на 1
                await db.execute("UPDATE users SET coins = coins - 1 WHERE user_id = ?", (user_id,))
                await db.commit()  # Не забываем сделать commit, чтобы изменения сохранились

                """Добавляем запрос в очередь для генерации изображения"""
                user_msg = message_prompt
                try:
                    msg = await bot.send_message(user_id,
                                                 text=f"Твоя картинка в очереди: {img_queue[0]}")
                    back_msg_id = msg.message_id
                    await add_task_to_queue(user_msg, user_id, user_id, coins, back_msg_id)
                except Exception as e:
                    await bot.send_message(user_id, e)
            else:
                await bot.send_message(chat_id=user_id,
                                       text="У тебя нет монеток(\nСвяжись с @Hedrafirus, чтобы он добавил немного.")


# Ремейк фотки. Уменьшаем входное изображение с пропорциями
@bot.message_handler(content_types=['photo'])
async def remake_image(message):
    chat_id = message.chat.id
    await bot.reply_to(message, text="Не забегай вперёд) Скоро будет...")
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_users.db') as db:
        async with db.execute("SELECT coins FROM users WHERE user_id = ?", (chat_id,)) as cursor:
            result = await cursor.fetchone()  # Получаем первую запись из результата
            if result[0] > 0:
                coins = result[0] - 1  # Возвращаем значение coins
                # Обновляем значение coins в базе данных, уменьшаем на 1
                await db.execute("UPDATE users SET coins = coins - 1 WHERE user_id = ?", (chat_id,))
                await db.commit()  # Не забываем сделать commit, чтобы изменения сохранились

                # Получаем файл изображения с сервера Telegram
                file_info = await bot.get_file(message.photo[-1].file_id)
                file = await bot.download_file(file_info.file_path)
                image = Image.open(BytesIO(file))
                original_size = image.size
                if original_size[0] / original_size[1] > 1:
                    new_size = (round(512 * (original_size[0] / original_size[1])), 512)
                else:
                    new_size = (512, round(512 * (original_size[1] / original_size[0])))

                img2img[chat_id] = new_size
                # Уменьшаем изображение пропорционально
                resized_image = image.resize(new_size, Image.LANCZOS)
                # Сохраняем уменьшенное изображение в память
                img_byte_arr = BytesIO()
                resized_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                # Преобразуем изображение в base64
                img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

                # Сохраняем изображение для пользователя, чтобы потом использовать в img2img
                user_images[chat_id] = img_base64

                # Сообщаем пользователю, что изображение уменьшено и ждем промпт
                await bot.send_message(chat_id, "Отправьте мне промпт для преобразования.")
            else:
                await bot.send_message(chat_id=chat_id,
                                       text="У тебя нет монеток(\nСвяжись с @Hedrafirus, чтобы он добавил немного.")


# Генерация ремейка
@bot.message_handler(func=lambda message: message.chat.id in user_images)
async def get_reimage(message):
    chat_id = message.chat.id
    prompt = message.text
    # Получаем сохраненное изображение в формате base64
    img_base64 = user_images.pop(chat_id)
    payload = {
        "prompt": prompt,
        "negative_prompt": "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
        "init_images": [img_base64],  # Отправляем уменьшенное изображение
        "steps": 30,  # Количество шагов генерации
        "denoising_strength": 0.3,
        "width": img2img[chat_id][0],
        "height": img2img[chat_id][1],
        "sampler_name": "Heun",
        "cfg_scale": 8,
        "restore_faces": "false",
        "script_name": "SD upscale",
        "script_args": [None, 64, "4x_NMKD-Siax_200k", 1.3]
    }
    # Отправляем запрос к Stable Diffusion API
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url_img2img, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    # Извлекаем изображение из ответа
                    img_base64 = result['images'][0]
                    img_data = base64.b64decode(img_base64)

                    # Отправляем преобразованное изображение пользователю
                    image_io = BytesIO(img_data)
                    image_io.name = 'output.png'
                    await bot.send_photo(chat_id, image_io)
                else:
                    await bot.send_message(chat_id, "Ошибка при генерации изображения через img2img.")
    except Exception as e:
        await bot.send_message(chat_id, f"Произошла ошибка: {str(e)}")


# Начислить монетки
@bot.message_handler(func=lambda message: '/uc' in message.text.lower())
async def update_coins(message):
    user_id = message.from_user.id
    if user_id not in AUTHORIZED_USERS:
        await bot.reply_to(message, "Нет, детка! Не нужно.")
        return
    user_msg = message.text.lower().replace('/uc ', '').strip().split()
    await bot.send_message(chat_id=user_id, text=f"{user_msg}")
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_users.db') as db:
        await db.execute('''UPDATE users SET coins = ? WHERE user_id = ?''', (user_msg[1], user_msg[0]))
        await db.commit()  # Не забываем сделать commit, чтобы изменения сохранились


# Обработчик текстовых сообщений по кнопке для ввода текста определения гендера
@bot.message_handler(func=lambda message: user_state_gen.get(message.chat.id) == "waiting_for_text")
async def handle_other_gen(message):
    chat_id = message.chat.id
    user_input = message.text
    gender = predict_gender(user_input)
    if gender == "мужчина":
        await bot.reply_to(message, text="Текст написал мужчина или парень")
    else:
        await bot.reply_to(message, text="Текст написала женщина или девушка")

    # Убираем состояние пользователя, так как мы больше не ждем его сообщение
    user_state_gen.pop(chat_id, None)


# Обработчик сообщений для генерации ответа
@bot.message_handler(func=lambda message: True)
async def get_generate_messages(message):
    user_info = message.from_user
    user_id = message.chat.id
    user_input = message.text
    first_name = user_info.first_name
    last_name = user_info.last_name
    username = user_info.username

    # Подключение к базе данных и сохранение сообщения
    async with aiosqlite.connect(db_dir / 'telegram_bot_gf_data.db') as db:
        await db.execute('''
                INSERT INTO users (user_id, user_name, user_surname, username, user_message)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, first_name, last_name, username, user_input))
        await db.commit()

    # Сохраняем сообщение пользователя в базу данных
    await add_message(user_id, 'user', user_input)
    if not message.text == "!ch":
        await bot.reply_to(message,
                           "Я сейчас не могу говорить( Если эта функция действительно нужна, то напиши !ch\nМой содатель соберёт информацию о необходимости ведения чата. Я ведь тоже не бесконечная).")
    else:
        await bot.send_message(user_id, "Спасибо))")

# Инициализация базы данных и запуск бота
async def main():
    asyncio.create_task(process_queue())
    # Здесь должен быть основной код для работы бота
    await bot.infinity_polling()


if __name__ == '__main__':
    asyncio.run(main())
