import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# --- 1. ФУНКЦИЯ НОРМАЛИЗАЦИИ ТЕКСТА (взята из вашего второго скрипта) ---
def preprocess_text(text: str) -> str:
    """
    Улучшенная функция предобработки текста:
    1. Приводит к нижнему регистру.
    2. Заменяет очевидные гомоглифы.
    3. Удаляет эмодзи и большинство символов, не являющихся текстом.
    4. Сохраняет важные знаки препинания (.,!?-).
    5. Нормализует пробелы.
    """
    text = str(text).lower()

    # Замена гомоглифов
    homoglyphs = {
        # Кириллица -> Кириллица (нормализация)
        'ё': 'е',

        # Латиница -> Кириллица
        'a': 'а', 'b': 'в', 'c': 'с', 'e': 'е', 'h': 'н',
        'i': 'и', 'k': 'к', 'l': 'л', 'm': 'м', 'o': 'о',
        'p': 'р', 's': 'с', 't': 'т', 'x': 'х', 'y': 'у',

        # Символы и цифры -> Кириллица
        '@': 'а', '$': 'с', '0': 'о', '3': 'з', '4': 'ч',
        '6': 'б', '8': 'в', '1': 'и',
    }
    for en_char, ru_char in homoglyphs.items():
        text = text.replace(en_char, ru_char)

    # Удаление эмодзи и специальных символов
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)

    # Оставляем кириллицу, пробелы и некоторые знаки препинания
    text = re.sub(r'[^а-я\s.,!?-]', '', text)
    # Нормализуем пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- 2. ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА ---
# Укажите путь к папке, где лежит ваша обученная модель.
MODEL_PATH = "./my_toxic_classifier_final_model_v8"

print(f"Загрузка модели из: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Переводим модель в режим оценки (важный шаг)
model.eval()

# --- 3. ПОДГОТОВКА ТЕКСТА ---

# Введите сюда любой текст на русском языке
# Пример с гомоглифами и лишними символами: "м@ть!!!!!"
# Другой пример: "это просто yжacно, я в ярости"
text_to_classify = "ебята"
print(f"\nИсходный текст: '{text_to_classify}'")

# **НОВЫЙ ШАГ**: Нормализуем текст перед классификацией
normalized_text = preprocess_text(text_to_classify)
print(f"Нормализованный текст: '{normalized_text}'")

# --- 4. ТОКЕНИЗАЦИЯ И ПРЕДСКАЗАНИЕ ---

# Токенизируем текст, добавляя специальные токены и преобразуя в тензоры PyTorch
# ВАЖНО: передаем в токенизатор уже нормализованный текст
inputs = tokenizer(normalized_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

# Отключаем вычисление градиентов для ускорения инференса
with torch.no_grad():
    # Передаем токенизированные данные в модель
    logits = model(**inputs).logits

# --- 5. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТА ---

# Logits - это "сырые" выходные значения модели для каждого класса.
# Чтобы получить предсказанный класс, находим индекс с максимальным значением.
predicted_class_id = logits.argmax().item()

# Получаем название метки по её ID (0 -> 'not_toxic', 1 -> 'toxic')
predicted_label = model.config.id2label[predicted_class_id]

# (Опционально) Рассчитаем вероятности для каждого класса с помощью Softmax
probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze() # squeeze() убирает лишние измерения
confidence_score = probabilities[predicted_class_id].item()

# --- 6. ВЫВОД РЕЗУЛЬТАТА ---

print(f"\nРезультат классификации:")
print(f"  - Предсказанная метка: **{predicted_label}**")
print(f"  - Уверенность модели: **{confidence_score:.2%}**\n")

# Посмотрим вероятности для всех классов
print("Вероятности по классам:")
for i, probability in enumerate(probabilities):
    label = model.config.id2label[i]
    print(f"  - {label}: {probability.item():.2%}")