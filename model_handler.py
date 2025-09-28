from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import pandas as pd
import logging

from backendPredict import classify_text_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = None
tokenizer = None


def load_model(model_path="./model"):
    global model, tokenizer
    try:
        logging.info(f"Начинаю загрузку модели из: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        logging.info("Модель успешно загружена.")
        logging.info(f"Метки модели (id2label): {model.config.id2label}")
    except Exception as e:
        logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель. {e}", exc_info=True)
        model, tokenizer = None, None


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = str(text).lower()

    homoglyphs = {
        'ё': 'е', 'a': 'а', 'b': 'в', 'c': 'с', 'e': 'е', 'h': 'н',
        'i': 'и', 'k': 'к', 'l': 'л', 'm': 'м', 'o': 'о', 'p': 'р',
        's': 'с', 't': 'т', 'x': 'х', 'y': 'у', '@': 'а', '$': 'с',
        '0': 'о', '3': 'з', '4': 'ч', '6': 'б', '8': 'в', '1': 'и',
    }
    for en_char, ru_char in homoglyphs.items():
        text = text.replace(en_char, ru_char)

    # Удаляем эмодзи
    emoji_pattern = re.compile("["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F900-\U0001F9FF"  # ...and so on
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(r'[^а-я\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_single(text: str):
    logging.info("--- Новый запрос на /predict ---")
    logging.info(f"Получен исходный текст: '{text}'")

    if not model or not tokenizer:
        logging.error("Модель не загружена. Невозможно выполнить предсказание.")
        raise RuntimeError("Модель не загружена.")

    normalized_text = preprocess_text(text)
    logging.info(f"Текст после предобработки: '{normalized_text}'")

    if not normalized_text:
        logging.warning("После очистки текст стал пустым. Возвращаю 'not_toxic'.")
        return {'id': 0, 'label': 'not_toxic', 'score': 1.0}

    data_to_classify = [{'id': 0, 'text': normalized_text}]

    results = classify_text_batch(data_to_classify, model, tokenizer)

    if not results:
        logging.error("Функция classify_text_batch вернула пустой результат.")
        return {'id': 0, 'label': 'error', 'score': 0.0}

    single_result = results[0]
    logging.info(f"Результат предсказания: {single_result}")

    return {
        'id': single_result['id'],
        'label': single_result['label'],
        'score': single_result['score']
    }


def predict_batch(df: pd.DataFrame):
    logging.info(f"--- Новый запрос на /upload_csv с {len(df)} строками ---")
    if not model or not tokenizer:
        logging.error("Модель не загружена. Невозможно выполнить предсказание.")
        raise RuntimeError("Модель не загружена.")

    df['text'] = df['text'].apply(preprocess_text)
    texts_with_ids = df.rename(columns={'ID': 'id'})[['id', 'text']].to_dict('records')

    results = classify_text_batch(texts_with_ids, model, tokenizer)

    result_df = pd.DataFrame(results)[['id', 'label']]
    result_df = result_df.rename(columns={'id': 'ID'})

    logging.info(f"Обработка CSV завершена. Возвращено {len(result_df)} результатов.")
    return result_df