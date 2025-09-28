import argparse
import pandas as pd
import sys
import os
import json
from model_handler import preprocess_text, load_model
from backendPredict import classify_text_batch


parser = argparse.ArgumentParser(
    description="Классификация токсичности текста из командной строки.",
    epilog="Пример 1 (один текст): python predict_cli.py --text 'какой же ты урод' \n"
           "Пример 2 (файл): python predict_cli.py --input-file data.csv --output-file results.csv",
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument(
    '--text',
    type=str,
    help='Текст для классификации.'
)

parser.add_argument(
    '--input-file',
    type=str,
    help='Путь к .csv файлу для обработки. Должен содержать колонки "ID" и "text".'
)
parser.add_argument(
    '--output-file',
    type=str,
    default='cli_results.csv',
    help='Путь для сохранения .csv файла с результатами.'
)


parser.add_argument(
    '--model-path',
    type=str,
    default='./my_toxic_classifier_final_model_v5',
    help='Путь к папке с обученной моделью.'
)


if __name__ == '__main__':
    args = parser.parse_args()

    if not args.text and not args.input_file:
        print("Ошибка: Необходимо указать либо --text для одного сообщения, либо --input-file для файла.")
        parser.print_help()
        sys.exit(1)

    try:
        model, tokenizer = load_model(args.model_path)
    except Exception as e:
        print(f"Критическая ошибка при загрузке модели из '{args.model_path}': {e}")
        sys.exit(1)


    if args.text:
        print(f"Классификация текста: '{args.text}'")
        processed_text = preprocess_text(args.text)

        data_to_classify = [{'id': 0, 'text': processed_text}]
        results = classify_text_batch(data_to_classify, model, tokenizer)

        if results:
            print("\nРезультат:")
            print(json.dumps(results[0], indent=2, ensure_ascii=False))
        else:
            print("Не удалось получить результат.")


    if args.input_file:
        print(f"Начинаю обработку файла: {args.input_file}")

        if not os.path.exists(args.input_file):
            print(f"Ошибка: Файл не найден по пути: {args.input_file}")
            sys.exit(1)

        try:
            df = pd.read_csv(args.input_file)
            if 'ID' not in df.columns or 'text' not in df.columns:
                print('Ошибка: CSV файл должен содержать колонки "ID" и "text"')
                sys.exit(1)


            df['text'] = df['text'].apply(preprocess_text)
            texts_with_ids = df.rename(columns={'ID': 'id'})[['id', 'text']].to_dict('records')

            results = classify_text_batch(texts_with_ids, model, tokenizer)

            result_df = pd.DataFrame(results)[['id', 'label', 'score']]
            result_df = result_df.rename(columns={'id': 'ID'})
            result_df.to_csv(args.output_file, index=False)

            print(f"\nОбработка завершена. Результаты сохранены в файл: {args.output_file}")

        except Exception as e:
            print(f"Произошла ошибка при обработке файла: {e}")
            sys.exit(1)