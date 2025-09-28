from flask import Flask, render_template, request, jsonify, send_file
from model_handler import load_model, predict_single, predict_batch
import pandas as pd
import io

app = Flask(__name__)

load_model()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'Отсутствует поле "text" в запросе'}), 400

        comment_text = data['text']

        result = predict_single(comment_text)

        return jsonify(result)

    except Exception as e:
        print(f"Ошибка в /predict: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера при обработке текста'}), 500


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден в запросе'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Пустое имя файла'}), 400

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)

            if 'ID' not in df.columns or 'text' not in df.columns:
                return jsonify({'error': 'CSV файл должен содержать колонки "ID" и "text"'}), 400

            result_df = predict_batch(df)

            output_buffer = io.StringIO()
            result_df.to_csv(output_buffer, index=False)
            output_buffer.seek(0)

            mem = io.BytesIO()
            mem.write(output_buffer.getvalue().encode('utf-8'))
            mem.seek(0)
            output_buffer.close()

            return send_file(
                mem,
                as_attachment=True,
                download_name='results.csv',
                mimetype='text/csv'
            )

        except Exception as e:
            print(f"Ошибка в /upload_csv: {e}")
            return jsonify({'error': f'Ошибка при обработке файла: {e}'}), 500

    return jsonify({'error': 'Неверный формат файла. Пожалуйста, загрузите .csv файл'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5555)