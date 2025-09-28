document.addEventListener('DOMContentLoaded', () => {
    // Форма для одного комментария
    const commentForm = document.getElementById('commentForm');
    const commentInput = commentForm.querySelector('textarea');
    const commentList = document.getElementById('commentList');
    const moderationResult = document.getElementById('moderationResult');
    const companyTitle = document.querySelector('.company-title');

    // Новая форма для CSV
    const csvUploadForm = document.getElementById('csvUploadForm');
    const csvFileInput = document.getElementById('csvFile');
    const csvStatus = document.getElementById('csvStatus');


    // Устанавливаем название компании
    if (companyTitle) {
        companyTitle.textContent = "WB Security";
    }

    // --- ОБРАБОТЧИК ДЛЯ ОДНОГО КОММЕНТАРИЯ ---
    commentForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Предотвращаем перезагрузку страницы

        const commentText = commentInput.value.trim();
        if (commentText.length === 0) {
            return; // Не отправляем пустые комментарии
        }

        moderationResult.innerHTML = 'Анализируем...';
        moderationResult.style.color = '#999';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // Отправляем только текст, id здесь не нужен
                body: JSON.stringify({ text: commentText }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Ошибка сервера: ${response.status}`);
            }

            const result = await response.json();
            displayModerationResult(result);
            addCommentToList(commentText);
            commentInput.value = '';

        } catch (error) {
            console.error('Ошибка при отправке запроса:', error);
            moderationResult.innerHTML = `⚠️ Произошла ошибка: <br/> ${error.message}`;
            moderationResult.style.color = '#d9534f';
        }
    });

    // --- НОВЫЙ ОБРАБОТЧИК ДЛЯ CSV ФАЙЛА ---
    csvUploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const file = csvFileInput.files[0];
        if (!file) {
            csvStatus.textContent = 'Пожалуйста, выберите файл.';
            csvStatus.style.color = '#d9534f';
            return;
        }

        csvStatus.textContent = 'Загрузка и обработка... Это может занять время.';
        csvStatus.style.color = '#5f27cd';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload_csv', {
                method: 'POST',
                body: formData, // FormData устанавливает правильный Content-Type автоматически
            });

            if (!response.ok) {
                 // Если сервер вернул ошибку в формате JSON
                const errorData = await response.json().catch(() => null);
                const errorMessage = errorData ? errorData.error : `Ошибка сервера: ${response.status}`;
                throw new Error(errorMessage);
            }

            // Сервер возвращает файл, а не JSON
            const blob = await response.blob();

            // Создаем временную ссылку для скачивания файла
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'results.csv'; // Имя файла для скачивания
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url); // Очищаем ссылку
            a.remove();

            csvStatus.textContent = 'Обработка завершена! Файл results.csv скачан.';
            csvStatus.style.color = '#27cd88';
            csvUploadForm.reset(); // Очищаем форму

        } catch (error) {
            console.error('Ошибка при загрузке CSV:', error);
            csvStatus.textContent = `Ошибка: ${error.message}`;
            csvStatus.style.color = '#d9534f';
        }
    });


    function displayModerationResult(result) {
        if (result.error) {
            moderationResult.innerHTML = `Ошибка: ${result.error}`;
            moderationResult.style.color = '#d9534f';
            return;
        }

        const isToxic = result.label === 'LABEL_1';
        const score = (result.score * 100).toFixed(2);
        const labelText = isToxic ? 'Токсично (1)' : 'Нетоксично (0)';

        if (isToxic) {
            moderationResult.innerHTML = `
                <strong style="color: #c44569; font-size: 22px;">Обнаружен токсичный контент!</strong>
                <p style="margin-top: 10px; color: #333;">Результат: <strong>${labelText}</strong></p>
                <p style="font-size: 14px; color: #555;">Уверенность модели: ${score}%</p>
            `;
        } else {
            moderationResult.innerHTML = `
                <strong style="color: #27cd88; font-size: 22px;">Комментарий безопасен</strong>
                <p style="margin-top: 10px; color: #333;">Результат: <strong>${labelText}</strong></p>
                <p style="font-size: 14px; color: #555;">Уверенность модели: ${score}%</p>
            `;
        }
    }

    function addCommentToList(text) {
        const commentDiv = document.createElement('div');
        commentDiv.className = 'comment';
        commentDiv.textContent = text;
        commentList.prepend(commentDiv);
    }
});