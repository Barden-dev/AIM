import torch

def classify_text_batch(texts_with_ids: list, model, tokenizer, max_length=128):
    if not texts_with_ids:
        return []

    results = []

    ids = [item['id'] for item in texts_with_ids]
    texts_to_classify = [item['text'] for item in texts_with_ids]

    inputs = tokenizer(
        texts_to_classify,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_ids = logits.argmax(dim=-1)

    for i, text_id in enumerate(ids):
        predicted_class_id = predicted_class_ids[i].item()

        label = predicted_class_id

        score = probabilities[i][predicted_class_id].item()

        results.append({
            "id": text_id,
            "label": label,
            "score": score
        })

    return results