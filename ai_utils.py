import numpy as np
from transformers import pipeline

def calculate_one_shot_embeddings(st,texts, labels, batch_size=8):
    progress_bar = st.progress(0)
    # Load the zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification")

    num_texts = len(texts)
    num_labels = len(labels)
    num_batches = int(np.ceil(num_texts / batch_size))
    zero_shot_scores = np.zeros((num_texts, num_labels))

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_texts)
        batch_texts = texts[start_idx:end_idx]

        # Perform the zero-shot classification using the batch of texts
        batch_scores = classifier(batch_texts, labels)

        for j, scores in enumerate(batch_scores):
            label_scores = scores['scores']
            zero_shot_scores[start_idx + j, :] = label_scores
        progress_bar.progress((i + 1) / num_batches)

    return zero_shot_scores
