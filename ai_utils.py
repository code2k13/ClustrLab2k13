import numpy as np
from transformers import pipeline


def order_scores(scores, score_labels, ordered_labels):
    output = []
    for lbl in ordered_labels:
        output.append(scores[score_labels.index(lbl)])
    return output


def calculate_one_shot_embeddings(st, texts, labels, batch_size=4):
    progress_bar = st.progress(0)
    # Load the zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification")

    num_texts = len(texts)
    num_labels = len(labels)
    num_batches = int(np.ceil(num_texts / batch_size))
    zero_shot_scores = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_texts)
        batch_texts = texts[start_idx:end_idx]

        # Perform the zero-shot classification using the batch of texts
        batch_scores = classifier(batch_texts, labels)
        print(batch_scores)

        batch_scores_list = [order_scores(
            scores['scores'], scores["labels"], labels.split(",")) for scores in batch_scores]
        zero_shot_scores.extend(batch_scores_list)

        progress_bar.progress((i + 1) / num_batches)

    zero_shot_scores = np.array(zero_shot_scores)
    return zero_shot_scores
