from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import csv
import io
import plotly.graph_objects as go
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
from transformers import pipeline
from sklearn import preprocessing
from sklearn.cluster import KMeans


def extract_keywords2(text, num_keywords):
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        keyword_scores = dict(zip(feature_names, tfidf_scores))
        sorted_keywords = sorted(keyword_scores.items(),
                                 key=lambda x: x[1], reverse=True)
        keywords = sorted_keywords[:num_keywords]
        return '/'.join([k[0] for k in keywords])
    except:
        return "undefined"


def get_sentence_embeddings(model, sentences):
    embeddings = model(sentences)
    return embeddings


def get_sentences_from_file(st, file):
    file_extension = file.name.split(".")[-1]
    if file_extension.lower() == "csv":
        csv_reader = csv.reader(io.TextIOWrapper(file, encoding='utf-8'))
        header = next(csv_reader)  # Read the header row
        if len(header) != 1:
            st.error("CSV file should have only one column.")
            return []
        sentences = [row[0] for row in csv_reader if len(row) == 1]
        return sentences
    else:
        text = file.read().decode("utf-8")
        sentences = sent_tokenize(text)
        return sentences


def get_embeddings_from_sentences(st, model, sentences, batch_size):
    progress_bar = st.progress(0)
    embeddings = []

    if len(sentences) < batch_size:
        batch_size = len(sentences)
    num_batches = int(np.ceil(len(sentences) / batch_size))

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(sentences))
        batch_sentences = sentences[start_index:end_index]
        batch_embeddings = get_sentence_embeddings(model, batch_sentences)
        embeddings.append(batch_embeddings)
        progress_bar.progress((i + 1) / num_batches)
    print(len(sentences), num_batches, batch_size,
          len(embeddings), len(embeddings[0]))
    print("~"*30)
    embeddings = np.concatenate(embeddings)
    progress_bar.empty()
    return embeddings


def get_sentence_chunks(embeddings, sentences, threshold):
    # Chunking sentences based on similarity of embeddings
    chunks = []
    current_chunk = [embeddings[0]]
    current_chunk_text = [sentences[0]]
    for i in range(1, len(embeddings)):
        similarity = cosine_similarity(
            [embeddings[i]], [current_chunk[-1]])
        if similarity < threshold:
            chunks.append((current_chunk, current_chunk_text))
            current_chunk = [embeddings[i]]
            current_chunk_text = [sentences[i]]
        else:
            current_chunk.append(embeddings[i])
            current_chunk_text.append(sentences[i])
    chunks.append((current_chunk, current_chunk_text))
    return chunks


def get_cluster_descriptions(n_visual_clusters, cluster_labels, chunks):
    cluster_descriptions = []
    for cluster in range(0, n_visual_clusters):
        cluster_indices = [idx for idx, x in enumerate(
            cluster_labels) if x == cluster]
        samples_to_choose = 5
        if len(cluster_indices) < samples_to_choose:
            samples_to_choose = len(cluster_indices)
        cluster_indices = random.sample(cluster_indices, samples_to_choose)
        text = ''
        for idx in cluster_indices:
            text = text + " ".join(chunks[idx][1])
            keywords = extract_keywords2(text, 4)
            if keywords == "undefined":
                keywords = "undefined_" + str(cluster)
        cluster_descriptions.append(keywords)
    return cluster_descriptions


def plot_data(st, df_chunks):
    # Create an empty figure
    fig = go.Figure()

    # Add separate traces for each unique ClusterLabel
    for label in df_chunks["ClusterLabel"].unique():
        df_label = df_chunks[df_chunks["ClusterLabel"] == label]
        fig.add_trace(
            go.Scatter(
                x=df_label["x"],
                y=df_label["y"],
                mode='markers',
                name=label,
                hovertext=df_label["TooltipText"],  
                hoverinfo='text'   
        ))

    # Customize the layout if needed
    fig.update_layout(
        title="Cluster Plot",
        xaxis_title="X",
        yaxis_title="Y",
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig, use_container_width=True)


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

def get_visual_clusters(chunk_embeddings,n_visual_clusters):
    clusterer = KMeans(n_clusters=n_visual_clusters)
    cluster_labels = clusterer.fit_predict(
    preprocessing.normalize(chunk_embeddings))
    return cluster_labels