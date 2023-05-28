
from operator import is_
import streamlit as st
import numpy as np
import pandas as pd

from openTSNE import TSNE
from sklearn.cluster import KMeans
import base64
from sklearn import preprocessing
import ai_utils
import tensorflow_hub as hub
import tensorflow_text  # dont remove this !


from utils import get_embeddings_from_sentences, get_sentences_from_file, get_sentence_chunks, get_cluster_descriptions, plot_data


if 'sentence_encoder' not in st.session_state:
    st.session_state['sentence_encoder'] = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    st.session_state["tsner"] = TSNE()


def main():
    # Set Streamlit app title

    is_csv = False

    st.set_page_config(page_title="Sentence Clustering", layout="wide")
    use_zero_shot_embeddings = st.sidebar.checkbox(
        'Use Zero Shot Embeddings', help="Uses zero shot embedding pipeline to get embeddings for sentences. Chunking is not possible. Will be considerably slow without a GPU.")
    if use_zero_shot_embeddings:
        zero_shot_labels = st.sidebar.text_area(
            "Comma Separated Labels", value='', height=None, max_chars=None, key=None, help="Enter comma separated labels : positive,negative,neutral")
    if use_zero_shot_embeddings:
        batch_size = st.sidebar.slider(
            "Batch Size", min_value=1, max_value=64, step=1, value=4,
            help="Batch size for calculating embeddings. Optimum value will depend on RAM or GPU(when using zero shot embeddings)")
    else:
        batch_size = st.sidebar.slider(
            "Batch Size", min_value=1, max_value=2048, step=8, value=64, help="Batch size for calculating embeddings. Optimum value will depend on RAM or GPU(when using zero shot embeddings)")

    if not use_zero_shot_embeddings:
        threshold = st.sidebar.slider(
            "Similarity Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.1,
            help="The program tries to club similar sentences together in a single chunk and averages out embeddings. A low similarity threshold will cause larger chunks to be created.")
    n_visual_clusters = st.sidebar.slider(
        "Visual Clusters", min_value=1, max_value=64, step=1, value=8,
        help="Number of visual clusters to represent in output of TSNE. Colors each cluster with different color in TSNE output.")

    # File uploader
    file = None
    file = st.sidebar.file_uploader("Upload a text file", type=["txt", "csv"],
                                    help="Supports plain text and CSV. CSV should have only one column.")
    status_message = st.sidebar.empty()
    with st.spinner("Processing File ..."):
        button_run = st.sidebar.button('Run')
        if button_run:
            if file is None:
                return

            if use_zero_shot_embeddings == True and len(zero_shot_labels) == 0:
                return

            status_message.text("📖 Reading file ...")
            sentences = get_sentences_from_file(st, file)

            if sentences == None or len(sentences) < 10:
                st.error("Need atleast more than 10 data items, aborting !")
                return

            if use_zero_shot_embeddings == False:
                status_message.text("⚙️ Generating encodings ...")
                embeddings = get_embeddings_from_sentences(
                    st, st.session_state["sentence_encoder"], sentences, batch_size)
                print(len(sentences), embeddings.shape)
                if file.name.split(".")[-1].lower() == "csv":
                    is_csv = True
                    chunks = [(embeddings[idx], [s])
                              for idx, s in enumerate(sentences)]
                    chunk_embeddings = embeddings
                else:
                    chunks = get_sentence_chunks(
                        embeddings, sentences, threshold)
                    chunk_embeddings = np.array(
                        [np.mean(chunk, axis=0) for chunk, _ in chunks])

            else:
                chunk_embeddings = ai_utils.calculate_one_shot_embeddings(
                    st, sentences, zero_shot_labels)
                chunks = [([chunk_embeddings[idx]], [s])
                          for idx, s in enumerate(sentences)]
                print(chunk_embeddings.shape)

            print(chunk_embeddings.shape)
            n_visual_clusters = min(
                n_visual_clusters, chunk_embeddings.shape[0]-1)

            status_message.text("🧮 Clustering data ...")
            clusterer = KMeans(n_clusters=n_visual_clusters)

            cluster_labels = clusterer.fit_predict(
                preprocessing.normalize(chunk_embeddings))
            status_message.text("🖹 Generating cluster definitions ...")
            cluster_descriptions = get_cluster_descriptions(
                n_visual_clusters, cluster_labels, chunks)
            print(cluster_descriptions)
            status_message.text("🍵 TSNEing (takes time) ...")
            chunk_tsne = st.session_state["tsner"].fit(chunk_embeddings)

            status_message.text("⌛Almost there ...")
            df_chunks = pd.DataFrame({"Chunk": range(len(
                chunks)),
                "x": chunk_tsne[:, 0],
                "y": chunk_tsne[:, 1],
                "TooltipText": [text[0][:1024] for _, text in chunks],
                "Cluster": cluster_labels,
                "FullText": ['.'.join(text) for _, text in chunks],
                "ClusterLabel": [cluster_descriptions[i] for i in cluster_labels]
            })

            csv = df_chunks.to_csv(index=False)
            # Create download link
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="raw_data.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            status_message.text("📈 Plotting data ...")
            plot_data(st, df_chunks)
            status_message.text("🎉 Done !")
            status_message.text("")

            st.spinner()
            file = None


# Run the Streamlit app
if __name__ == "__main__":
    main()
