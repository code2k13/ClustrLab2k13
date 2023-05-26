
from operator import is_
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import base64

from utils import get_sentence_embeddings, extract_keywords2, get_embeddings_from_file, get_sentences_from_file, get_sentence_chunks, get_cluster_descriptions


def main():
    # Set Streamlit app title
    is_csv  = False
    st.set_page_config(page_title="Sentence Clustering", layout="wide")
    batch_size = st.sidebar.slider(
        "Batch Size", min_value=1, max_value=512, step=7, value=64)
    threshold = st.sidebar.slider(
        "Similarity Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.1)
    n_visual_clusters = st.sidebar.slider(
        "Visual Clusters", min_value=1, max_value=64, step=1, value=8)

    # File uploader
    file = st.sidebar.file_uploader("Upload a text file", type=["txt", "csv"])

    if file is not None:
        sentences = get_sentences_from_file(st, file)
        if sentences == None or len(sentences) < 10:
            st.error("Need atleast more than 10 data items, aborting !")
            return
        embeddings = get_embeddings_from_file(st, sentences, batch_size)
        print(len(sentences),embeddings.shape)
        if file.name.split(".")[-1].lower() == "csv":
            is_csv = True
            chunks = [(embeddings[idx], [s]) for idx, s in enumerate(sentences)]
            chunk_embeddings = embeddings
        else:
            chunks = get_sentence_chunks(embeddings, sentences, threshold)
            chunk_embeddings = np.array(
                [np.mean(chunk, axis=0) for chunk, _ in chunks])

        # Perform TSNE on chunk embeddings
        # chunk_embeddings = chunk_embeddings.reshape((-1,512))
        print(chunk_embeddings.shape)
        perplexity = min(20, chunk_embeddings.shape[0]-1)
        n_visual_clusters = min(n_visual_clusters, chunk_embeddings.shape[0]-1)
        kmeans = KMeans(n_clusters=n_visual_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(chunk_embeddings)
        cluster_descriptions = get_cluster_descriptions(
            n_visual_clusters, cluster_labels, chunks)
        print(cluster_descriptions)

        chunk_tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(
            chunk_embeddings)

       
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

        fig = px.scatter(df_chunks, x="x", y="y", color="Cluster", hover_data={
                         "Chunk": True, "TooltipText": True}, title="Sentence Chunks")
        st.plotly_chart(fig, use_container_width=True)


# Run the Streamlit app
if __name__ == "__main__":
    main()
