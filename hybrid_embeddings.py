import numpy as np
from gensim.models import KeyedVectors

def run():
    print("Merging Graph & NLP-Based Embeddings...")

    # Load Graph and NLP embeddings
    graph_model = KeyedVectors.load_word2vec_format("models/taxonomy_embeddings.txt")
    text_model = KeyedVectors.load("models/fasttext_embeddings.bin")

    merged_embeddings = {}

    # Merge the embeddings
    alpha = 0.7  # Weighting factor
    for word in graph_model.key_to_index:
        if word in text_model.key_to_index:
            merged_embeddings[word] = alpha * np.array(graph_model[word]) + (1 - alpha) * np.array(text_model[word])
        else:
            merged_embeddings[word] = graph_model[word]

    print("Hybrid embeddings created!")

    return merged_embeddings
