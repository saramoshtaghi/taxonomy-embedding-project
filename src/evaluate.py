import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def run():
    print("Evaluating Embeddings...")

    # Load Hybrid Embeddings
    hybrid_embeddings = np.load("models/hybrid_embeddings.npy", allow_pickle=True).item()
    
    # Convert to array
    words = list(hybrid_embeddings.keys())
    vectors = np.array([hybrid_embeddings[word] for word in words])

    # t-SNE Visualization
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced = tsne.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0], reduced[:,1], color="blue")

    for i, word in enumerate(words):
        plt.annotate(word, (reduced[i,0], reduced[i,1]))

    plt.title("t-SNE Visualization of Taxonomy-Based Embeddings")
    plt.show()

    print("Evaluation Completed!")
