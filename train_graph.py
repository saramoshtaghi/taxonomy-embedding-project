import networkx as nx
from node2vec import Node2Vec

def run():
    print("Training Graph-Based Embeddings...")

    # Load taxonomy as a graph
    df = pd.read_csv("data/google-product-taxonomy.csv")
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["Parent"], row["Child"])

    # Train Node2Vec
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # Save embeddings
    model.wv.save_word2vec_format("models/taxonomy_embeddings.txt")

    print("Graph embeddings saved!")

    return model
