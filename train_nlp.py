from gensim.models import FastText

def run():
    print("Training NLP-Based Embeddings...")

    # Sample training sentences (Replace with real descriptions)
    sentences = [
        ["Milk", "Dairy", "Fresh", "Organic"],
        ["Cheddar", "Cheese", "Dairy"],
        ["Bananas", "Fruits"]
    ]

    # Train FastText model
    fasttext_model = FastText(sentences, vector_size=64, window=3, min_count=1, sg=1)

    # Save embeddings
    fasttext_model.wv.save("models/fasttext_embeddings.bin")

    print("NLP embeddings saved!")

    return fasttext_model
