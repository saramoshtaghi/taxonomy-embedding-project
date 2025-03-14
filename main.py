import preprocess
import train_graph
import train_nlp
import hybrid_embeddings
import evaluate

def main():
    print("Starting Taxonomy Embedding Project...")

    # Step 1: Preprocess Data
    preprocess.run()

    # Step 2: Train Graph-Based Embeddings
    train_graph.run()

    # Step 3: Train NLP-Based Embeddings
    train_nlp.run()

    # Step 4: Merge Embeddings
    hybrid_embeddings.run()

    # Step 5: Evaluate Results
    evaluate.run()

    print("Project Completed Successfully!")

if __name__ == "__main__":
    main()
