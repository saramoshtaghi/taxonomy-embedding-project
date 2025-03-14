# 🏆 Taxonomy-Based Embeddings for Hierarchical Product Recommendations

![GitHub repo 
size](https://img.shields.io/github/repo-size/saramoshtaghi/taxonomy-embeddings-project)
![GitHub 
issues](https://img.shields.io/github/issues/saramoshtaghi/taxonomy-embeddings-project)
![GitHub 
stars](https://img.shields.io/github/stars/saramoshtaghi/taxonomy-embeddings-project?style=social)
![GitHub 
forks](https://img.shields.io/github/forks/saramoshtaghi/taxonomy-embeddings-project?style=social)

## 🔥 Overview
This project builds **taxonomy-based embeddings** for structured product 
hierarchies using:
- **Graph Neural Networks (Node2Vec)** 📊
- **NLP-based embeddings (FastText, BERT)** 📖
- **Hybrid embeddings (Graph + NLP)** 🔀

These embeddings improve **product search, retrieval, and 
recommendations** by leveraging **hierarchical taxonomy data**.

---

## 🚀 Features
✅ **Graph-Based Learning** 📊: Uses **Node2Vec** to generate embeddings 
from hierarchical product taxonomies.  
✅ **Text-Based Learning** 📖: Trains **FastText** embeddings for semantic 
meaning.  
✅ **Hybrid Model** 🔀: Combines **graph & text embeddings** for better 
accuracy.  
✅ **Evaluation** 📉: Uses **t-SNE, cosine similarity, and Recall@K**.  

---

## 📂 Project Structure

taxonomy-embedding-project/ │── data/ # Store dataset files │── models/ # Trained embedding models │── src/ # Source code │ ├── main.py # Main script │ ├── preprocess.py # Data processing functions │ ├── train_graph.py # Train Node2Vec embeddings │ ├── train_nlp.py # Train FastText embeddings │ ├── hybrid_embeddings.py # Merge Graph & NLP embeddings │ ├── evaluate.py # Evaluation metrics │── README.md # Documentation │── requirements.txt # Dependencies │── .gitignore # Ignore unnecessary files


