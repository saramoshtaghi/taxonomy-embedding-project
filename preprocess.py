import os
import pandas as pd
import kagglehub

# Define dataset path
DATA_PATH = "data/google-product-taxonomy.csv"

def download_data():
    """Downloads the dataset if not already present."""
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Downloading from Kaggle...")
        
        # Download dataset
        path = kagglehub.dataset_download("alvations/google-product-taxonomy")

        # Find the correct file
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    os.rename(os.path.join(root, file), DATA_PATH)
                    print(f"Dataset saved to: {DATA_PATH}")
                    return DATA_PATH

        print("Dataset download failed.")
    else:
        print("Dataset already exists.")

def load_data():
    """Loads and returns the taxonomy dataset."""
    download_data()
    df = pd.read_csv(DATA_PATH)
    print("Sample Data:")
    print(df.head())
    return df

if __name__ == "__main__":
    load_data()
