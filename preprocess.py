import pandas as pd

def run():
    print("Preprocessing dataset...")

    # Load the taxonomy dataset
    df = pd.read_csv("data/google-product-taxonomy.csv")
    
    # Print sample data
    print(df.head())

    return df
