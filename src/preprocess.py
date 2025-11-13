import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import TextVectorization

# -----------------------------
# Step 1: Load data
# -----------------------------
def load_data(path):
    print("Step 1: Loading IMDb dataset")
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape}")
    print(df.head())
    return df

# -----------------------------
# Step 2: Clean text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Step 3-5: Preprocess data
# -----------------------------
def preprocess_data(df, max_words=10000, seq_lengths=[25, 50, 100]):
    print()
    print("Step 2: Lowercasing all text and removing punctuation/special characters")
    df["review"] = df["review"].apply(clean_text)
    print("Text cleaning completed")

    print()
    print("Step 3: Converting text to sequences using TextVectorization")
    vectorizer = TextVectorization(
        max_tokens=max_words,
        output_sequence_length=max(seq_lengths),  
        standardize=None 
    )
    vectorizer.adapt(df["review"].values)
    sequences = vectorizer(df["review"].values).numpy()
    labels = np.array(df["sentiment"].apply(lambda x: 1 if x == "positive" else 0))
    print("Sequence conversion complete")

    results = {}
    for length in seq_lengths:
        print()
        print(f"Step 4: Adjusting sequences to fixed length = {length}")
        padded = sequences[:, :length]
        if padded.shape[1] < length:
            padded = np.pad(padded, ((0,0),(0,length-padded.shape[1])), mode='constant')

        X_train, X_test, y_train, y_test = train_test_split(
            padded, labels, test_size=0.5, random_state=42
        )
        results[length] = (X_train, X_test, y_train, y_test)
        print(f"Sequence length {length}: Training set shape {X_train.shape}, Test set shape {X_test.shape}")

    print()
    print("Preprocessing completed successfully")
    return vectorizer, results

# -----------------------------
# Step 6: Save processed data
# -----------------------------
def save_processed(results, save_dir="../data"):
    print()
    print("Step 5: Saving processed data files")
    os.makedirs(save_dir, exist_ok=True)

    for length, (X_train, X_test, y_train, y_test) in results.items():
        np.savez_compressed(
            os.path.join(save_dir, f"imdb_seq{length}.npz"),
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        print(f"Saved imdb_seq{length}.npz")

    print("All processed datasets saved successfully")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    data_path = r"C:\Users\teppo\Desktop\nlp_hw3\data\IMDB Dataset.csv"
    df = load_data(data_path)
    vectorizer, results = preprocess_data(df)
    save_processed(results)
    print()
    print("All preprocessing steps completed")

