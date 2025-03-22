import pandas as pd
import os
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
from tqdm import tqdm  

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load Hugging Face emotion detection model on GPU if available
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def preprocess_text(text):
    """Clean and preprocess text."""
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words

    # Handle missing stopwords safely
    try:
        stop_words = set(stopwords.words('english')) - {"not", "no", "nor"}  # Keep negations
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english')) - {"not", "no", "nor"}

    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def analyze_text(text):
    """Analyze emotion from preprocessed text."""
    preprocessed_text = preprocess_text(text)
    return emotion_classifier(preprocessed_text)

def process_csv(file_path):
    """Read CSV file and analyze text along with its label."""
    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("The CSV file must contain both 'text' and 'label' columns.")
    
    results = []
    total_rows = len(df)
    print(f"Total Rows in Dataset: {total_rows}\n")

    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing Rows", unit="row"):
        text = row["text"]
        label = row["label"]
        
        if pd.notna(text) and pd.notna(label):  # Ensure text and label are not NaN
            predicted_emotion = analyze_text(text)
            results.append({"text": text, "label": label, "predicted_emotion": predicted_emotion})
    
    return results, total_rows

if __name__ == "__main__":
    csv_file = "./Dataset/emotions.csv"  # Change to your CSV file path
    results, processed_rows = process_csv(csv_file)
    
    for entry in results:
        print(f"Text: {entry['text']}\nActual Label: {entry['label']}\nPredicted Emotion: {entry['predicted_emotion']}\n{'-'*50}")

    print(f"\n✅ Processing complete. Total rows processed: {processed_rows}")