import pickle
import pandas as pd
import re
import os

# ✅ Define correct save path
save_path = r"E:\NLP_Project\naive_bayes_model.pkl"
file_path = r"E:\NLP_Project\Emails.csv"

# ✅ Load dataset
df = pd.read_csv(file_path)[['content', 'Class']]

# ✅ Function to clean & tokenize text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9+\-.\s]', '', text)  # Keep numbers & symbols
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

# ✅ Train Naive Bayes Model
def train_naive_bayes(data):
    abusive_emails = [preprocess_text(email) for email, label in zip(data['content'], data['Class']) if label == "Abusive"]
    non_abusive_emails = [preprocess_text(email) for email, label in zip(data['content'], data['Class']) if label == "Non Abusive"]

    # Count word frequencies
    def count_word_freq(dataset):
        word_freq = {}
        total_words = 0
        for words in dataset:
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
                total_words += 1
        return word_freq, total_words

    abusive_word_freq, abusive_total_words = count_word_freq(abusive_emails)
    non_abusive_word_freq, non_abusive_total_words = count_word_freq(non_abusive_emails)

    return abusive_word_freq, non_abusive_word_freq, abusive_total_words, non_abusive_total_words, len(abusive_emails), len(non_abusive_emails)

# ✅ Train and save model in the correct directory
model_data = train_naive_bayes(df)

with open(save_path, "wb") as model_file:
    pickle.dump(model_data, model_file)

print(f"✅ Model saved successfully at {save_path}!")
