import streamlit as st
import pandas as pd
import re

# ✅ Load dataset from CSV
@st.cache_data  # Cache to prevent reloading every time
def load_data():
    file_path = "E:/NLP_Project/Emails.csv"  # Ensure this file is in the same directory
    df = pd.read_csv(file_path)
    return df[['content', 'Class']]  # Keep only relevant columns

df = load_data()

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

# Train the model on startup
abusive_word_freq, non_abusive_word_freq, abusive_total_words, non_abusive_total_words, total_abusive, total_non_abusive = train_naive_bayes(df)

# ✅ Naive Bayes Classification Function
def classify_email(email_text, alpha=1.0):
    words = preprocess_text(email_text)

    # Prior probabilities
    prob_abusive = total_abusive / (total_abusive + total_non_abusive)
    prob_non_abusive = total_non_abusive / (total_abusive + total_non_abusive)

    # Vocabulary size for Laplace smoothing
    vocab_size = len(set(list(abusive_word_freq.keys()) + list(non_abusive_word_freq.keys())))

    # Calculate likelihood for each class
    abusive_likelihood = prob_abusive
    non_abusive_likelihood = prob_non_abusive

    for word in words:
        abusive_likelihood *= (abusive_word_freq.get(word, 0) + alpha) / (abusive_total_words + alpha * vocab_size)
        non_abusive_likelihood *= (non_abusive_word_freq.get(word, 0) + alpha) / (non_abusive_total_words + alpha * vocab_size)

    return "Abusive" if abusive_likelihood > non_abusive_likelihood else "Non Abusive"

# ✅ Streamlit UI
st.title("📝 Email Classification App (Naive Bayes)")
st.write("Enter an email message below and classify it as **Abusive** or **Non-Abusive**.")

# Input text box
email_text = st.text_area("📩 Enter Email Content:", "")

if st.button("🔍 Classify"):
    prediction = classify_email(email_text)
    st.success(f"✅ Prediction: {prediction}")
