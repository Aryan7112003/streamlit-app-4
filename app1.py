import streamlit as st
import pickle
import os
import re

# ✅ Define correct model path
model_path = r"E:\NLP_Project\naive_bayes_model.pkl"

# ✅ Ensure model file exists before loading
if not os.path.exists(model_path):
    st.error(f"🚨 Model file not found at {model_path}! Please run `save_model.py` first.")
    st.stop()

# ✅ Load trained Naive Bayes model
with open(model_path, "rb") as model_file:
    abusive_word_freq, non_abusive_word_freq, abusive_total_words, non_abusive_total_words, total_abusive, total_non_abusive = pickle.load(model_file)

# ✅ Function to clean & tokenize text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9+\-.\s]', '', text)  # Keep numbers & symbols
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

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
st.set_page_config(page_title="Email Classification", page_icon="📩")
st.title("📝 Email Classification App (Naive Bayes)")
st.write("Enter an email message below and classify it as **Abusive** or **Non-Abusive**.")

# Input text box
email_text = st.text_area("📩 Enter Email Content:", "")

if st.button("🔍 Classify"):
    prediction = classify_email(email_text)

    # ✅ Apply dynamic background color based on prediction
    background_color = "#ffcccc" if prediction == "Abusive" else "#ccffcc"

    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {background_color};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.success(f"✅ Prediction: {prediction}")


