# ---------------------------------------------------
# 1️⃣ Imports
# ---------------------------------------------------
import streamlit as st
import numpy as np
import re
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions


# ---------------------------------------------------
# 2️⃣ Load Model + Tokenizer
# ---------------------------------------------------
model = joblib.load("best_model.h5")

# VERY IMPORTANT: load the same tokenizer used in training
tokenizer = joblib.load("tokenizer.pkl")


# ---------------------------------------------------
# 3️⃣ Preprocessing - SAME AS TRAINING
# ---------------------------------------------------
def preprocess_text(text):

    text = text.lower()
    text = contractions.fix(text)

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    negations = {"not", "no", "nor", "never"}
    stop_words = set(stopwords.words("english")) - negations
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)



# ---------------------------------------------------
# 4️⃣ Mapping
# ---------------------------------------------------
label_map = {
    0: "Irrelevant",
    1: "Negative",
    2: "Neutral",
    3: "Positive"
}


# ---------------------------------------------------
# 5️⃣ Streamlit UI
# ---------------------------------------------------
st.title("📊 Twitter Sentiment Analysis (Deep Learning)")
st.subheader("Analyze tweet sentiment instantly")

tweet_input = st.text_area("Write a tweet here:")

if tweet_input:

    processed = preprocess_text(tweet_input)

    seq = tokenizer.texts_to_sequences([processed])
    pad_seq = pad_sequences(seq, maxlen=200)

    pred = model.predict(pad_seq)
    pred_class = np.argmax(pred)

    sentiment = label_map[pred_class]

    st.markdown(f"### 🧠 **Predicted Sentiment: {sentiment}**")

    if sentiment == "Negative":
        st.error("⚠️ This tweet has a negative sentiment!")
    elif sentiment == "Positive":
        st.success("✅ This tweet is positive!")
    elif sentiment == "Neutral":
        st.info("ℹ️ Neutral sentiment.")
    else:
        st.warning("🔍 Irrelevant tweet.")


