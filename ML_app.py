import pickle 
import numpy as np 
import streamlit as st
import time
import re

# Load model + vectorizer
with open(r"C:\Users\Top10\Downloads\rfc_model.pkl", "rb") as f:
    model=pickle.load(f)
    
with open(r"C:\Users\Top10\Desktop\Twitter-Sentiment-Analysis-1\vectorizor.pkl","rb") as f:
    vectorizor = pickle.load(f)
    

# Page Title 
st.markdown(
    """
    <h1 style="text-align:center; 
               animation: fadein 6s ease-in-out;">
        Twitter Sentiment Analyzer
    </h1>

    <style>
        @keyframes fadein {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        #typing {
            font-size: 18px;
            color: #888;
            height: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Input
tweet = st.text_area("", placeholder="Enter your tweet...")
def preprocess(tweet):
    tweet = tweet.lower() 
    tweet = re.sub(r"http\S+", "", tweet) 
    tweet = re.sub(r"@\w+", "", tweet)     
    tweet = re.sub(r"#", "", tweet)        
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)  
    tweet = tweet.strip()
    return tweet

# Predict
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet first.")
    else:
        with st.spinner("Analyzing tweet... hang tight "):
            time.sleep(1.5)  
            clean_tweet = preprocess(tweet)
            transformed = vectorizor.transform([clean_tweet])
            st.session_state.prediction = model.predict(transformed)[0]
if st.session_state.prediction is not None:
    pred = st.session_state.prediction
    if pred == 1:
        st.error("Negative Tweet!")
    elif pred == 3:
        st.success("Positive Tweet!")
    elif pred == 2:
        st.info("Neutral")
    elif pred == 0:
        st.info("Irrelevant")


