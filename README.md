# Twitter-Sentiment-Analysis
A full end-to-end NLP & ML system for real-time sentiment analysis, topic discovery, and social media insights based on 31,962 tweets.  This project integrates advanced Natural Language Processing, Machine Learning, Deep Learning, and interactive deployment using Streamlit to support brand monitoring and trend detection.

📌 Project Overview

This project aims to build a scalable pipeline for processing and analyzing large volumes of social media data. It includes:

Comprehensive text preprocessing

Exploratory data analysis & visualization

Supervised & unsupervised ML models

Transformer-based deep learning models

Real-time interactive dashboard for analytics

The dataset contains 31,962 tweets, each labeled with positive, negative, or neutral sentiment.

🧹 1. Data Preprocessing Pipeline

A full cleaning & feature extraction pipeline built for noisy Twitter text:

✔ Removal of URLs, mentions, hashtags, punctuation, emojis
✔ Tokenization using NLTK TweetTokenizer
✔ POS-tagged lemmatization for context-aware normalization
✔ Hashtag & mention extraction as engineered features
✔ TF-IDF vectorization with sublinear TF scaling
✔ Ready for ML, DL, and topic modeling tasks

📊 2. Exploratory Data Analysis (EDA)

Interactive and statistical analysis for understanding behavior and trends:

Sentiment distribution over time and by entity

Word clouds, n-gram exploration per sentiment class

Retweet & mention network graphs

Temporal anomaly detection for sentiment spikes

Geographic sentiment mapping where available

🤖 3. Supervised Machine Learning

Multiple baseline and advanced models:

Logistic Regression with TF-IDF (baseline)

Random Forest with (1,2)-gram features

Linear SVM optimized for text classification

Stratified sampling + k-fold cross-validation

Class imbalance handling (SMOTE / undersampling)

🧭 4. Unsupervised Learning & Topic Modeling

Discovering hidden patterns & emerging themes:

LDA with 15–20 topics

KMeans clustering (k=10)

NMF for additive topic discovery

pyLDAvis for interactive topic visualization

Human-in-the-loop evaluation for topic quality

🤯 5. Deep Learning Models

State-of-the-art architectures for tweet sentiment:

Fine-tuned DistilBERT

LSTM with embeddings

CNN text classifier with multi-filter convolution

Attention-based models for explainable predictions

Comparison against ML baselines

🖥️ 6. Streamlit Real-Time Dashboard

A production-ready dashboard for instant analysis:

Real-time sentiment prediction from user input

Topic assignment + similar tweet recommendation

Temporal sentiment trends visualized with Plotly

Alerts for negative sentiment spikes

Exportable reports for social media teams

📦 Tech Stack

Python, NLTK, SpaCy, Scikit-Learn, PyTorch / TensorFlow, Transformers, pyLDAvis, Plotly, NetworkX, Streamlit.

🎯 Objectives

Build robust NLP pipelines

Achieve high-accuracy sentiment classification

Identify emerging trends automatically

Deploy insights in a real-time dashboard