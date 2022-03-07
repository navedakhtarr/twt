# importing relevant python packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
# preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
# modeling
from sklearn import svm
# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# creating page sections
site_header = st.container()
business_context = st.container()
data_desc = st.container()
performance = st.container()
tweet_input = st.container()
model_results = st.container()
sentiment_analysis = st.container()
contact = st.container()

with site_header:
    st.title('Profanity Detection')






with tweet_input:
    st.header('Is Your Message Considered Hate Speech?')

    user_text = st.text_input('Enter Message', max_chars=280) # setting input as user_text

with model_results:
    st.subheader('Prediction:')
    if user_text:
    # processing user_text
        # removing punctuation
        user_text = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
        # tokenizing
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(user_text)
        # removing stop words
        stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
        # taking root word
        lemmatizer = WordNetLemmatizer()
        lemmatized_output = []
        for word in stopwords_removed:
            lemmatized_output.append(lemmatizer.lemmatize(word))

        # instantiating count vectorizor
        count = CountVectorizer(stop_words=stop_words)
        X_train = pickle.load(open('pickle/X_train_2.pkl', 'rb'))
        X_test = lemmatized_output
        X_train_count = count.fit_transform(X_train)
        X_test_count = count.transform(X_test)

        # loading in model
        final_model = pickle.load(open('pickle/final_log_reg_count_model.pkl', 'rb'))

        # apply model to make predictions
        prediction = final_model.predict(X_test_count[0])

        if prediction == 0:
            st.subheader('**Not Hate Speech**')
        else:
            st.subheader('**Hate Speech**')
        st.text('')

with sentiment_analysis:
    if user_text:
        st.header('Sentiment Analysis')

        # explaining VADER

        st.text('')

        # instantiating VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text)
        if sentiment_dict['compound'] >= 0.05 :
            category = ("**Positive ✅**")
        elif sentiment_dict['compound'] <= - 0.05 :
            category = ("**Negative 🚫**")
        else :
            category = ("**Neutral ☑️**")

        # score breakdown section with columns
        breakdown, graph = st.beta_columns(2)
        with breakdown:
            # printing category
            st.write("Your message is rated as", category)
            # printing overall compound score
            st.write("**Compound Score**: ", sentiment_dict['compound'])
            # printing overall compound score
            st.write("**Polarity Breakdown:**")
            st.write(sentiment_dict['neg']*100, "% Negative")
            st.write(sentiment_dict['neu']*100, "% Neutral")
            st.write(sentiment_dict['pos']*100, "% Positive")
        with graph:
            sentiment_graph = pd.DataFrame.from_dict(sentiment_dict, orient='index').drop(['compound'])
            st.bar_chart(sentiment_graph)
