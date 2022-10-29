# Import required packages
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from random import randrange, randint

# Initialize parameters
base_path = '/workspaces/email topic model/email_topic_model'
test_corpus_path = '/data/test_corpus.pkl'
test_pdf_path = '/data/test_pdf.pkl'
tf_vectorizer_model_path = '/models/tf_vectorizer_model.pkl'
tfidf_vectorizer_model_path = '/models/tfidf_vectorizer_model.pkl'
topic_dict_path = '/data/topic_dict.pkl'
lda_model_path = '/models/lda_model.pkl'

# Main function that drives the webpage
def main():

    # Load data and NLP models
    test_corpus = pickle.load(open(test_corpus_path, "rb"))
    test_pdf = pickle.load(open(test_pdf_path, "rb"))
    tf_vectorizer_model = pickle.load(open(tf_vectorizer_model_path, "rb"))
    tfidf_vectorizer_model = pickle.load(open(tfidf_vectorizer_model_path, "rb"))
    topic_dict = pickle.load(open(topic_dict_path, "rb"))
    lda_model = pickle.load(open(lda_model_path, "rb"))

    # Set page config to wide for content to spread out
    st.set_page_config(layout = "wide")

    # Top header for primary button
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write(" ")
    with col2:
        st.write(" ")
    with col3:
        st.write(" ")
    with col4:
        exec_button = st.button(label = "Organize Inbox")

    # Top email count indicator
    if exec_button:
        st.write(f"You have 0 messages in your inbox")
    else:
        st.write(f"You have {test_pdf.shape[0]} messages in your inbox")

    # If button is pressed, randomly compute stats (demo purposes)
    if exec_button:

        topic_1_cnt = randint(100, 300)
        topic_2_cnt = randint(100, 300)
        topic_3_cnt = test_pdf.shape[0] - topic_1_cnt - topic_2_cnt

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"#### {topic_dict[0]}: {topic_1_cnt}")
        with col2:
            st.markdown(f"#### subscriptions: {topic_2_cnt}")
        with col3:
            st.markdown(f"#### {topic_dict[2]}: {topic_3_cnt}")

    # Header picture image
    with st.container():
        st.image(Image.open('images/header.JPG'))

    # Sidebar picture image
    st.sidebar.image(Image.open('images/sidebar.JPG'))

    # Condition to display test dataframe or not based on button executed
    if exec_button:
        st.text(f"Your inbox mail has been organized into {len(topic_dict)} buckets for you based on email content similarity.")
    else:
        stage_pdf = test_pdf[['sender', 'clean_body', 'date']]
        stage_pdf['sender'] = stage_pdf['sender'].str.replace(r"\<.*\>", "")
        stage_pdf['date'] = pd.to_datetime(stage_pdf['date']).dt.date
        stage_pdf.columns = [" ", "  ", "   "]

        stage_pdf = stage_pdf.sample(frac=1).head(100).reset_index(drop = True)

        st.dataframe(stage_pdf)


if __name__ == '__main__':

    main()