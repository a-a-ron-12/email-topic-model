<<<<<<< HEAD
# Import required packages
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from random import randrange, randint

# Initialize parameters
base_path = '/workspaces/email topic model/email_topic_model'
test_corpus_path = f'{base_path}/data/test_corpus.pkl'
test_pdf_path = f'{base_path}/data/test_pdf.pkl'
tf_vectorizer_model_path = f'{base_path}/models/tf_vectorizer_model.pkl'
tfidf_vectorizer_model_path = f'{base_path}/models/tfidf_vectorizer_model.pkl'
topic_dict_path = f'{base_path}/data/topic_dict.pkl'
lda_model_path = f'{base_path}/models/lda_model.pkl'

#st.session_state.inbox_button = False


# Main function that drives the webpage
def main():

    topic_1_button = False
    topic_2_button = False
    topic_3_button = False
    
    # Load data and NLP models
    test_corpus = pickle.load(open(test_corpus_path, "rb"))
    test_pdf = pickle.load(open(test_pdf_path, "rb"))
    tf_vectorizer_model = pickle.load(open(tf_vectorizer_model_path, "rb"))
    tfidf_vectorizer_model = pickle.load(open(tfidf_vectorizer_model_path, "rb"))
    topic_dict = pickle.load(open(topic_dict_path, "rb"))
    lda_model = pickle.load(open(lda_model_path, "rb"))

    # Setup email message dataframe
    stage_pdf = test_pdf[['sender', 'clean_body', 'date']]
    stage_pdf['sender'] = stage_pdf['sender'].str.replace(r"\<.*\>", "")
    stage_pdf['date'] = pd.to_datetime(stage_pdf['date']).dt.date
    stage_pdf.columns = [" ", "  ", "   "]
    stage_pdf = stage_pdf.sample(frac=1).reset_index(drop = True)

    # Set page config to wide for content to spread out
    st.set_page_config(layout = "wide")

    # Initialize all session state button variables
    if 'inbox_button' not in st.session_state:
        st.session_state.inbox_button = False

    if 'topic_1_button' not in st.session_state:
        st.session_state.topic_1_button = False

    if 'topic_2_button' not in st.session_state:
        st.session_state.topic_2_button = False

    if 'topic_3_button' not in st.session_state:
        st.session_state.topic_3_button = False



    # Top header for primary button
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write(" ")
    with col2:
        st.write(" ")
    with col3:
        st.write(" ")
    with col4:
        inbox_button = st.button(label = "Organize Inbox")

    # Top email count indicator
    if inbox_button or st.session_state.inbox_button:
        st.session_state.inbox_button = True
        st.write(f"You have 0 messages in your inbox")
    else:
        st.write(f"You have {stage_pdf.shape[0]} messages in your inbox")

    # If button is pressed, perform inference on dataset from LDA model
    if st.session_state.inbox_button:

        # Transform and predict on test corpus
        test_matrix = tf_vectorizer_model.transform(test_corpus)
        test_matrix = tfidf_vectorizer_model.transform(test_corpus)
        predictions = lda_model.transform(test_matrix)

        # Pick highest probability topic and map to text topic
        stage_pdf['prediction'] = np.argmax(predictions, axis = 1)
        stage_pdf = stage_pdf.replace({'prediction': topic_dict})

        # Sum up the counts for each topic
        topic_1 = topic_dict[0]
        topic_2 = topic_dict[1]
        topic_3 = topic_dict[2]
        topic_1_cnt = sum(stage_pdf['prediction'] == topic_1)
        topic_2_cnt = sum(stage_pdf['prediction'] == topic_2)
        topic_3_cnt = sum(stage_pdf['prediction'] == topic_3)

        # Setup columns on webpage
        col1, col2, col3 = st.columns(3)

        with col1:
            topic_1_button = st.button(label = f"{topic_1}: {topic_1_cnt}")
        with col2:
            topic_2_button = st.button(label = f"{topic_2}: {topic_2_cnt}")
        with col3:
            topic_3_button = st.button(label = f"{topic_3}: {topic_3_cnt}")

    if topic_1_button:
        st.session_state.topic_1_button = True
    elif topic_2_button:
        st.session_state.topic_2_button = True
    elif topic_3_button:
        st.session_state.topic_3_button = True

    # Header picture image
    with st.container():
        st.image(Image.open('images/header.JPG'))

    # Sidebar picture image
    st.sidebar.image(Image.open('images/sidebar.JPG'))

    # Condition to display test dataframe or not based on button executed
    if st.session_state.inbox_button:

        if st.session_state.topic_1_button:
            stage_pdf = stage_pdf[stage_pdf['prediction'] == topic_1].drop('prediction', axis = 1).reset_index(drop = True)
            st.dataframe(stage_pdf)
            st.session_state.inbox_button = False
            st.session_state.topic_1_button = False

        elif st.session_state.topic_2_button:
            stage_pdf = stage_pdf[stage_pdf['prediction'] == topic_2].drop('prediction', axis = 1).reset_index(drop = True)
            st.dataframe(stage_pdf)
            st.session_state.inbox_button = False
            st.session_state.topic_2_button = False

        elif st.session_state.topic_3_button:
            stage_pdf = stage_pdf[stage_pdf['prediction'] == topic_3].drop('prediction', axis = 1).reset_index(drop = True)
            st.dataframe(stage_pdf)
            st.session_state.inbox_button = False
            st.session_state.topic_3_button = False

        elif st.session_state.inbox_button:
            st.text(f"Your inbox mail has been organized into {len(topic_dict)} buckets for you based on email content similarity.")

    else:
        st.dataframe(stage_pdf)
        st.session_state.inbox_button = False


if __name__ == '__main__':

=======
# Import required packages
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from random import randrange, randint

# Initialize parameters
#base_path = '/workspaces/email topic model/email_topic_model'
test_corpus_path = 'data/test_corpus.pkl'
test_pdf_path = 'data/test_pdf.pkl'
tf_vectorizer_model_path = 'models/tf_vectorizer_model.pkl'
tfidf_vectorizer_model_path = 'models/tfidf_vectorizer_model.pkl'
topic_dict_path = 'data/topic_dict.pkl'
lda_model_path = 'models/lda_model.pkl'

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

>>>>>>> 11d905ad626cd2c5b5a4517424a25f8e6b91cb6e
    main()