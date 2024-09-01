import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
from nltk.probability import FreqDist
from nltk.tokenize import *
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from imblearn.over_sampling import RandomOverSampler
import re
from bs4 import BeautifulSoup
from nltk.util import ngrams
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
import spacy
from langdetect import detect
from nltk.corpus import words
from spellchecker import SpellChecker
import nltk
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
import json


def get_latest_file_df():

    # Pattern to match the file names with date at the end
    pattern = r'raw_scraped_(\d{8})_filtered'
    
    # Initialize variables to store the latest file and date
    latest_file = None
    latest_date = None
    
    # Iterate over files in the directory
    for filename in os.listdir('data'):
        match = re.search(pattern, filename)
        if match:
            # Extract the date string from the filename
            date_str = match.group(1)
            # Convert the date string to a datetime object
            file_date = datetime.strptime(date_str, '%Y%m%d')
            
            # Check if this is the latest date found
            if latest_date is None or file_date > latest_date:
                latest_date = file_date
                latest_file = filename
    
    # If a latest file is found, read it into a DataFrame
    if latest_file:
        file_path = os.path.join('data', latest_file)
        df = pd.read_csv(file_path)  # Change to pd.read_excel(file_path) if the files are Excel files
        return df[['review','airline_name','NPS_score','overall_rating']]
    else:
        return None


def data_cleaning():
    filtered_data = get_latest_file_df()
    filtered_data.dropna(inplace=True)


    # Create a copy of the DataFrame
    data_copy = filtered_data.copy()

    # Calculate the IQR for the text length
    Q1 = data_copy['review'].apply(len).quantile(0.25)
    Q3 = data_copy['review'].apply(len).quantile(0.75)
    IQR = Q3 - Q1

    # Set a threshold for IQR to identify outliers
    IQR_threshold = 1.5
    lower_bound = Q1 - IQR_threshold * IQR
    upper_bound = Q3 + IQR_threshold * IQR

    # Create a new column 'removed_outliers' based on IQR
    data_copy['removed_outliers'] = np.where((data_copy['review'].apply(len) >= lower_bound) & (data_copy['review'].apply(len) <= upper_bound), data_copy['review'], np.nan)

    # Drop rows with NaN in the 'removed_outliers' column
    data_copy = data_copy.dropna(subset=['removed_outliers'])
    data_copy = data_copy.reset_index(drop=True)


    # Define a function to remove everything except lowercase letters, numbers, and spaces using regex
    def remove_non_alphanumeric(text):
        return re.sub(r'[^a-z0-9 ]+', '', text)

    # Remove non-alphanumeric characters from the 'removed_outliers' column
    data_copy['removed_punctuation'] = data_copy['removed_outliers'].apply(remove_non_alphanumeric)


    data_copy['lowercased'] = data_copy['removed_punctuation'].str.lower()



    def remove_stopwords(text):
        stop_words = set(stopwords.words('english')) - {'not', 'no'}
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)

    # Remove stopwords from the 'lowercased' column
    data_copy['remove_stopwords'] = data_copy['lowercased'].apply(remove_stopwords)



    def remove_numbers(text):
        return re.sub(r'\d+', '', text)

    # Remove numbers from the 'filtered_lowercased' column
    data_copy['removed_numbers'] = data_copy['remove_stopwords'].apply(remove_numbers)



    def remove_non_english(text):
        english_words = set(nltk.corpus.words.words())
        tokens = word_tokenize(text.lower())  # Convert to lowercase before tokenization
        filtered_tokens = [word for word in tokens if word in english_words]
        return ' '.join(filtered_tokens)

    # Remove non-English words from the 'removed_numbers' column
    data_copy['english_only'] = data_copy['removed_numbers'].apply(remove_non_english)


    def removed_mix(text):
        cleaned_text = re.sub(r'\b\w*[0-9]+\w*[a-zA-Z]+\w*\b', '', text)
        return cleaned_text

    data_copy['removed_mix'] = data_copy['english_only'].apply(remove_non_english)


    data_copy['bigram'] = data_copy['removed_mix'].apply(lambda x: list(ngrams(word_tokenize(x),2)))

    all_ngrams = [item for sublist in data_copy['bigram'] for item in sublist]

    ngram_counts = Counter(all_ngrams)

    ngram_list = list(ngram_counts.items())

    sorted_ngram_list = sorted(ngram_list, key=lambda x: x[1], reverse=True)

    sorted_ngram_list = sorted_ngram_list[0:100]

    top_50_bigrams = [bigram[0] for bigram in sorted_ngram_list]

    compound_words = top_50_bigrams
    mwe_tokenizer = MWETokenizer(compound_words, separator='_')
    data_copy['tokenized_chunking'] = data_copy['removed_mix'].apply(lambda x: mwe_tokenizer.tokenize(x.split()))


    def lemmatize_tokens(tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    # Lemmatize the 'tokenized_column' column
    data_copy['lemmatized_column'] = data_copy['tokenized_chunking'].apply(lemmatize_tokens)

    pre_data = data_copy[['overall_rating','NPS_score','review','remove_stopwords','removed_numbers','english_only','removed_mix','tokenized_chunking','lemmatized_column']]

    return pre_data


def topic_modelling_function():
    df = data_cleaning()  # Your function to clean the data and return a DataFrame
    print('This is cleaned')
    print(df)
    # Create DataFrames for Promoters, Neutrals, and Detractors
    promoters_df = df[df['NPS_score'] == 'Promoter']
    neutrals_df = df[df['NPS_score'] == 'Neutral']
    detractors_df = df[df['NPS_score'] == 'Detractor']

    # List of DataFrames and their corresponding labels
    list_df = [promoters_df, neutrals_df, detractors_df]
    list_labels = ['Promoters', 'Neutrals', 'Detractors']
    
    # Dictionary to store the results for each segment
    results = {}

    # Loop through each DataFrame and apply topic modelling
    for df_segment, label in zip(list_df, list_labels):
        text_data = df_segment['lemmatized_column'].apply(lambda x: ' '.join(x))  # Join the list into a single string per document

        # Split the text into training and test sets
        text_data_train, text_data_test = train_test_split(text_data, test_size=0.2, random_state=42)

        # Vectorization: Create TF-IDF and Count matrices
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix_train = tfidf.fit_transform(text_data_train)
        tfidf_matrix_test = tfidf.transform(text_data_test)
        corpus_tfidf = Sparse2Corpus(tfidf_matrix_train, documents_columns=False)
        dictionary_tfidf = Dictionary([text.split() for text in text_data_train])

        count = CountVectorizer(max_features=1000)
        count_matrix_train = count.fit_transform(text_data_train)
        count_matrix_test = count.transform(text_data_test)
        corpus_count = Sparse2Corpus(count_matrix_train, documents_columns=False)
        dictionary_count = Dictionary([text.split() for text in text_data_train])
        
        # Initialize variables for best model tracking
        best_num_topics = 0
        best_coherence_score = 0
        best_model = None
        best_model_type = None
        
        for num_topics in range(3, 6):
            # 1. LDA + Count
            lda_count_model = LdaModel(corpus=corpus_count, num_topics=num_topics, id2word=dictionary_count, passes=15)
            coherence_model_lda_count = CoherenceModel(model=lda_count_model, texts=text_data_train.apply(lambda x: x.split()), dictionary=dictionary_count, coherence='c_v')
            coherence_score_lda_count = coherence_model_lda_count.get_coherence()

            # 2. LSI + TF-IDF
            lsi_tfidf_model = LsiModel(corpus=corpus_tfidf, num_topics=num_topics, id2word=dictionary_tfidf)
            coherence_model_lsi_tfidf = CoherenceModel(model=lsi_tfidf_model, texts=text_data_train.apply(lambda x: x.split()), dictionary=dictionary_tfidf, coherence='c_v')
            coherence_score_lsi_tfidf = coherence_model_lsi_tfidf.get_coherence()

            # 3. LDA + TF-IDF
            lda_tfidf_model = LdaModel(corpus=corpus_tfidf, num_topics=num_topics, id2word=dictionary_tfidf, passes=15)
            coherence_model_lda_tfidf = CoherenceModel(model=lda_tfidf_model, texts=text_data_train.apply(lambda x: x.split()), dictionary=dictionary_tfidf, coherence='c_v')
            coherence_score_lda_tfidf = coherence_model_lda_tfidf.get_coherence()

            # 4. LSI + Count
            lsi_count_model = LsiModel(corpus=corpus_count, num_topics=num_topics, id2word=dictionary_count)
            coherence_model_lsi_count = CoherenceModel(model=lsi_count_model, texts=text_data_train.apply(lambda x: x.split()), dictionary=dictionary_count, coherence='c_v')
            coherence_score_lsi_count = coherence_model_lsi_count.get_coherence()

            # Select the best model based on coherence score
            if coherence_score_lda_count > best_coherence_score:
                best_coherence_score = coherence_score_lda_count
                best_num_topics = num_topics
                best_model = lda_count_model
                best_model_type = 'LDA + Count'
                
            if coherence_score_lsi_tfidf > best_coherence_score:
                best_coherence_score = coherence_score_lsi_tfidf
                best_num_topics = num_topics
                best_model = lsi_tfidf_model
                best_model_type = 'LSI + TF-IDF'

            if coherence_score_lda_tfidf > best_coherence_score:
                best_coherence_score = coherence_score_lda_tfidf
                best_num_topics = num_topics
                best_model = lda_tfidf_model
                best_model_type = 'LDA + TF-IDF'
                
            if coherence_score_lsi_count > best_coherence_score:
                best_coherence_score = coherence_score_lsi_count
                best_num_topics = num_topics
                best_model = lsi_count_model
                best_model_type = 'LSI + Count'
        
        # Store the best model and results for the current segment
        best_model_output = {
            'best_model_type': best_model_type,
            'best_num_topics': best_num_topics,
            'best_coherence_score': best_coherence_score,
            'topics': best_model.print_topics(num_topics=best_num_topics)
        }

        # Save the output to a dictionary
        results[label] = best_model_output

    # Save the results to a file (JSON format)
    with open('best_topic_models.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

    # Optionally, print the results for each NPS segment
    for label, result in results.items():
        print(f"{label} - Best Model Type: {result['best_model_type']}, Number of Topics: {result['best_num_topics']}, Coherence Score: {result['best_coherence_score']}")
        pprint(result['topics'])


