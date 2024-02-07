import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from textblob import TextBlob
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt


def clean_text(text: str):
    pre_text = text
    pre_text = re.sub(r'\n|\r', ' ', pre_text)
    pre_text = re.sub(r' +', ' ', pre_text)
    pre_text = pre_text.strip()
    return pre_text


def normalize_doc(doc: str, stop_words):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    ps = PorterStemmer()
    filtered_tokens = [ps.stem(token) for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc


def normalize_text(text: str, stop_words):
    cleaned_text = clean_text(text)
    sentences = nltk.sent_tokenize(cleaned_text)
    norm_sentences = [normalize_doc(sentence, stop_words) for sentence in sentences]
    return norm_sentences


def tf_idf_vectorization(docs: list[str]):
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(docs)
    dt_matrix = dt_matrix.toarray()
    vocab = tv.get_feature_names_out()
    return vocab, dt_matrix


def term_document_vectorization(text: str, stop_words):
    normalized_documents = normalize_text(text, stop_words)
    vocab, dt_matrix = tf_idf_vectorization(normalized_documents)
    td_matrix = dt_matrix.T
    td_df = pd.DataFrame(np.round(td_matrix, 2), index=vocab)
    return td_df


def estimate_sentiments(text: str):
    blob = TextBlob(text)
    return blob.sentiment


def evaluate_summarization(candidate_text, reference_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
    score = scorer.score(reference_text, candidate_text)
    return score


def retrieve_all_files_from_directory(directory_path):
    return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content


def plot_category_summary_results(category_name, category_results):
    summary_methods = ("LSA", "Clustering", "Statistical")
    metrics_means = {
        'ROUGE1': (category_results['lsa_rouge1'], category_results['clustering_rouge1'], category_results['statistical_rouge1']),
        'ROUGE2': (category_results['lsa_rouge2'], category_results['clustering_rouge2'], category_results['statistical_rouge2']),
        'ROUGE3': (category_results['lsa_rouge3'], category_results['clustering_rouge3'], category_results['statistical_rouge3']),
        'ROUGEL': (category_results['lsa_rougeL'], category_results['clustering_rougeL'], category_results['statistical_rougeL']),
        'SENTIMENT_MSE': (category_results['lsa_sentiment_mse'], category_results['clustering_sentiment_mse'],
                          category_results['statistical_sentiment_mse'])
    }

    x = np.arange(len(summary_methods))  # the label locations
    width = 0.18  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in metrics_means.items():
        offset = width * multiplier
        measurement = tuple(map(lambda value: round(value, 3), measurement))
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fontsize=8)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Metric Mean Value')
    ax.set_title(f'{category_name} category summarization results with different methods')
    ax.set_xticks(x + width*2, summary_methods)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)
    plt.show()
