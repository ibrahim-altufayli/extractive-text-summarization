from scipy.sparse.linalg import svds
import nltk
import numpy as np
from utils import term_document_vectorization, clean_text


def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt


def lsa_summarizer(text, stop_words, num_topics=2, summarization_perc=0.35):
    sentences = nltk.sent_tokenize(clean_text(text))
    num_sentences = int(len(sentences)*summarization_perc)
    td_df = term_document_vectorization(text, stop_words)
    td_matrix = td_df.to_numpy()
    u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)
    term_topic_mat, singular_values, topic_document_mat = u, s, vt
    # remove singular values below threshold
    sv_threshold = 0.5
    min_sigma_value = max(singular_values) * sv_threshold
    singular_values[singular_values < min_sigma_value] = 0
    salience_scores = np.sqrt(np.dot(np.square(singular_values),
                                     np.square(topic_document_mat)))

    top_sentence_indices = (-salience_scores).argsort()[:min(len(sentences), num_sentences)]
    top_sentence_indices.sort()
    return '\n'.join(np.array(sentences)[top_sentence_indices])
