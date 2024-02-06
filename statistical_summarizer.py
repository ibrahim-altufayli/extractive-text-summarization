import nltk
import numpy as np
from utils import clean_text, normalize_text, tf_idf_vectorization


def count_pos_of_interest(text):
    part_of_speach_set = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG'
        , 'VBN', 'VBP', 'VBZ'}
    text_pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
    count = np.sum(np.array([1 if pos_tag[1] in part_of_speach_set else 0 for pos_tag in text_pos_tags]))
    return count


def statistical_summarizer(text, stop_words, summarization_perc=0.35):
    cleaned_text = clean_text(text)
    text_pos_count = count_pos_of_interest(cleaned_text)
    sentences = nltk.sent_tokenize(cleaned_text)
    num_sentences = int(len(sentences) * summarization_perc)
    normalized_text = normalize_text(text, stop_words)
    terms, document_vectors = tf_idf_vectorization(normalized_text)
    sentences_importance = []
    for doc_idx, doc_vec in enumerate(document_vectors):
        tf_idf_weight = np.sum(doc_vec)
        sentence_pos_count = count_pos_of_interest(sentences[doc_idx])
        sentence_weight = tf_idf_weight * (sentence_pos_count/text_pos_count)
        sentences_importance.append(sentence_weight)
    summary_sentences_ids = (-1 * np.array(sentences_importance)).argsort()[:min(len(sentences), num_sentences)]
    return '\n'.join(map(lambda sentence_idx: sentences[sentence_idx], summary_sentences_ids))
