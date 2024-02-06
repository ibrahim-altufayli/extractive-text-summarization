from sklearn.cluster import KMeans
import nltk
from scipy.spatial import distance
from utils import clean_text, normalize_text, tf_idf_vectorization


def clustering_document_term_summarizer(text: str, stop_words, num_clusters=2, summarization_perc=0.35):
    sentences = nltk.sent_tokenize(clean_text(text))
    num_sentences = int(len(sentences) * summarization_perc)
    normalized_text = normalize_text(text, stop_words)
    terms, document_vectors = tf_idf_vectorization(normalized_text)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    y_pred = kmeans.fit_predict(document_vectors)
    # decide which cluster holds more sentences
    clustered_documents_ids = {}
    for i in range(num_clusters):
        for j in range(document_vectors.shape[0]):
            if y_pred[j] == i:
                distance_from_centroid = distance.euclidean(kmeans.cluster_centers_[i], document_vectors[j])
                clustered_documents_ids[i] = clustered_documents_ids.get(i, []) + [(j, distance_from_centroid)]

    utmost_cluster_id = max(clustered_documents_ids, key=lambda key: len(clustered_documents_ids.get(key, [])))
    summary_sentences_ids = map(lambda item: item[0], sorted(clustered_documents_ids[utmost_cluster_id],
                                                    key=lambda item: item[1], reverse=True)[:min(len(sentences), num_sentences)])
    summary_sentences_ids = sorted(summary_sentences_ids)
    return '\n'.join(map(lambda sent_id: sentences[sent_id], summary_sentences_ids))
