import os
import nltk
import numpy as np
import pandas as pd
from scipy.spatial import distance
from lsa_summarizer import lsa_summarizer
from culstering_summarizer import clustering_document_term_summarizer
from statistical_summarizer import statistical_summarizer
from utils import estimate_sentiments, evaluate_summarization, retrieve_all_files_from_directory, read_text_file

RESULTS_FILE_PATH = "./results/bbc_news_db_eval.csv"
DATA_SET_PATH = './datasets/bbc_news_ds'
NEWS_PATH = DATA_SET_PATH + '/news'
SUMMARIES_PATH = DATA_SET_PATH + '/summaries'

NEWS_CATEGORIES = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']
NEWS_CATEGORIES_PATHS = ['/business', '/entertainment', '/politics', '/sport', '/tech']


def dataset_eval(lsa_topics_num=2, num_clusters=2, summarization_perc=0.35):
    stop_words = nltk.corpus.stopwords.words('english')
    results_df = pd.DataFrame(index=NEWS_CATEGORIES, columns=["lsa_rouge1", "lsa_rouge2", "lsa_rouge3", "lsa_rougeL", "lsa_sentiment_mse",
                                                              "clustering_rouge1", "clustering_rouge2", "clustering_rouge3", "clustering_rougeL", "clustering_sentiment_mse",
                                                              "statistical_rouge1", "statistical_rouge2", "statistical_rouge3", "statistical_rougeL", "statistical_sentiment_mse"])

    for category_idx, category_path in enumerate(NEWS_CATEGORIES_PATHS):
        print(f"Processing category {NEWS_CATEGORIES[category_idx]}")
        lsa_rouge_1 = []
        lsa_rouge_2 = []
        lsa_rouge_3 = []
        lsa_rouge_l = []
        lsa_sentiments = []

        clust_rouge_1 = []
        clust_rouge_2 = []
        clust_rouge_3 = []
        clust_rouge_l = []
        clust_sentiments = []

        stas_rouge_1 = []
        stas_rouge_2 = []
        stas_rouge_3 = []
        stas_rouge_l = []
        stas_sentiments = []

        news_files_names = retrieve_all_files_from_directory(NEWS_PATH + category_path)
        target_summaries_files_names = retrieve_all_files_from_directory(SUMMARIES_PATH + category_path)
        assert len(news_files_names) == len(target_summaries_files_names)
        for news_file_name, target_summary_file_name in zip(news_files_names, target_summaries_files_names):
            print(f"Processing file: {news_file_name}")
            news_file_path = os.path.join(NEWS_PATH + category_path, news_file_name)
            target_summary_file_path = os.path.join(SUMMARIES_PATH + category_path, target_summary_file_name)

            text = read_text_file(news_file_path)
            target_summary = read_text_file(target_summary_file_path)

            text_sentiment = estimate_sentiments(text)
            text_sentiment_vec = [text_sentiment.polarity, text_sentiment.subjectivity]

            lsa_summary = lsa_summarizer(text, stop_words, num_topics=lsa_topics_num, summarization_perc=summarization_perc)

            summary_sentiment = estimate_sentiments(lsa_summary)
            summary_sentiment_vec = [summary_sentiment.polarity, summary_sentiment.subjectivity]
            lsa_sentiments.append(np.square(np.subtract(text_sentiment_vec, summary_sentiment_vec)))

            score = evaluate_summarization(lsa_summary, target_summary)
            lsa_rouge_1.append(score['rouge1'].fmeasure)
            lsa_rouge_2.append(score['rouge2'].fmeasure)
            lsa_rouge_3.append(score['rouge3'].fmeasure)
            lsa_rouge_l.append(score['rougeL'].fmeasure)

            clustering_summary = clustering_document_term_summarizer(text, stop_words, num_clusters=num_clusters,
                                                                     summarization_perc=summarization_perc)
            summary_sentiment = estimate_sentiments(clustering_summary)
            summary_sentiment_vec = [summary_sentiment.polarity, summary_sentiment.subjectivity]
            clust_sentiments.append(np.square(np.subtract(text_sentiment_vec, summary_sentiment_vec)))

            score = evaluate_summarization(clustering_summary, target_summary)
            clust_rouge_1.append(score['rouge1'].fmeasure)
            clust_rouge_2.append(score['rouge2'].fmeasure)
            clust_rouge_3.append(score['rouge3'].fmeasure)
            clust_rouge_l.append(score['rougeL'].fmeasure)

            statistical_summary = statistical_summarizer(text, stop_words, summarization_perc=summarization_perc)
            summary_sentiment = estimate_sentiments(statistical_summary)
            summary_sentiment_vec = [summary_sentiment.polarity, summary_sentiment.subjectivity]
            stas_sentiments.append(np.square(np.subtract(text_sentiment_vec, summary_sentiment_vec)))

            score = evaluate_summarization(statistical_summary, target_summary)
            stas_rouge_1.append(score['rouge1'].fmeasure)
            stas_rouge_2.append(score['rouge2'].fmeasure)
            stas_rouge_3.append(score['rouge3'].fmeasure)
            stas_rouge_l.append(score['rougeL'].fmeasure)

        results_df.at[NEWS_CATEGORIES[category_idx], "lsa_rouge1"] = np.average(np.array(lsa_rouge_1))
        results_df.at[NEWS_CATEGORIES[category_idx], "lsa_rouge2"] = np.average(np.array(lsa_rouge_2))
        results_df.at[NEWS_CATEGORIES[category_idx], "lsa_rouge3"] = np.average(np.array(lsa_rouge_3))
        results_df.at[NEWS_CATEGORIES[category_idx], "lsa_rougeL"] = np.average(np.array(lsa_rouge_l))
        results_df.at[NEWS_CATEGORIES[category_idx], "lsa_sentiment_mse"] = np.sqrt(np.average(np.array(lsa_sentiments)))

        results_df.at[NEWS_CATEGORIES[category_idx], "clustering_rouge1"] = np.average(np.array(clust_rouge_1))
        results_df.at[NEWS_CATEGORIES[category_idx], "clustering_rouge2"] = np.average(np.array(clust_rouge_2))
        results_df.at[NEWS_CATEGORIES[category_idx], "clustering_rouge3"] = np.average(np.array(clust_rouge_3))
        results_df.at[NEWS_CATEGORIES[category_idx], "clustering_rougeL"] = np.average(np.array(clust_rouge_l))
        results_df.at[NEWS_CATEGORIES[category_idx], "clustering_sentiment_mse"] = np.sqrt(np.average(np.array(clust_sentiments)))

        results_df.at[NEWS_CATEGORIES[category_idx], "statistical_rouge1"] = np.average(np.array(stas_rouge_1))
        results_df.at[NEWS_CATEGORIES[category_idx], "statistical_rouge2"] = np.average(np.array(stas_rouge_2))
        results_df.at[NEWS_CATEGORIES[category_idx], "statistical_rouge3"] = np.average(np.array(stas_rouge_3))
        results_df.at[NEWS_CATEGORIES[category_idx], "statistical_rougeL"] = np.average(np.array(stas_rouge_l))
        results_df.at[NEWS_CATEGORIES[category_idx], "statistical_sentiment_mse"] = np.sqrt(np.average(np.array(stas_sentiments)))

    results_df.to_csv(RESULTS_FILE_PATH, sep=',', index=True, encoding='utf-8')


