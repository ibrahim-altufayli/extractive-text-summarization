import nltk
import pandas as pd
from lsa_summarizer import lsa_summarizer
from culstering_summarizer import clustering_document_term_summarizer
from statistical_summarizer import statistical_summarizer
from utils import estimate_sentiments, evaluate_summarization, plot_category_summary_results
from bbc_news_ds_eval import dataset_eval

# Run it once!!
# nltk.download()
stop_words = nltk.corpus.stopwords.words('english')


print("***********************DATASET EVALUATION************************")

#dataset_eval(lsa_topics_num=2, num_clusters=4, summarization_perc=0.35)

print("*****************Visualize DATASET EVALUATION******************")
results_csv = pd.read_csv('./results/bbc_news_db_eval.csv', index_col=0)

plot_category_summary_results('Business', results_csv.loc['Business'])

plot_category_summary_results('Entertainment', results_csv.loc['Entertainment'])

plot_category_summary_results('Politics', results_csv.loc['Politics'])

plot_category_summary_results('Sport', results_csv.loc['Sport'])

plot_category_summary_results('Tech', results_csv.loc['Tech'])


print("*****************Want to SUMMARIZE****************")
text = """
Ad sales boost Time Warner profit

Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (£600m) for the three months to December, from $639m year-earlier.

The firm, which is now one of the biggest investors in Google, benefited from sales of high-speed internet connections and higher advert sales. TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn. Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL.

Time Warner said on Friday that it now owns 8% of search-engine Google. But its own internet business, AOL, had has mixed fortunes. It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters. However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues. It hopes to increase subscribers by offering the online service free to TimeWarner internet customers and will try to sign up AOL's existing customers for high-speed broadband. TimeWarner also has to restate 2000 and 2003 results following a probe by the US Securities Exchange Commission (SEC), which is close to concluding.

Time Warner's fourth quarter profits were slightly better than analysts' expectations. But its film division saw profits slump 27% to $284m, helped by box-office flops Alexander and Catwoman, a sharp contrast to year-earlier, when the third and final film in the Lord of the Rings trilogy boosted results. For the full-year, TimeWarner posted a profit of $3.36bn, up 27% from its 2003 performance, while revenues grew 6.4% to $42.09bn. "Our financial performance was strong, meeting or exceeding all of our full-year objectives and greatly enhancing our flexibility," chairman and chief executive Richard Parsons said. For 2005, TimeWarner is projecting operating earnings growth of around 5%, and also expects higher revenue and wider profit margins.

TimeWarner is to restate its accounts as part of efforts to resolve an inquiry into AOL by US market regulators. It has already offered to pay $300m to settle charges, in a deal that is under review by the SEC. The company said it was unable to estimate the amount it needed to set aside for legal reserves, which it previously set at $500m. It intends to adjust the way it accounts for a deal with German music publisher Bertelsmann's purchase of a stake in AOL Europe, which it had reported as advertising revenue. It will now book the sale of its stake in AOL Europe as a loss on the value of that stake.
"""

target_summary = """
TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn.For the full-year, TimeWarner posted a profit of $3.36bn, up 27% from its 2003 performance, while revenues grew 6.4% to $42.09bn.Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (£600m) for the three months to December, from $639m year-earlier.However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues.Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL.For 2005, TimeWarner is projecting operating earnings growth of around 5%, and also expects higher revenue and wider profit margins.It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters.Time Warner's fourth quarter profits were slightly better than analysts' expectations.
"""

print("***ORIGINAL TEXT SENTIMENTS***")
print(estimate_sentiments(text))
print('***LSA SUMMARY***')
lsa_summary = lsa_summarizer(text, stop_words, num_topics=2, summarization_perc=0.35)
print(lsa_summary)
print('***SUMMARIZATION EVALUATION***')
print(evaluate_summarization(target_summary, lsa_summary))
print('***SUMMARIZATION SENTIMENTS***')
print(estimate_sentiments(lsa_summary))
print('--------------------------------------------')
print('***CLUSTERING SUMMARY***')
clustering_summary = clustering_document_term_summarizer(text, stop_words, num_clusters=4, summarization_perc=0.35)
print(clustering_summary)
print('***SUMMARIZATION EVALUATION***')
print(evaluate_summarization(target_summary, clustering_summary))
print('***SUMMARIZATION SENTIMENTS***')
print(estimate_sentiments(clustering_summary))
print('--------------------------------------------')
print('***STATISTICAL SUMMARY***')
statistical_summary = statistical_summarizer(text, stop_words, summarization_perc=0.35)
print(statistical_summary)
print('***SUMMARIZATION EVALUATION***')
print(evaluate_summarization(target_summary, statistical_summary))
print('***SUMMARIZATION SENTIMENTS***')
print(estimate_sentiments(statistical_summary))

