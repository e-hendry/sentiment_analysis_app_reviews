# refer to: https://github.com/JoMingyu/google-play-scraper

#####################################
#  pip install google-play-scraper  #
#####################################

from google_play_scraper import app

# summary about walmart app
walmart_url = 'com.walmart.android' # walmart app downloading url, complete version: https://play.google.com/store/apps/details?id=com.walmart.android

app_summary = app(
    walmart_url,
    lang='en', # defaults to 'en'
    country='us' # defaults to 'us'
)
print('Got app summary...')

# scrawl most recent 10,000 reviews on walmart app
from google_play_scraper import Sort, reviews

result_most_10K, continuation_token = reviews(
    walmart_url,
    lang='en', # defaults to 'en'
    country='us', # defaults to 'us'
    sort=Sort.NEWEST, # scrawl most recent reviews
    count=10000, # srawl 10,000 reviews
    filter_score_with= None # defaults to None(means all score)
)
print('Got most recent 10K reviews...')

# scrawl another 10,000 reviews after the most recent 10,000 reviews 
result_less_10k, _ = reviews(
    walmart_url,
    sort=Sort.NEWEST,
    continuation_token=continuation_token # it makes the scrapper to crawl reviews after 10,000 reviews, defaults to None(load from the beginning)
)
print('Got 2nd 10k reviews...')

# save data into csv files
import pandas as pd

reviews_1st_df = pd.DataFrame(result_most_10K)
reviews_1st_df.to_csv('../data/reviews_1st_10K.csv', index=None, header=True)

reviews_2nd_df = pd.DataFrame(result_less_10k)
reviews_2nd_df.to_csv('../data/reviews_2nd_10K.csv', index=None, header=True)
print('Reviews saved in csv files (folder: proj/data)!')