import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import demoji   
demoji.download_codes()
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from nltk import FreqDist
import urllib
import re
from nltk.tokenize import sent_tokenize


def read_data(url):
  data = pd.read_csv(url)
  return data
  
def missing_values_check(data):
  num_missing = data.isnull().sum()
  return num_missing

def handling_missing_values(data): 
  data = data[data.columns[data.isnull().mean() < 0.6]]
  data = data.fillna(data.mean())
  data = data.apply(lambda x:x.fillna(x.value_counts().index[0]))
  data = data.dropna(how='any', subset=['at', 'content'])
  df = missing_values_check(data)
  return df

def check_duplicates(data):
    # if these two numbers differ, drop these rows
    data1 = data.drop_duplicates()
    # check whether any duplicates in the data irrespective of ID
    data2 = data1.drop_duplicates(subset=[col for col in data1.columns if col != ['reviewId']])
    return data2

def reformat_date(data): 
    split = pd.to_datetime(data['at'], format='%Y-%m-%d %H:%M:%S')
    data['Year'] = split.dt.strftime('%Y')
    data['Month'] = split.dt.strftime('%m')
    data['Day'] = split.dt.strftime('%d')
    data['Hour'] = split.dt.strftime('%H')
    data['Minute'] = split.dt.strftime('%M')
    return data

def normalization(data, feature, low, high):
  Max = np.max(feature) 
  Min = np.min(feature)
  norm = [round((i-Min)/(Max-Min)*(high-low)+low, 3) for i in feature] 
  feature = norm
  print(data.describe())
  return data

    
def MonthlyAvgScore(data):

    df1=pd.DataFrame(data.groupby('Month')['score'].mean().sort_values(ascending=True))
    plt.xlabel('score')
    plt.ylabel('Month')
    plt.title('score of app Reviews')
    score_graph=plt.barh(np.arange(len(df1.index)),df1['score'],color='purple',)
    # Writing score names on bar
    for bar,score in zip(score_graph, df1.index):
        plt.text(1,bar.get_y()-2.4 +bar.get_width(),'{}'.format(score),va='center',fontsize=20,color='white')
    # Writing month values on graph
    for bar,month in zip(score_graph, df1['score']):
        plt.text(bar.get_width()-0.5, bar.get_y() + 0.2* bar.get_width(),'%.3f'%month,va='center',fontsize=20,color='black')
    plt.yticks([])
    plt.show()
  
# function to plot most frequent terms
def freq_words(x, terms = 20):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()
    

#Line plot showing the count of reviews over time
def show_groupbymonth_count(data):
    
    #data.groupby('Month').score.value_counts().plot.bar()
    data.groupby('Month').score.count().plot.bar()
    
    plt.title("count of reviews by month")
    plt.xlabel("Month")
    plt.ylabel("count of reviews")

# plot the WordCloud image 
# first, pip install wordcloud
def word_cloud(df):
    
    comment_words = '' 
    comment_words += " ".join(review for review in df.content)+" "
    stopwords = set(STOPWORDS)
    
    #wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(comment_words)
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words)
    
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show() 


####################################################### Clean Reviews ###################################################
# install brew & wget: to get the engilish_contractions file (json), which will be used to normalize words with contraction

## refer to: https://stackoverflow.com/questions/33886917/how-to-install-wget-in-macos

## install in terminal:
### step 1: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
### step 2: brew install wget
### step 3: wget https://gist.githubusercontent.com/Sirsirious/c70400176a4532899a483e06d72cf99e/raw/e46fa7620c4f378f5bf39608b45cddad7ff447a4/english_contractions.json

import json
import string 
from textblob import TextBlob
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

url = 'https://raw.githubusercontent.com/rishabhverma17/sms_slang_translator/master/slang.txt'
    
file = urllib.request.urlopen(url)    
slang_dict = {}
    
for line in file:
  decoded_line = line.decode("utf-8")
  split = decoded_line.split('=')
  key = split[0]
  value = split[-1].rstrip('\n').lstrip('\t')
  slang_dict[key] = value
    
del slang_dict['QPSA?\tQue Pasa?\n']


# define a reviews cleaning pipline
def reviews_cleaning_pipline(text):
    '''
    Normalize text, including removing punctuations, extra whitespaces, normalizing contractions, correcting spelling, lematizing text, and removing stop words
    @text: a given string
    @return a string
    '''
    text = replace_emojis(text)
    text = remove_punctuations(text)
    text = removeDigits(text)
    text = remove_extra_whitespaces(text)
    text = replace_slang(text,slang_dict)
    # text = normalize_contractions(text) 
    # text = lematize_text (text)
    text = lower_text(text)
    # text = remove_stop_words(text)
    return text

    # remove punctuations

def replace_emojis(text): 
    text = demoji.replace_with_desc(text)
    return text

def remove_punctuations(text):
    '''
    Remove all punctuations in a given text and use one whitespace to replace them.
    @text: a given string
    @return a string without any punctuations
    ''' 
    text = str(text)
#     print(f'text: {text}')
    punctuations = string.punctuation.replace("'", '')
#     print(punctuations)
    for char in punctuations:
        text = text.replace(char, " ")
#     print(f'corrected text: {text}')
    return text

# remove digits 
def removeDigits(text): 
    return re.sub('\d', '', text)

# remove extra whitespaces
def remove_extra_whitespaces(text):
    '''
    Remove extra whitespaces in a given text. It should be used after the function of remove_punctuation
    @text: a given string
    @return a string
    '''
    text = str(text)
    whitespace = r'[\n,\t,\r, ]+'
#     print(f'text: {text}')
    correct_text = re.sub(whitespace, r' ', text)
#     print(f'corrected text: {correct_text}')
    return correct_text.strip(' ')

# replace slang 
def replace_slang(text,slang_dict): 
    
    text = re.split('\W+', text)
    
    for i in range(0,len(text)):
        if text[i] in slang_dict.keys(): 
            text[i] = slang_dict[text[i]]
        
    text = ' '.join(text)
    
    return text

# normalize contractions
def normalize_contractions(text):
    '''
    Change English words with contraction to normal ones
    @text: a given string
    @return a string
    '''
    text = str(text)
#     print(f'text: {text}')
    contractions_dict = json.loads(open('english_contractions.json', 'r').read())
    new_tokens = []
#     print(f'new_tokens: {new_tokens}')
    tokens = text.split()
#     print(f'tokens: {tokens}')
    for token in tokens:
        char_1_upper = False
#         print(f'token: {token}')
        if token[0].isupper():
            char_1_upper = True
#         print(f'char_1_upper: {char_1_upper}')
        if token.lower() in contractions_dict:
            replacement = contractions_dict[token.lower()]
#             print(f'replacement: {replacement}')
            if char_1_upper:
                replacement = replacement[0].upper() + replacement[1:] # if the first char of the raw token is capital, keep it in the replacement
#                 print(f'replacement_upper: {replacement}')
            replacement_tokens = replacement.split()
            for repl_token in replacement_tokens:
                new_tokens.append(repl_token)
        else:
            new_tokens.append(token)
#         print(f'new_tokens: {new_tokens}')
    correct_text = " ".join(new_tokens).strip(' ')
#     print(f'corrected text: {correct_text}')
    return correct_text


# lemmatize text via spacy package
def lematize_text(text):
    '''
    Lemmatize text
    @text: a given string
    @return a string
    '''
    nlp = spacy.load('en')
    text = nlp(text)
#     print(nlp)
    correct_text = ''
    for token in text:
#         print(token)
        correct_text += " "+token.lemma_
#         print(token.lemma_)
    return correct_text


# switched order to match our normalization pipeline in doc [EH]
def lower_text(text):
    '''
    Lower all words in a text
    @text: a given string
    @return a string
    '''
    correct_text = text.lower()
    return correct_text

def remove_stop_words(text):
    '''
    Remove stop words in a text via the package nltk
    @text: a given string
    @return a string
    '''    
    stops = stopwords.words('english')
    text = TextBlob(text)
    correct_text = ' '.join([token for token in text.words if token not in stops]).strip()
    return correct_text


# added these because we need text tokenized for word cloud
def split_text(col_name):
    text = re.split('\W+', col_name)
    return text

def tokenization(df, col_name): 
    df[col_name] = df[col_name].apply(lambda x: split_text(x))
    return df 


# the LDA visualization 

import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim

def LDA_vis(data):
    dictionary = corpora.Dictionary(data)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in data]

    LDA = gensim.models.ldamodel.LdaModel

    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=5, random_state=100,
                chunksize=1000, passes=50)

    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary, sort_topics=True)
    return vis


# split reviews into sentences and create a new dataframe

def gen_df_sent(data, reviews='content'):
    '''
    Generate a new dataframe. Each row represent a sentence from a review.
    @data: a given dataframe
    @reviews: a string, the column name which records reviews in data
    @return: a new dataframe
    '''
    data['review_sentences'] = data[reviews].apply(sent_tokenize)
    reviewId =[]
    score = []
    review_sent = []
    for i in data.index:
        for sent in data.review_sentences[i]:
            reviewId.append(data.reviewId[i])
            score.append(data.score[i])
            review_sent.append(sent)

    new = {"reviewId": reviewId, "score": score, 'review_sent': review_sent}
    df_sent = pd.DataFrame.from_dict(new)
    return df_sent