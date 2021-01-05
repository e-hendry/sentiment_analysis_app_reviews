import nltk
from nltk.util import ngrams

#https://www.pythonprogramming.in/generate-the-n-grams-for-the-given-sentence-using-nltk-or-textblob.html
def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [' '.join(grams) for grams in n_grams]

### count the term frequency
import pandas as pd
def getTermFrequency(ngram, split_reviews):
    
    dict_list = []
    for review in split_reviews:
        wordDict = dict.fromkeys(ngram, 0)
        for word in review:
            wordDict[word]+=1
        dict_list.append(dict(wordDict))
    return dict_list

## create the final dataframe with score and ID
def combineDataFrame(ngram, split_reviews, review_ID, score):
    dict_list = getTermFrequency(ngram, split_reviews)
    df = pd.DataFrame(dict_list, index = review_ID)
    df['Score'] = score
    return df

########################## tf_idf_weight_norm #######################
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

def tf_idf_weight_norm (data, norm = "l2"):
    '''
    Calculate normalized tf-idf weight.
    @data: a dataframe, representing term frequency
    @norm: a string ('l2', 'l1', 'max'), representing normalization method
    @return: a dataframe
    '''
    
    data_array = data.to_numpy()

    if norm == 'l2': # Euclidean normalization
        tf_idf = TfidfTransformer(norm = 'l2')
        tf_idf_weight_norm = tf_idf.fit_transform(data_array).todense()
    elif norm == 'l1': # Manhattan normalization
        tf_idf = TfidfTransformer(norm = 'l1')
        tf_idf_weight_norm = tf_idf.fit_transform(data_array).todense()
    elif norm == 'max': # Maximum normalization
        idf = TfidfTransformer().fit(data_array).idf_
        idf_matrix = np.diag(idf)
        tf_idf_weight = pd.DataFrame(data_array @ idf_matrix, columns = data.columns)
        tf_idf_weight_norm = tf_idf_weight.apply(lambda x: x/max(abs(x)), axis = 1)
    else:
      print("Error: normaliztion method should be one of 'l1', 'l2' or 'max'!")
      return 
       
    tf_idf_weight_norm = pd.DataFrame(tf_idf_weight_norm, columns = data.columns)

    return tf_idf_weight_norm

###################################### reviews embedding ############################
from tqdm import tqdm
import pandas as pd
import numpy as np

def load_word_embeds(fname_word_embeds):
    '''
    load pre-trained dictionary from local download
    @fname_word_embeds: a string, the file name of the pre-trained dictionary from local download
    @return: a dictionary, a vector space to represent words
    '''
    print()
    print('#'*30)
    print("Loading pretrained word embedding dictionary:")
    pre_trained_dict = open(fname_word_embeds, 'r', encoding='utf-8', newline='\n', errors='ignore')
    embeds = {}
    for line in tqdm(pre_trained_dict):
        tokens = line.rstrip().split(' ')
        embeds[tokens[0]] = [float(x) for x in tokens[1:]]
    print("Got pretrained word embedding dictionary.")    
    return embeds

def reviews_embedding_and_save(reviews, embedding_dic, fname_save_reviews_embedding=None):
    '''
    Generate a vector space to represent all reviews.
    @reviews: 1-d array of strings;
    @embeeding_dic: dictionary, a pre-trained word embedding dictionary.
    @fname_save_reviews_embedding: a string, path to save embed_features (csv file)
    @return: a dataframe.
    '''
    print("\nStarting reviews embedding:")
    embed_features = pd.DataFrame()
    for review in tqdm(reviews):
        words = review.split()
        feature_vec = np.zeros((1,300))
        for word in words:
            try:
                feature_vec += np.array(embedding_dic[word])
            except:
                pass # skip the word that is not included in the dictionary
        feature_vec = feature_vec/len(words)
        # append the feature vector of each review to the feature table
        embed_features = embed_features.append(pd.DataFrame(feature_vec), ignore_index=True)
    # end for
    print(f'Shape of embed_features: {embed_features.shape}')
    
    # save word embedding features to .csv files
    if fname_save_reviews_embedding:
        embed_features.to_csv(fname_save_reviews_embedding, index=False)
        print('\nReviews embedding saved in a .csv file.')
        print('#'*30)
    else:
        print('\nReviews embedding not saved.')
        print('#'*30)

    return embed_features
