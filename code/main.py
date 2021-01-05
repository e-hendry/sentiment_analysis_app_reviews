

################### Read Files ####################
import pandas as pd
from tqdm import tqdm
raw_data = pd.read_csv('reviews_1st_10K.csv')


################### Data Exploration ####################
# check missing values
from data_exploration import missing_values_check
missing_info = missing_values_check(raw_data)
# handle missing values
from data_exploration import handling_missing_values
data = handling_missing_values(raw_data)
# reformat date
from data_exploration import reformat_date
data = reformat_date(raw_data)
# visualize monthly average score
from data_exploration import MonthlyAvgScore
data = MonthlyAvgScore(raw_data)
# normalize data
from data_exploration import normalization
data = normalization(raw_data, raw_data['thumbsUpCount'], 0, 1)

# clean reviews 
from data_exploration import reviews_cleaning_pipline
reviews = raw_data.content
norm_reviews = []
for review in tqdm(reviews):
    norm_reviews.append(reviews_cleaning_pipline(review))
raw_data['norm_reviews'] = pd.Series(norm_reviews)


####################### Feature engineering #######################
# tf-idf
## n-grams 
from feature_engineering import extract_ngrams
data['ngram_1'] = data['clean_reviews'].apply(lambda x: extract_ngrams(x,1))
data['ngram_2'] = data['clean_reviews'].apply(lambda x: extract_ngrams(x,2))
## tf-idf matrix
from feature_engineering import tf_idf_weight_norm
products = {'Product': [356,561,0,321],
            'Price': [250,0,1200,300]}
term_freq_test = pd.DataFrame(products, columns= ['Product', 'Price'])
tf_idf_weight_norm = tf_idf_weight_norm(term_freq_test, norm = 'max') # term_freq is dataframe

# word embedding
## load the pre-trained word embedding dictionary
from reviews_embedding import load_word_embeds
fname_word_embeds = 'wiki-news-300d-1M.vec' # first download this file from https://fasttext.cc/docs/en/english-vectors.html
word_embeds = load_word_embeds(fname_word_embeds)
## map reviews via word embedding
from reviews_embedding import reviews_embedding_and_save
X_train_fname = '../data/X_train.csv' # file to save train features
X_test_fname = '../data/X_test.csv' # file to save test features
reviews_train = pd.read_csv('../data/reviews_train.csv')
reviews_train = reviews_train.clean_reviews.to_numpy()
reviews_test = pd.read_csv('../data/reviews_test.csv')
reviews_test = reviews_test.clean_reviews.to_numpy()

reviews_embedding_and_save(reviews_train, word_embeds, X_train_fname)
reviews_embedding_and_save(reviews_test, word_embeds, X_test_fname)

# split reviews into sentences
from data_exploration import gen_df_sent
df_sent = gen_df_sent(data)


########################################### Modeling and evaluation #################################################################

'''document-based sentiment analysis: features generated from tf-idf'''
from modeling_and_evaluation import loadData, splitDataset, sampleDataset, applyTfidf_extract, apply_model, tune_model, kFoldCrossVal, apply_SGD, tune_SGD
## loading the cleaned data
data_cleaned = loadData('cleaned_data.csv')
## splitting data into test and train 
X_train, X_test, y_train, y_test = splitDataset(data = data_cleaned, review_id = 'reviewId', cleaned_reviews = 'clean_reviews', score_col = 'review_category', train_size_ = 0.7, test_size_=0.3)
## checking the balance of the data 
y_train.review_category.value_counts()
## downsampling the training data
X_train_s, y_train_s = sampleDataset(X_train, y_train, 'review_category', 0, 1, 'downsample', 'reviewId', 'clean_reviews')
## generating test and train tf-idf
train_tfidf, test_tfidf = applyTfidf_extract(train_df = X_train_s, test_df = X_test, raw_data_col_ = 'clean_reviews', method = 'l2', review_df_train = y_train_s[['reviewId']], review_df_test = y_test[['reviewId']], max_features_ = None, max_df_=1.0, min_df_=1, ngram = (1,1))
## checking the shape of the data
print('shape of training data:', train_tfidf.shape)
print('shape of test data:', test_tfidf.shape )
## baseline decision tree model 
dct = apply_model(model_type='dt',tfidf_train = train_tfidf, y_train_ = y_train_s, 
                  tfidf_test = test_tfidf, y_test_ = y_test, method = 'gini')
## baseline random forest 
rf = apply_model(model_type='rf',tfidf_train = train_tfidf, y_train_ = y_train_s, 
                 tfidf_test = test_tfidf, y_test_ = y_test, method = 'gini')
## baseline multinomial naive bayes
mnb = apply_model(model_type='mnb',tfidf_train = train_tfidf, y_train_ = y_train_s, 
                  tfidf_test = test_tfidf, y_test_ = y_test, method = 'gini')
## baseline SGDclassification
sgd = apply_SGD(train_tfidf, y_train_s, test_tfidf, y_test)

## tuning the decision tree model
dt_best_params = tune_model(model_type='dt',tfidf_train = train_tfidf, y_train_ = y_train_s, tfidf_test = test_tfidf, y_test_ = y_test)
## tuning the random forest model
rf_best_params = tune_model(model_type='rf',tfidf_train = train_tfidf, y_train_ = y_train_s, tfidf_test = test_tfidf, y_test_ = y_test)
## tuning the naive bayes model  
mnb_best_params = tune_model(model_type='mnb',tfidf_train = train_tfidf, y_train_ = y_train_s, tfidf_test = test_tfidf, y_test_ = y_test)
## tuning SGDclassification
tune_SGD(train_tfidf, y_train_s, test_tfidf, y_test)

## K-fold cross validation on decision tree 
kFoldCrossVal(5, dct, x_train=X_train, y_train=y_train)
## K-fold cross validation on random forest 
kFoldCrossVal(5, rf, x_train=X_train, y_train=y_train)
## K-fold cross validation on multinomial naive bayes
kFoldCrossVal(5, mnb, x_train=X_train, y_train=y_train)
## K-fold cross validation
kFoldCrossVal(5, sgd, x_train=X_train, y_train=y_train)

'''document-based sentiment analysis: features generated from word embedding'''
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")
## get baseline model
from modeling_and_evaluation import compare_clfs_perf
random_state = 0
cv = StratifiedKFold(n_splits=10, shuffle=True)
clfs = [DecisionTreeClassifier(min_samples_split=10, random_state=random_state),
       KNeighborsClassifier(),
       GaussianNB(),
       LogisticRegression(random_state = random_state,),
       SVC(random_state=random_state),
       RandomForestClassifier(random_state=random_state),
       GradientBoostingClassifier(random_state=random_state)]
compare_clfs_perf(clfs, X_train, y_train, X_test, y_test, cv, scoring='accuracy', n_jobs=-1)

## tune hyper-parameters 
from modeling_and_evaluation import tune_params
cv = StratifiedKFold(n_splits=10,shuffle=True)
random_state = 0
knn = KNeighborsClassifier(n_jobs=-1)
knn_p_grid = {'n_neighbors': range(1,15,1)}
knn_best = tune_params(clf=knn, p_grid=knn_p_grid, cv=cv, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
lr = LogisticRegression(random_state=random_state, max_iter=500, n_jobs=-1)
lr_p_grid = dict(C=[1,2,4,10,20,40,80,160])
lr_best = tune_params(clf=lr, p_grid=lr_p_grid, cv=cv, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
svc = SVC(random_state=random_state)
svc_p_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
svc_best = tune_params(clf=svc, p_grid=svc_p_grid, cv=cv, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
## cross validate the tuned models
clfs = [knn_best, svc_best, lr_best]
compare_clfs_perf(clfs, X_train, y_train, X_test, y_test, cv, scoring='accuracy', n_jobs=-1)
## create final ensemble model
from modeling_and_evaluation import ensemble_best_clfs_and_predict
clfs_best = [('knn', knn_best), ('svc', svc_best), ('lr', lr_best)]
ensemble_best_clfs_and_predict(clfs_best, X_train, y_train, X_test, y_test)
## validate the generalization of the final ensemble model via new testing set
X_test_final = pd.read_csv('../data/X_test_final.csv').to_numpy()
y_test_final = pd.read_csv('../data/y_test_final.csv')['label'].to_numpy()
ensemble_best_clfs_and_predict(clfs_best, X_train, y_train, X_test_final, y_test_final)


'''aspect-based sentiment analysis: aspect extraction (topic modeling)'''
from modeling_and_evaluation import feature, topicModel, LDA_vis
feature(data, data['clean_sents'])
data = data.dropna()
topicModel(data, 5, 100, 'NMF')
## visualize LDA
LDA_vis(data)