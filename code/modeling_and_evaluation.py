

################################## Document-based sentiment analysis: word embedding ##############################
# import libraries
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import VotingClassifier

########## cross valudation
def compare_clfs_perf(clfs, X_train, y_train, X_test, y_test, cv, scoring='accuracy', n_jobs=-1):
    '''
    train, cross validate & test models
    @clf: a given classifier
    @X_train: 2-d array, train data of descriptive features
    @y_train: 1-d array, train data of target feature
    @X_test: 2-d array, test data of descriptive features
    @y_test: 1-d array, test data of target feature
    @cv: cross validation method
    @scoring: metric to measure model perfomance
    @n_jobs: int
    @return a dataframe, the result of model perfomance comparison.
    '''
    #create table to compare model performance (metric: accuracy)
    clfs_columns = ['clf name', 'clf params', f'clf validation {scoring} std (CrossVal)', f'clf train {scoring} mean (CrossVal)', f'clf validation {scoring} mean (CrossVal)', 'accuracy (testing dataset)','clf time']
    clfs_compare = pd.DataFrame(columns = clfs_columns)

    for i in tqdm(range(len(clfs))):  

        # train & cross validate (training dataset)
        clf_name = clfs[i].__class__.__name__
        clfs_compare.loc[i, 'clf name'] = clf_name
        clfs_compare.loc[i, 'clf params'] = str(clfs[i].get_params())

        # get cross validation results
        cv_results = cross_validate(estimator=clfs[i], X=X_train, y=y_train, scoring=scoring, cv=cv, n_jobs=n_jobs, return_train_score=True)
        clfs_compare.loc[i, 'clf time'] = cv_results['fit_time'].mean()
        clfs_compare.loc[i, f'clf train {scoring} mean (CrossVal)'] = cv_results['train_score'].mean()
        clfs_compare.loc[i, f'clf validation {scoring} mean (CrossVal)'] = cv_results['test_score'].mean()
        clfs_compare.loc[i, f'clf validation {scoring} std (CrossVal)'] = cv_results['test_score'].std()

        # test model (testing dataset)
        clfs[i].fit(X_train, y_train)
        clfs_compare.loc[i, 'accuracy (testing dataset)'] = accuracy_score(y_test, clfs[i].predict(X_test))

    clfs_compare.sort_values(by=f'clf validation {scoring} mean (CrossVal)', ascending=False, inplace=True, ignore_index=True)
    
    # plot 
    g = sns.barplot(f'clf validation {scoring} mean (CrossVal)','clf name',data = clfs_compare, palette="Blues_d",orient = "h", saturation=.8,linewidth=2.5, **{'xerr':clfs_compare[f'clf validation {scoring} std (CrossVal)']})
    g.set_xlabel("Mean Validation Accuracy (CrossVal)")
    g.set_ylabel("Classifer")
    g.set_title("Cross validation scores")
    
    return clfs_compare

##### hyper-parameters tuning
    def tune_params(clf, p_grid, cv, X_train, y_train, X_test, y_test, scoring='accuracy', n_jobs=-1):
    '''
    Tune paramaters of a classifier through GridSearchCV, and plot the confusion matrix with the best parameters set. 
    @clf: a given classifier
    @p_grid: a dictionary, different sets of parameters
    @X_train: 2-d array, train data of descriptive features
    @y_train: 1-d array, train data of target feature
    @X_test: 2-d array, test data of descriptive features
    @y_test: 1-d array, test data of target feature
    @scoring: metric to measure model perfomance
    @n_jobs: int
    @return the classier with the best parameters set.
    '''
    # train, tune and validate
    gs = GridSearchCV(clf, p_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)    
    gs.fit(X_train, y_train)
    gs_best = gs.best_estimator_
    print("Best parameters set found on development set:")
    print(f'\n{gs.best_params_}')
    if len(p_grid) <= 2: 
        print("\nGrid scores on development set:\n")
        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            print(f'{mean:.3f} (+/-{std:.3f}) for {params}')
    
    # test
    print("\nDetailed classification report:")
    print("\nThe model is trained on the full train set.")
    print("The scores are computed on the full test set.")
    y_true, y_pred = y_test, gs.predict(X_test)
    print(f'\n{classification_report(y_true, y_pred)}')
    print("\nNormalized confusion matrix")
    plot_confusion_matrix(gs, X_test, y_test, cmap=plt.cm.Blues, normalize='all')
    plt.show()
    return gs_best

###### ensemble models
    def ensemble_best_clfs_and_predict(clfs_best, X_train, y_train, X_query, y_test=None,n_jobs=-1):
    '''
    Ensemble best classifiers through VotingClassifier and predict the given X_query. If given y_test, then test the ensembled classifer.
    @clf_best:
    @X_train: 2-d array, train data of descriptive features
    @y_train: 1-d array, train data of target feature
    @X_query: 2-d array, query data of descriptive features for predicting the y
    @y_test: 1-d array, test data of target feature. If None, means do only prediction, no testing.
    @n_jobs: an int.
    @return: 1-d array, predicted y.
    '''
    voting = VotingClassifier(estimators=clfs_best, voting='hard', n_jobs=n_jobs)
    voting.fit(X_train,y_train)
    y_pred = voting.predict(X_query)
    y_true = y_test
    
    # if given y_test, do testing process
    if y_test.any():
        print("\nDetailed classification report:")
        print(f'\n{classification_report(y_true, y_pred)}')
        print("\nNormalized confusion matrix")
        plot_confusion_matrix(voting, X_query, y_test, cmap=plt.cm.Blues, normalize='all')
        plt.show()
    return y_pred



################### Document-based sentiment analysis: ti-idf #################
import pandas as pd
import numpy as np
import nltk
import re
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, truncnorm, randint
from sklearn.model_selection import KFold
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.model_selection import cross_validate

def splitDataset(data, review_id, cleaned_reviews, score_col, train_size_ = 0.7, test_size_=0.3): 
    '''
    data is the dataframe containing all the values
    '''
    X = data[[review_id, cleaned_reviews]]
    y = data[[review_id, score_col]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size_, test_size=test_size_, random_state=42)
    
    return X_train, X_test, y_train, y_test

def loadData(datafile): 
    '''
    datafile: cleaned data in a csv file 
    returns the data loaded into a dataframe with target categorical feature added
    '''
    
    data_cleaned = pd.read_csv(datafile) 
    data_cleaned.rename(columns = {'score' : 'review_score'}, inplace=True)
    data_cleaned['review_category'] = [0 if x == 5 or x == 4 else 1 for x in data_cleaned['review_score']]
    
    return data_cleaned
  
def sampleDataset(X, y, target, minority, majority, sampling, id_col, reviews_col): 
    
    '''
    X: the X data to sample 
    y: the y data to sample 
    target: the col containing the target variable (e.g. 'review_category')
    minority: the minority class label (str)
    majority: the majority class label (str)
    sampling: sampling method - upsample or downsample 
    id_col: the column containing the ids (e.g. 'reviewId')
    reviews_col: the column containing the reviews (e.g. 'clean_reviews')
    '''
    
    training = pd.concat([X, y[[target]]], axis=1)
    minority = training[training[target] == minority]
    majority = training[training[target] == majority]
    
    # upsample the minority class 
    if sampling == 'upsample': 
        sampled = resample(minority,
                          replace=True, 
                          n_samples=len(majority), 
                          random_state=42) 
        
        sampled_df =  pd.concat([majority, sampled])
    
    # downsample the majority class 
    elif sampling == 'downsample': 
        sampled = resample(majority,
                                replace = False, 
                                n_samples = len(minority), 
                                random_state = 42) 
        sampled_df = pd.concat([minority, sampled])
    
    
    X_sampled = sampled_df[[id_col, reviews_col]]
    y_sampled = sampled_df[[id_col, target]]
    
    return X_sampled, y_sampled   

def tf_idf_weight_norm_extract(train_data, test_data, raw_data_col, norm = "l2", max_features=None,
                               max_df = 1.0, min_df = 1, ngram_range=(1,1)) :
    '''
    Calculate normalized tf-idf weight.
    @data: the dataframe containing the raw (unvectorized) data training. 
    Please note that this is a modified version of the tf-idf function 
    tf_idf_weight_norm. we are using the TfidfVectorizer module so the input is the unvectorized data.
    @raw_data_col : the column containing the unvectorized cleaned reviews. e.g. clean_reviews
    @norm: a string ('l2', 'l1'), representing normalization method
    @max features : only consider the top max_features ordered by term frequency across the corpus
    @max_df : ignore terms that have a document frequency strictly higher than the given threshold  
    @min_df : ignore terms that have a document frequency strictly lower than the given threshold
    @df_type : indicates if the tf-idf is being generated for 'train' or 'test' data
    @return: a dataframe
    '''
    
    data_array = list(train_data[raw_data_col])
    data_array_test = list(test_data[raw_data_col])
    
    if norm == 'l2': # Euclidean normalization
        tf_idf = TfidfVectorizer(max_features = max_features, max_df = max_df, min_df = min_df, 
                                 norm = 'l2', token_pattern=r"\b\w+\b", stop_words=None, 
                                 ngram_range=ngram_range, analyzer='word')
        tf_idf_weight_norm_train = tf_idf.fit_transform(data_array).todense()  
        tf_idf_weight_norm_test = tf_idf.transform(data_array_test).todense()
    
    elif norm == 'l1': # Manhattan normalization
        tf_idf = TfidfVectorizer(max_features = max_features, max_df = max_df, min_df = min_df, 
                                 norm = 'l1', token_pattern=r"\b\w+\b", stop_words=None, 
                                 ngram_range=ngram_range, analyzer='word')
        tf_idf_weight_norm_train = tf_idf.fit_transform(data_array).todense()
        tf_idf_weight_norm_test = tf_idf.transform(data_array_test).todense()
    
    else: 
        print("method must be 'l1 or 'l2', default is l2")

    tf_idf_weight_norm_train = pd.DataFrame(tf_idf_weight_norm_train, 
                                            columns = tf_idf.get_feature_names())
    tf_idf_weight_norm_test = pd.DataFrame(tf_idf_weight_norm_test, 
                                           columns = tf_idf.get_feature_names())

    return tf_idf_weight_norm_train, tf_idf_weight_norm_test

def applyTfidf_extract(train_df, test_df, raw_data_col_, method, review_df_train, review_df_test, max_features_, max_df_=1.0, min_df_=1, ngram = (1,1)):
    '''
    @to_norm : data to normalize, as a df 
    @method : normalization method to pass to tf_idf_weight_norm function
    @review_df_train : df containing the review ids for the training data, index gets reset in the function 
    @review_df_test : df containing the review ids for the test data, index gets reset in the function 
    '''
    tf_idf_weight_norm_train, tf_idf_weight_norm_test = tf_idf_weight_norm_extract(train_data = train_df, test_data = test_df, raw_data_col = raw_data_col_, norm = method, max_features= max_features_, max_df=max_df_, min_df=min_df_, ngram_range=ngram)
    
    
    review_df_train.reset_index(inplace=True)
    review_df_train = review_df_train[['reviewId']]
    
    review_df_test.reset_index(inplace=True)
    review_df_test = review_df_test[['reviewId']]
    
    train_tfidf = pd.concat([review_df_train, tf_idf_weight_norm_train], axis=1)
    test_tfidf = pd.concat([review_df_test, tf_idf_weight_norm_test], axis=1)
    
    train_tfidf.set_index('reviewId', inplace = True)
    test_tfidf.set_index('reviewId', inplace = True)
    
    return train_tfidf, test_tfidf


def apply_model(model_type,tfidf_train, y_train_, tfidf_test, y_test_, method='gini'):
    
    '''
    model_type: 'dt' for decision tree, 'rf' for random forest, or 'mnb' for multinomial naive bayes
    '''
    y_train_list = list(y_train_.review_category)
    y_test_list = list(y_test_.review_category)
    
    if model_type == 'dt': 
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == 'rf': 
        model = RandomForestClassifier(random_state=42)
    else: 
        model = MultinomialNB()
    
    model.fit(tfidf_train,y_train_list)
    
    y_pred = model.predict(tfidf_test)
    
    acc = accuracy_score(y_test_list, y_pred)
    
    print(f'Accuracy score: {acc:.3%}')
    print("\nDetailed classification report:")
    print("\nThe model is trained on the full train set.")
    print("The scores are computed on the full test set.")
    y_true, y_pred = y_test_list, model.predict(tfidf_test)
    print(f'\n{classification_report(y_true, y_pred)}')
    print("\nNormalized confusion matrix")
    plot_confusion_matrix(model, tfidf_test, y_test_list, cmap=plt.cm.Blues, normalize='all')
    plt.show()
    
    return model

def tune_model(model_type, tfidf_train, y_train_, tfidf_test, y_test_): 
    
    '''
    use randomized grid search to find best parameters for the model. 
    
    '''
    
    if model_type == 'dt':
        model_params = {
            'max_features': randint(1,100),
            'min_samples_split': randint(2,100),
            'max_depth' : randint(2,150)}
        
        model = DecisionTreeClassifier(random_state=42)
    
    elif model_type == 'rf':
        model_params = {
            'n_estimators': randint(4,200),
            'max_features': randint(1,100),
            'min_samples_split': randint(2,100),
            'max_depth' : randint(2,150)}
        model = RandomForestClassifier(random_state=42)
    
    elif model_type == 'mnb':
        model_params = {'alpha': [1, 2, 3, 5]}
        model = MultinomialNB()
        
    else: 
        print("model_type should be one of 'dt', 'rf' or 'mnb'")
        
    
    y_train_list = list(y_train_.review_category)
    y_test_list = list(y_test_.review_category)
    
    
    if model_type == 'mnb':
        cv_model = GridSearchCV(estimator=mnb, param_grid = model_params)
    else:
        cv_model = RandomizedSearchCV(estimator=model, param_distributions = model_params, random_state=42)
        
    cv_model.fit(tfidf_train,y_train_list)
    
    print('the best parameters calculated are:')
    print(cv_model.best_params_)
    print('\n')
    
    y_true, y_pred = y_test_list, cv_model.predict(tfidf_test)
    acc = accuracy_score(y_test_list, y_pred)
    
    print(f'Accuracy score: {acc:.3%}')
    print("\nDetailed classification report:")
    print("\nThe model is trained on the full train set.")
    print("The scores are computed on the full test set.")
    print(f'\n{classification_report(y_true, y_pred)}')
    print("\nNormalized confusion matrix")
    plot_confusion_matrix(cv_model, tfidf_test, y_test_list, cmap=plt.cm.Blues, normalize='all')
    plt.show()
    
    return cv_model.best_params_


def kFoldCrossVal(n, model, x_train, y_train):  
    
    '''
    n : number of splits 
    model : the classifier to validate
    x_train : unsampled training descriptive feature data
    y_train : unsampled training data labels
    function written based on content here: 
    https://stackoverflow.com/questions/46010617/do-i-use-the-same-tfidf-vocabulary-in-k-fold-cross-validation
    
    '''
    x_data = x_train.copy()
    y_data = y_train.copy()

    x_data.set_index('reviewId', inplace=True)
    y_data.set_index('reviewId', inplace=True)
                 
    kf = KFold(n_splits=n, shuffle=True, random_state=0)
    
    scores = 0
    split = 0
    for train_index, test_index in kf.split(x_data):
        x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
        
        # resetting the index so we can use the downsampling function
        y_train = y_train.reset_index()
        x_train = x_train.reset_index()
        x_test = x_test.reset_index()
        y_test = y_test.reset_index()
    
        # downsampling the training data
        x_train, y_train = sampleDataset(x_train, y_train, 'review_category', 0, 1, 'downsample', 'reviewId', 'clean_reviews')

        # tf-idf vectorization 
        tfidf = TfidfVectorizer()
        x_train = tfidf.fit_transform(list(x_train.clean_reviews)).todense()
        x_test = tfidf.transform(list(x_test.clean_reviews)).todense()

        # modelling 
        model.fit(x_train, list(y_train.review_category))
        y_pred = model.predict(x_test)
        y_test = list(y_test.review_category)
        
        # evaluation
        score = accuracy_score(y_test, y_pred)
        scores += score 
        split += 1 
        
        print(f'Evaluation Metrics for Fold={split}')
        print(f'Accuracy score: {score:.3%}')
        print("\nDetailed classification report:")
        print("\nThe model is trained on the full train set.")
        print("The scores are computed on the full test set.")
        print(f'\n{classification_report(y_test, y_pred)}')
        print("\nNormalized confusion matrix")
        plot_confusion_matrix(model, x_test, y_test, cmap=plt.cm.Blues, normalize='all')
        plt.show()

    avg_score = scores / n
    print(f'The Average Accuracy Score is: {avg_score:.2%}')


def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i]) for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i]) for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


############################# Aspect-based sentiment analysis: aspect extraction (topic modeling) ########################
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
stop_words = set(stopwords.words('english')) 
nltk.download('punkt')

def feature(df, feature):
    
    feature = feature.astype(str) # Dummy text 
  
    # sent_tokenize is one of instances of  
    # PunktSentenceTokenizer from the nltk.tokenize.punkt module 
    res = []
    remove_word = ['love','walmart']
    for i in feature: 
        unique = ' '.join(set(i.split(' ')))
        wordsList = nltk.word_tokenize(unique) #     Word tokenizers is used to find the words and punctuation in a string 
        wordsList = [w for w in wordsList if not w in stop_words and not w in remove_word]  # removing stop words from wordList 

        #  Using a Tagger. Which is part-of-speech tagger or POS-tagger.  
        tagged = [nltk.pos_tag(wordsList)]
        for row in tagged:
            tmp = []
            for pair in row:
                word, tag = pair
                if tag == 'NN':
                    tmp.append(word)
            res.append(' '.join(tmp))
        df['tagged_review'] = pd.DataFrame(res)
    return df.head()


def topicModel(df, no_topics, no_top_words, model):
    documents = df['tagged_review']
    no_terms = 10000 #Set variable number of terms

    # NMF uses the tf-idf count vectorizer
    # Initialise the count vectorizer with the English stop words
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=no_terms, stop_words='english')
    document_matrix = vectorizer.fit_transform(documents)     # Fit and transform the text
    feature_names = vectorizer.get_feature_names() #get features

        # Apply NMF topic model to document-term matrix
    nmf_model = NMF(n_components=no_topics, random_state=42, alpha=.1, l1_ratio=.5, init='nndsvd').fit(document_matrix)

        # Run LDA
    lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(document_matrix)

        # Use NMF model to assign topic to papers in corpus
    nmf_topic_values = nmf_model.transform(document_matrix)
    lda_topic_values = lda_model.transform(document_matrix)
    df['NMF Topic'] = nmf_topic_values.argmax(axis=1)
    df['LDA Topic'] = lda_topic_values.argmax(axis=1)

    if model == 'NMF':
        return display_topics(nmf_model, feature_names, no_top_words)
    if model == 'LDA':
        return display_topics(lda_model, feature_names, no_top_words)

def apply_SGD(tfidf_train, y_train_, tfidf_test, y_test_):
    

    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)
    clf.fit(X_train_tfidf, y_train)
    
    
    y_pred = clf.predict(tfidf_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f'Accuracy score: {acc:.3%}')
    print("\nDetailed classification report:")
    print("\nThe model is trained on the full train set.")
    print("The scores are computed on the full test set.")
    
    print(f'\n{classification_report(y_test, y_pred)}')
    print("\nNormalized confusion matrix")
    plot_confusion_matrix(clf, tfidf_test, y_test, cmap=plt.cm.Blues, normalize='all')
    plt.show()
    
    return clf


def tune_SGD(tfidf_train, y_train_, tfidf_test, y_test_): 
    
    '''
    use grid search to find best parameters for the model. 
    
    '''
    
    clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)
    
    parameters = {'alpha': (1e-1, 1e-3)}
    
    gs_SGD = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    gs_SGD.fit(tfidf_train, y_train_)
    
    print('the best parameters calculated are:')
    print(gs_SGD.best_params_)
    print('\n')
    
    best_model = gs_MNB.best_estimator_
    print('the score of the best estimator is:')
    print(best_model.score(tfidf_test, y_test_))
