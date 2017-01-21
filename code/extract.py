from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
import pandas as pd
import re
from sklearn.cross_validation import KFold
import cPickle
from bs4 import BeautifulSoup
from sklearn.cross_validation import StratifiedKFold

 
def stem_data(data):
    stemmer = PorterStemmer()

    for i, row in data.iterrows():

        q = (" ").join([z for z in BeautifulSoup(row["query"]).get_text(" ").split(" ")])
        t = (" ").join([z for z in BeautifulSoup(row["product_title"]).get_text(" ").split(" ")]) 
        d = (" ").join([z for z in BeautifulSoup(row["product_description"]).get_text(" ").split(" ")])

        q=re.sub("[^a-zA-Z0-9]"," ", q)
        t=re.sub("[^a-zA-Z0-9]"," ", t)
        d=re.sub("[^a-zA-Z0-9]"," ", d)

        q= (" ").join([stemmer.stem(z) for z in q.split()])
        t= (" ").join([stemmer.stem(z) for z in t.split()])
        d= (" ").join([stemmer.stem(z) for z in d.split()])
        
        data.set_value(i, "query", unicode(q))
        data.set_value(i, "product_title", unicode(t))
        data.set_value(i, "product_description", unicode(d))

def remove_stop_words(data):

    stop = stopwords.words('english')

    for i, row in data.iterrows():

        q = row["query"].lower().split(" ")
        t = row["product_title"].lower().split(" ")
        d = row["product_description"].lower().split(" ")

        q = (" ").join([z for z in q if z not in stop])
        t = (" ").join([z for z in t if z not in stop])
        d = (" ").join([z for z in d if z not in stop])

        data.set_value(i, "query", q)
        data.set_value(i, "product_title", t)
        data.set_value(i, "product_description", d)

def get_n_gram_string_similarity(s1, s2, n):

    s1 = set(get_n_grams(s1, n))
    s2 = set(get_n_grams(s2, n))
    if len(s1.union(s2)) == 0:
        return 0
    else:
        return float(len(s1.intersection(s2)))/float(len(s1.union(s2)))

def get_n_grams(s, n):

    token_pattern = re.compile(r"(?u)\b\w+\b")
    word_list = token_pattern.findall(s)
    n_grams = []


    if n > len(word_list):
        return []
    
    for i, word in enumerate(word_list):
        n_gram = word_list[i:i+n]
        if len(n_gram) == n:
            n_grams.append(tuple(n_gram))
    return n_grams

def calculate_nearby_relevance_tuple(group, row, col_name, ngrams):

    ngrams = range(1, ngrams + 1)
    weighted_ratings = {rating: {ngram: [0,0] for ngram in ngrams} for rating in range(1,5)}

    for i, group_row in group.iterrows():
        if group_row['id'] != row['id']:

            for ngram in ngrams:
                similarity = get_n_gram_string_similarity(row[col_name], group_row[col_name], ngram)
                weighted_ratings[group_row['median_relevance']][ngram][1] += similarity
                weighted_ratings[group_row['median_relevance']][ngram][0] += 1

    return weighted_ratings
 
def extract_features(data):
    token_pattern = re.compile(r"(?u)\b\w+\b")
    data["query_tokens_in_title"] = 0.0
    data["query_tokens_in_description"] = 0.0
    data["percent_query_tokens_in_description"] = 0.0
    data["percent_query_tokens_in_title"] = 0.0
    for i, row in data.iterrows():
        query = set(x.lower() for x in token_pattern.findall(row["query"]))
        title = set(x.lower() for x in token_pattern.findall(row["product_title"]))
        description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
        if len(title) > 0:
            data.set_value(i, "query_tokens_in_title", float(len(query.intersection(title)))/float(len(title)))
            data.set_value(i, "percent_query_tokens_in_title", float(len(query.intersection(title)))/float(len(query)))
        if len(description) > 0:
            data.set_value(i, "query_tokens_in_description", float(len(query.intersection(description)))/float(len(description)))
            data.set_value(i, "percent_query_tokens_in_description", float(len(query.intersection(description)))/float(len(query)))
        data.set_value(i, "query_length", len(query))
        data.set_value(i, "description_length", len(description))
        data.set_value(i, "title_length", len(title))

        two_grams_in_query = set(get_n_grams(row["query"], 2))
        two_grams_in_title = set(get_n_grams(row["product_title"], 2))
        two_grams_in_description = set(get_n_grams(row["product_description"], 2))

        data.set_value(i, "two_grams_in_q_and_t", len(two_grams_in_query.intersection(two_grams_in_title)))
        data.set_value(i, "two_grams_in_q_and_d", len(two_grams_in_query.intersection(two_grams_in_description)))

def extract_training_and_test_features(train, test):
    train_group = train.groupby('query')
    test_group = test.groupby('query')
    test["q_mean_of_training_relevance"] = 0.0
    test["q_median_of_training_relevance"] = 0.0
    test["avg_relevance_variance"] = 0
    for i, row in train.iterrows():
        group = train_group.get_group(row["query"])
        
        q_mean = group["median_relevance"].mean()
        train.set_value(i, "q_mean_of_training_relevance", q_mean)
        test.loc[test["query"] == row["query"], "q_mean_of_training_relevance"] = q_mean

        q_median = group["median_relevance"].median()
        train.set_value(i, "q_median_of_training_relevance", q_median)
        test.loc[test["query"] == row["query"], "q_median_of_training_relevance"] = q_median

        avg_relevance_variance = group["relevance_variance"].mean()
        train.set_value(i, "avg_relevance_variance", avg_relevance_variance)
        test.loc[test["query"] == row["query"], "avg_relevance_variance"] = avg_relevance_variance

        weight_dict = calculate_nearby_relevance_tuple(group, row, col_name = 'product_title', ngrams = 2)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_title_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    train.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    train.set_value(i, variable_name, 0)

        weight_dict = calculate_nearby_relevance_tuple(group, row, col_name = 'product_description', ngrams = 2)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_description_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    train.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    train.set_value(i, variable_name, 0)


    for i, row in test.iterrows():

        #if not len(row['query']):
        group = test_group.get_group(row["query"])
        weight_dict = calculate_nearby_relevance_tuple(group, row, col_name = 'product_title', ngrams = 2)
        for rating in weight_dict:
            print "test################"
            for ngram in weight_dict[rating]:
                variable_name = "average_title_" + str(ngram) + "gram_similarity_" + str(rating)
                #print "weight_dict[rating][ngram][0]",weight_dict[rating][ngram][0]
                if weight_dict[rating][ngram][0] != 0:
                    test.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    test.set_value(i, variable_name, 0)

        weight_dict = calculate_nearby_relevance_tuple(group, row, col_name = 'product_description', ngrams = 2)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_description_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    test.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    test.set_value(i, variable_name, 0)
        #return (train,test)

def extract_bow_v1_features(train, test):

    traindata = train['query'] + ' ' + train['product_title']
    y_train = train['median_relevance']
    testdata = test['query'] + ' ' + test['product_title']
    if 'median_relevance' in test.columns.values:
        y_test = test['median_relevance']
    else:
        y_test = []

    return (traindata, y_train, testdata, y_test)


def extract_bow_v2_features(train, test, test_contains_labels = False):

    s_data = []
    s_labels = []
    t_data = []
    t_labels = []
    stemmer = PorterStemmer()    
    
    for i, row in train.iterrows():
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        s_labels.append(str(train["median_relevance"][i]))
    for i, row in test.iterrows():
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
        if test_contains_labels:
            t_labels.append(str(test["median_relevance"][i]))
            
    return (s_data, s_labels, t_data, t_labels)

def extract(train1, test1):

    print "Extracting training features"
    extract_features(train1)
    print "Extracting test features"
    extract_features(test1)

    print "Extracting features that must use data in the training set for both test and training data extraction"
    extract_training_and_test_features(train1, test1)

    y_train = train.loc[:,"median_relevance"]
    train.drop("median_relevance", 1)
    if 'median_relevance' in test.columns.values:
        y_test = test.loc[:, "median_relevance"]
        test.drop("median_relevance", 1)
    else:
        y_test = []

    return train, y_train, test, y_test


if __name__ == '__main__':

    train = pd.read_csv('input/train1.csv').fillna("")
    test = pd.read_csv('input/test1.csv').fillna("")

    kfold_train_test = []
    bow_v1_kfold_trian_test = []
    bow_v2_kfold_trian_test = []    
    kf = StratifiedKFold(train["query"], n_folds=5)

    for train_index, test_index in kf:
        
        X_train = train.loc[train_index]
        y_train = train.loc[train_index,"median_relevance"]
        X_test = train.loc[test_index]
        y_test = train.loc[test_index, "median_relevance"]

        bow_v1_features = extract_bow_v1_features(X_train, X_test)
        bow_v1_kfold_trian_test.append(bow_v1_features)
        bow_v2_features = extract_bow_v2_features(X_train, X_test, test_contains_labels = True)
        bow_v2_kfold_trian_test.append(bow_v2_features)

        extract(X_train, X_test)
        kfold_train_test.append((X_train, y_train, X_test, y_test))

    cPickle.dump(kfold_train_test, open('kfold_train_test.pkl', 'w'))
    cPickle.dump(bow_v1_kfold_trian_test, open('bow_v1_kfold_trian_test.pkl', 'w'))
    cPickle.dump(bow_v2_kfold_trian_test, open('bow_v2_kfold_trian_test.pkl', 'w'))

    print "Extracting bag of words v1 features"
    bow_v1_features = extract_bow_v1_features(train, test)
    cPickle.dump(bow_v1_features, open('bow_v1_features_full_dataset.pkl', 'w'))
    
    print "Extracting bag of words v2 features"
    bow_v2_features = extract_bow_v2_features(train, test)
    cPickle.dump(bow_v2_features, open('bow_v2_features_full_dataset.pkl', 'w'))

    extract(train, test)
    cPickle.dump(train, open('train_extracted_df.pkl', 'w'))
    cPickle.dump(test, open('test_extracted_df.pkl', 'w'))
