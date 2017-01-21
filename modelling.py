import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier 
from sklearn.cross_validation import StratifiedKFold
import evaluation
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


#################################################
###########CROSS VALIDATION CODE#################
#################################################

def perform_cross_validation(model, kfold_train_test, features):
  score_count = 0
  score_total = 0.0
  test_data = []
  for X_train, y_train, X_test, y_test in kfold_train_test:
    #print pd.DataFrame(X_train).head(0),pd.DataFrame(X_test).head(0)
    #print X_train.shape,X_test.shape
    X_train = X_train[features]
    X_test = X_test[features]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score_count += 1
    score = evaluation.quadratic_weighted_kappa(y = y_test, y_pred = predictions)
    score_total += score
    print("Score " + str(score_count) + ": " + str(score))

    y_and_y_pred = pd.DataFrame({'y': y_test, 'y_pred': predictions})

    test_data.append(y_and_y_pred)

  average_score = score_total/float(score_count)
  print("Average score: " + str(average_score))
  return test_data

def perform_tfidf_cross_validation(tfv, pipeline, kfold_train_test):
    score_count = 0
    score_total = 0.0
    test_data = []
    for X_train, y_train, X_test, y_test in kfold_train_test:

      tfv.fit(X_train)
      X_train =  tfv.transform(X_train) 
      X_test = tfv.transform(X_test)
      pipeline.fit(X_train, y_train)
      predictions = pipeline.predict(X_test)
      score_count += 1
      score = evaluation.quadratic_weighted_kappa(y = y_test, y_pred = predictions)
      score_total += score
      print("Score " + str(score_count) + ": " + str(score))
      y_and_y_pred = pd.DataFrame({'y': y_test, 'y_pred': predictions})
      test_data.append(y_and_y_pred)
      
    average_score = score_total/float(score_count)
    print("Average score: " + str(average_score))
    return test_data


#################################################
###############MODELLING CODE####################
#################################################


def ouput_final_model(model, train, test, features):

  y = train["median_relevance"]
  train_with_features = train[features]
  test_with_features = test[features]
  model.fit(train_with_features, y)
  predictions = model.predict(test_with_features)
  outputresult = pd.DataFrame({"id": test["id"], "prediction": predictions})
  return outputresult


if __name__ == '__main__':
  train = pickle.load(open('train_extracted_df.pkl', 'r'))
  test = pickle.load(open('test_extracted_df.pkl', 'r'))
  y_train = train["median_relevance"]
  kfold_train_test = pickle.load(open('kfold_train_test.pkl', 'r'))
  print "kfold_train_test",kfold_train_test
  bow_v1_features = pickle.load(open('bow_v1_features_full_dataset.pkl', 'r'))
  bow_v2_features = pickle.load(open('bow_v2_features_full_dataset.pkl', 'r'))
  bow_v1_kfold_trian_test = pickle.load(open('bow_v1_kfold_trian_test.pkl', 'r'))
  bow_v2_kfold_trian_test = pickle.load(open('bow_v2_kfold_trian_test.pkl', 'r'))

  features = ['query_tokens_in_title', 'query_tokens_in_description', 'percent_query_tokens_in_description', 'percent_query_tokens_in_title', 'query_length', 'description_length', 'title_length', 'two_grams_in_q_and_t', 'two_grams_in_q_and_d', 'q_mean_of_training_relevance', 'q_median_of_training_relevance', 'avg_relevance_variance', 'average_title_1gram_similarity_1', 'average_title_2gram_similarity_1', 'average_title_1gram_similarity_2', 'average_title_2gram_similarity_2', 'average_title_1gram_similarity_3', 'average_title_2gram_similarity_3', 'average_title_1gram_similarity_4', 'average_title_2gram_similarity_4', 'average_description_1gram_similarity_1', 'average_description_2gram_similarity_1', 'average_description_2gram_similarity_2', 'average_description_1gram_similarity_2', 'average_description_1gram_similarity_3', 'average_description_2gram_similarity_3', 'average_description_1gram_similarity_4', 'average_description_2gram_similarity_4']
  
  ####Random forest model#####
  print("Begin random forest model")
  model = RandomForestClassifier(n_estimators=300, n_jobs=1, min_samples_split=10, random_state=1, class_weight='auto')
  rf_cv_test_data = perform_cross_validation(model, kfold_train_test, features)
  pickle.dump(rf_cv_test_data, open('rf_cv_test_data.pkl', 'w'))
  rf_final_predictions = ouput_final_model(model, train, test, features)
  pickle.dump(rf_final_predictions, open('rf_final_predictions.pkl', 'w'))


  ####SVC Model####
  print("Begin SVC model")
  scl = StandardScaler()
  svm_model = SVC(C=10.0, random_state = 1, class_weight = {1:1, 2:1, 3:1, 4:1})
  model = Pipeline([('scl', scl), ('svm', svm_model)])
  svc_cv_test_data = perform_cross_validation(model, kfold_train_test, features)
  pickle.dump(svc_cv_test_data, open('svc_cv_test_data.pkl', 'w'))
  svc_final_predictions = ouput_final_model(model, train, test, features)
  pickle.dump(svc_final_predictions, open('svc_final_predictions.pkl', 'w'))


  ####AdaBoost Model####
  print("Begin AdaBoost model")
  model = AdaBoostClassifier(n_estimators=200, random_state = 1, learning_rate = 0.25)
  adaboost_cv_test_data = perform_cross_validation(model, kfold_train_test, features)
  pickle.dump(adaboost_cv_test_data, open('adaboost_cv_test_data.pkl', 'w'))
  adaboost_final_predictions = ouput_final_model(model, train, test, features)
  pickle.dump(adaboost_final_predictions, open('adaboost_final_predictions.pkl', 'w'))

  ####GBDT####
  print ("Begin GBDT model")
  model =  GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=0)
  gbrt_cv_test_data = perform_cross_validation(model, kfold_train_test, features)
  pickle.dump(gbrt_cv_test_data, open('gbdt_cv_test_data.pkl', 'w'))
  gbrt_final_predictions = ouput_final_model(model, train, test, features)
  pickle.dump(gbrt_final_predictions, open('gbdt_final_predictions.pkl', 'w'))

  ####Model using bag of words TFIDF v1####
  print("Begin TFIDF v1 model")
  idx = test.id.values.astype(int)
  train_v1, y_v1, test_v1, y_test_v1 = bow_v1_features

  tfv = TfidfVectorizer(min_df=3,  max_features=None, 
          strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
          ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
          stop_words = 'english')

  pipeline = Pipeline([('svd', TruncatedSVD(n_components=400)), ('scl', StandardScaler()), ('svm', SVC(C=10))])

  tfidf_v1_test_data = perform_tfidf_cross_validation(tfv, pipeline, bow_v1_kfold_trian_test)
  pickle.dump(tfidf_v1_test_data, open('tfidf_v1_test_data.pkl', 'w'))

  tfv.fit(train_v1)
  X_train =  tfv.transform(train_v1) 
  X_test = tfv.transform(test_v1)
  pipeline.fit(X_train, y_v1)
  predictions = pipeline.predict(X_test)

  outputresult = pd.DataFrame({"id": idx, "prediction": predictions})
  pickle.dump(outputresult, open('tfidf_v1_final_predictions.pkl', 'w'))

  
  ####Model using bag of words TFIDF v2####
  print("Begin TFIDF v2 model")
  data = pickle.load(open('bow_v2_features_full_dataset.pkl', 'r'))
  bow_v2_kfold_trian_test = pickle.load(open('bow_v2_kfold_trian_test.pkl', 'r'))
  idx = test.id.values.astype(int)

  tfv = TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')
  #create sklearn pipeline, fit all, and predit test data
  pipeline = Pipeline([('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
  ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
  ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
  tfidf_v2_test_data = perform_tfidf_cross_validation(tfv, pipeline, bow_v2_kfold_trian_test)
  pickle.dump(tfidf_v2_test_data, open('tfidf_v2_test_data.pkl', 'w'))

  #Output final model for TFIDF v2
  train_v2, y_v2, test_v2, y_v2_empty = bow_v2_features
  tfv.fit(train_v2)
  X_train =  tfv.transform(train_v2) 
  X_test = tfv.transform(test_v2)
  pipeline.fit(X_train, y_v2)
  predictions = pipeline.predict(X_test)


