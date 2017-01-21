import evaluation
import pickle
import math
import pandas as pd

rf_cv_predictions = pickle.load(open('rf_cv_test_data.pkl', 'r'))
svc_cv_predictions = pickle.load(open('svc_cv_test_data.pkl', 'r'))
adaboost_cv_predictions = pickle.load(open('adaboost_cv_test_data.pkl', 'r'))
gbdt_cv_predictions = pickle.load(open('gbdt_cv_test_data.pkl', 'r'))
tfidf_v1_cv_predictions = pickle.load(open('tfidf_v1_test_data.pkl', 'r'))
tfidf_v2_cv_predictions = pickle.load(open('tfidf_v2_test_data.pkl', 'r'))
knn_final_predictions = pickle.load(open('knn_final_predictions.pkl', 'r'))

preds = [rf_cv_predictions, svc_cv_predictions, adaboost_cv_predictions, gbdt_cv_predictions,tfidf_v1_cv_predictions, tfidf_v2_cv_predictions]

y_true = preds[0][0]['y']

wt = [0.2, .1, .25, .5, .2,0,5]

#41.6%
# wt_list = [(a,b,c,d,e,f,g) for a in wt for b in wt for c in wt for d in wt for e in wt for f in wt for g in wt]
wt_list = [(a,b,c,d,e,f) for a in wt for b in wt for c in wt for d in wt for e in wt for f in wt]

print "preds",preds[0]
wt_final = []
for w in wt_list:
	if sum(w) == 1.0:
		wt_final.append(w)

#Find the optimal weights.
max_average_score = 0
max_weights = None
for wt in wt_final:
	total_score = 0
	for i in range(5):
		y_true = preds[0][i]['y']
		weighted_prediction = sum([wt[x] * preds[x][i]['y_pred'].astype(int).reset_index() for x in range(6)])
		weighted_prediction = [round(p) for p in weighted_prediction['y_pred']]
		total_score += evaluation.quadratic_weighted_kappa(y = y_true, y_pred = weighted_prediction)
	average_score = total_score/5.0
	if average_score > max_average_score:
		max_average_score = average_score
		max_weights = wt
print "Best set of weights: " + str(max_weights)
print "Corresponding score: " + str(max_average_score)


rf_final_predictions = pickle.load(open('rf_final_predictions.pkl', 'r'))
svc_final_predictions = pickle.load(open('svc_final_predictions.pkl', 'r'))
adaboost_final_predictions = pickle.load(open('adaboost_final_predictions.pkl', 'r'))
tfidf_v1_final_predictions = pickle.load(open('tfidf_v1_final_predictions.pkl', 'r'))
tfidf_v2_final_predictions = pickle.load(open('tfidf_v2_final_predictions.pkl', 'r'))
knn_final_predictions = pickle.load(open('knn_final_predictions.pkl', 'r'))
preds = [rf_final_predictions, svc_final_predictions, adaboost_final_predictions, tfidf_v1_final_predictions, tfidf_v2_final_predictions,knn_final_predictions]

weighted_prediction = sum([max_weights[x] * preds[x]["prediction"].astype(int) for x in range(5)])
weighted_prediction = [int(round(p)) for p in weighted_prediction]
