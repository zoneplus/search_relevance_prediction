#encoding=utf-8
from pybrain.structure import *
from pybrain.tools.shortcuts import buildNetwork
import pickle
import evaluation
import pandas as pd
import math
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,ClassificationDataSet
import matplotlib.pyplot as plt
from pybrain.tools.validation import CrossValidator,Validator

def nn():
    DS = ClassificationDataSet(28, 1, nb_classes=4)
    train = pickle.load(open('train_extracted_df.pkl', 'r'))
    y = train["median_relevance"]
    kfold_train_test = pickle.load(open('kfold_train_test.pkl', 'r'))
    features = ['query_tokens_in_title', 'query_tokens_in_description', 'percent_query_tokens_in_description', 'percent_query_tokens_in_title', 'query_length', 'description_length', 'title_length', 'two_grams_in_q_and_t', 'two_grams_in_q_and_d', 'q_mean_of_training_relevance', 'q_median_of_training_relevance', 'avg_relevance_variance', 'average_title_1gram_similarity_1', 'average_title_2gram_similarity_1', 'average_title_1gram_similarity_2', 'average_title_2gram_similarity_2', 'average_title_1gram_similarity_3', 'average_title_2gram_similarity_3', 'average_title_1gram_similarity_4', 'average_title_2gram_similarity_4', 'average_description_1gram_similarity_1', 'average_description_2gram_similarity_1', 'average_description_2gram_similarity_2', 'average_description_1gram_similarity_2', 'average_description_1gram_similarity_3', 'average_description_2gram_similarity_3', 'average_description_1gram_similarity_4', 'average_description_2gram_similarity_4']
    train = train[features]
    for i in range(len(y)):
        DS.addSample(train.values[i],y[i])
     X = DS['input']
    Y = DS['target']
    dataTrain, dataTest = DS.splitWithProportion(0.8)
    xTrain, yTrain = dataTrain['input'], dataTrain['target']
    xTest, yTest = dataTest['input'], dataTest['target']
    #fnn = RecurrentNetwork()
    fnn = FeedForwardNetwork()
    #fnn=buildNetwork(1,40,1,hiddenclass=TanhLayer, bias=True, outclass=SoftmaxLayer)
    #fnn=buildNetwork(1,40,1,hiddenclass=LSTMLayer, bias=True, outclass=SoftmaxLayer)
    inLayer = LinearLayer(28, name='inLayer')
    hiddenLayer = SigmoidLayer(40, name='hiddenLayer0')
    outLayer =LinearLayer(4, name='outLayer')

    fnn.addInputModule(inLayer)
    fnn.addModule(hiddenLayer)
    fnn.addOutputModule(outLayer)

    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    fnn.addConnection(in_to_hidden)
    fnn.addConnection(hidden_to_out)
    fnn.sortModules()

    trainer = BackpropTrainer(fnn, DS, verbose = True, learningrate=0.01)
    #trainer.trainUntilConvergence(maxEpochs=1000)
    trainer.trainEpochs(epochs=5)
    prediction = fnn.activateOnDataset(dataTest)
    out=[]
    total_score = 0
    for i in prediction:
        class_index = max(xrange(len(i)), key=i.__getitem__)
        out.append(class_index+1)
        print str((class_index+1-yTest[class_index+1])/yTest[class_index+1])
    df=pd.DataFrame(out,columns=['predict'])
    df['real']=dataTest['target']
    coun = 0
    for i,row in df.iterrows():
        if  row[0]== row[1]:
            coun+=1
    print coun
    print "df['real']", df['real'],type(df['real'][0])
    print "df['predict']",df['predict'],type(df['predict'][0])
    print df

    v=Validator()
    #v.MSE(out,dataTest['target'])
    print "out",out
    print "dataTest['target']",dataTest['target']

    #CrossValidator(trainer,DS,n_folds=5,max_epochs=20)

'''
#使用pandas作图  
df=pd.DataFrame(out,columns=['predict'])
df['real']=dataTest['target']
print "ssssss"
df1=df.sort_values(by='real')
df1.index=range(df.shape[0])
print "df",df
df.plot()
plt.show()'''
#print('true number is: ' + str(yTest[c]),
 #   'prediction number is:' + str(prediction),
 #     'error:' + str((prediction-yTest[c])/yTest[c]))'''
