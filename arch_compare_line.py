# Loading prerequisites

from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import keras
import csv
from sklearn.model_selection import train_test_split
# neural network with keras 

from numpy import loadtxt
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
import pickle
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#%% Loading and conditioning data - as in sdne_ANN - ankit
# for now, only using sdne. Could add other embeddings later

train_data = np.load('dummy_data/train_data.npy')
test_data = np.load('dummy_data/test_data.npy')
val_data = np.load('dummy_data/val_data.npy')


#node_num2str=pd.read_csv('dummy_data/node_dict.csv')
#dict_num2str=dict(zip(node_num2str.int_names, node_num2str.nodes))

sdne_file=open("dummy_data/embed_line.pickle","rb")
sdne_embed=pickle.load(sdne_file)
sdne_file.close()

dim_emb=list(sdne_embed.values())[0].shape[0]

def get_samples(X,sdne_embed):
        
    samples=np.zeros((X.shape[0],dim_emb+1))
    i=0
    for x in range(X.shape[0]):
        try:
            temp = np.multiply(sdne_embed[str(int(X[x,0]))],sdne_embed[str(int(X[x,1]))])
            samples[i] = np.append(temp,X[x,2])
            i+=1
            
        except:
            pass
            
            print("One of the nodes in connection dont have embeddings")       
    
    samples = samples[~np.all(samples == 0, axis=1)]
    return samples

train = get_samples(train_data,sdne_embed)
X_train = train[:,:-1]
y_train = train[:,-1]

test = get_samples(test_data,sdne_embed)
X_test = test[:,:-1]
y_test = test[:,-1]

val = get_samples(val_data,sdne_embed)
X_final = val[:,:-1]
y_final = val[:,-1]

"""
plt.figure(figsize=(8,6))
width = 0.35       # the width of the bars

pos_links = {'Train': np.where(y_train==1)[0].shape[0], 'Test': np.where(y_test==1)[0].shape[0], 'Val': np.where(y_final==1)[0].shape[0]}

rects1 = plt.bar(pos_links.keys(), pos_links.values(), -width, align='edge',
                color='r', label='Positive links')

neg_links = {'Train': np.where(y_train==0)[0].shape[0], 'Test': np.where(y_test==0)[0].shape[0], 'Val': np.where(y_final==0)[0].shape[0]}


rects2 = plt.bar(neg_links.keys(), neg_links.values(), +width, align='edge',
                color='b', label='Negative links')



# add some text for labels, title and axes ticks
plt.xlabel('Datasets')
plt.ylabel('Frequency')
plt.title('Number of positve and negative samples in train, test and val datasets')
plt.legend()

def autolabel(rects):
    
    #Attach a text label above each bar displaying its height
    
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.ylim([0,200000])
plt.show()
"""
#%% Model 1 - ANN
model_ANN = Sequential(
    [
        Dense(32, input_dim=128, activation="relu", name="layer1"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(8, activation="relu", name="layer2"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation="sigmoid", name="output")
    ]
)

opt_ANN = keras.optimizers.Adam(lr=0.005)
model_ANN.compile(loss='binary_crossentropy', optimizer=opt_ANN, metrics=['accuracy'])

model_name="embed_line.hdf5"
#model_checkpoint=keras.callbacks.ModelCheckpoint(model_name,monitor='val_loss',verbose=1,save_best_only=True,min_delta=0.001)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=25,mode='min',min_delta=0.001) # saves only the best ones
red_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.25,patience=10, verbose=1, mode='auto',min_lr=1e-7,min_delta=0.001)            
model_checkpoint=keras.callbacks.ModelCheckpoint(model_name,monitor='val_loss',verbose=1,save_best_only=True)


history_ANN=model_ANN.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=75, batch_size=500,callbacks=[early_stopping,red_lr,model_checkpoint])

model_ANN=load_model(model_name)
y_pred = model_ANN.predict_classes(X_test)
_, test_acc = model_ANN.evaluate(X_test, y_test, verbose=0)

y_pred_final_ANN=model_ANN.predict_classes(X_final)
y_prob_final_ANN = model_ANN.predict_proba(X_final)
_, test_acc_final = model_ANN.evaluate(X_final, y_final, verbose=0)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_final, y_pred_final_ANN)
print('Accuracy ANN: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_final, y_pred_final_ANN)
print('Precision ANN: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_final, y_pred_final_ANN)
print('Recall ANN: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_final, y_pred_final_ANN)
print('F1 score ANN: %f' % f1)


#%% Model 2 - SVM
#Import svm model
from sklearn import svm

#Create a svm Classifier
#model_svm= svm.SVC(kernel='linear') # Linear Kernel
#model_svm= svm.SVC(kernel='rbf') # Radial Basis Function Kernel
model_svm= svm.SVC(kernel='poly') # polynomial Kernel

#Train the model using the training sets
model_svm.fit(X_train[:10000], y_train[:10000])

#Predict the response for test dataset
#y_pred_svm = model_svm.predict(X_test)
#accuracy_test = accuracy_score(y_test, y_pred_svm)
#print('Accuracy svm on test: %f' % accuracy_test)

y_pred_final_svm = model_svm.predict(X_final)
#y_prob_final_svm = model_svm.predict_proba(X_final)
#_, test_acc_final = model_svm.evaluate(X_final, y_final, verbose=0)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_final, y_pred_final_svm)
print('Accuracy ANN: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_final, y_pred_final_svm)
print('Precision ANN: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_final, y_pred_final_svm)
print('Recall ANN: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_final, y_pred_final_svm)
print('F1 score ANN: %f' % f1)

#%% Logistic Regression 

from sklearn.linear_model import LogisticRegression

model_LR = LogisticRegression(random_state=0, solver= 'lbfgs', max_iter=500)
model_LR.fit(X_train, y_train)

#y_pred = model_LR.predict_classes(X_test)
#_, test_acc = model_LR.evaluate(X_test, y_test, verbose=0)

y_pred_final_LR=model_LR.predict(X_final)
y_prob_final_LR = model_LR.predict_proba(X_final)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_final, y_pred_final_LR)
print('Accuracy ANN: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_final, y_pred_final_LR)
print('Precision ANN: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_final, y_pred_final_LR)
print('Recall ANN: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_final, y_pred_final_LR)
print('F1 score ANN: %f' % f1)


#%% Gaussians Naive Bayes Classifiers

from sklearn.naive_bayes import GaussianNB

model_GNB = GaussianNB()
model_GNB.fit(X_train, y_train)

y_pred_final_GNB = model_GNB.predict(X_final)
y_prob_final_GNB = model_GNB.predict_proba(X_final)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_final, y_pred_final_GNB)
print('Accuracy ANN: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_final, y_pred_final_GNB)
print('Precision ANN: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_final, y_pred_final_GNB)
print('Recall ANN: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_final, y_pred_final_GNB)
print('F1 score ANN: %f' % f1)

#%%  SGD Classifier

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model_SGD = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
model_SGD.fit(X_train, y_train)

y_pred_final_SGD = model_SGD.predict(X_final)
#y_prob_final_SGD = model_SGD.predict_proba(X_final)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_final, y_pred_final_SGD)
print('Accuracy ANN: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_final, y_pred_final_SGD)
print('Precision ANN: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_final, y_pred_final_SGD)
print('Recall ANN: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_final, y_pred_final_SGD)
print('F1 score ANN: %f' % f1)

#%% Random Forests
# max depth = 2 is optimal. 1 gives 87, 3 gives 88 recall and accuracy
# n_estimators = 150 is optimal. 50 gices 88.9, 200 gives 89.08, 100 gives 89.11
#                           125 gives 89.11

from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier(n_estimators=150, max_depth=2, random_state=0)
model_RF.fit(X_train, y_train)

y_pred_final_RF = model_RF.predict(X_final)
y_prob_final_RF = model_RF.predict_proba(X_final)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_final, y_pred_final_RF)
print('Accuracy ANN: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_final, y_pred_final_RF)
print('Precision ANN: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_final, y_pred_final_RF)
print('Recall ANN: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_final, y_pred_final_RF)
print('F1 score ANN: %f' % f1)

#%% AdaBoost
# accuracy and recall around 86-87% after trying a few hyper parameters

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model_DTC = DecisionTreeClassifier(max_depth=2)
model_ABC = AdaBoostClassifier(base_estimator=model_DTC, n_estimators=50, random_state=0)
model_ABC.fit(X_train, y_train)

y_pred_final_ABC = model_ABC.predict(X_final)
y_prob_final_ABC = model_ABC.predict_proba(X_final)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_final, y_pred_final_ABC)
print('Accuracy ANN: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_final, y_pred_final_ABC)
print('Precision ANN: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_final, y_pred_final_ABC)
print('Recall ANN: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_final, y_pred_final_ABC)
print('F1 score ANN: %f' % f1)

#%% Bagging Classifier
from sklearn.ensemble import BaggingClassifier
model_BC = BaggingClassifier(base_estimator=model_DTC, n_estimators=10, random_state=0)
model_BC.fit(X_train, y_train)

y_pred_final_BC = model_BC.predict(X_final)
y_prob_final_BC = model_BC.predict_proba(X_final)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_final, y_pred_final_BC)
print('Accuracy ANN: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_final, y_pred_final_BC)
print('Precision ANN: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_final, y_pred_final_BC)
print('Recall ANN: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_final, y_pred_final_BC)
print('F1 score ANN: %f' % f1)
