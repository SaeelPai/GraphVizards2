from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import keras
import csv
from sklearn.model_selection import train_test_split
# neural network with keras 

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
import pickle


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

train_pos_u = np.load('dummy_data/train_pos_u.npy')
train_neg_u = np.load('dummy_data/train_neg_u.npy')
test_pos_u = np.load('dummy_data/test_pos_u.npy')
test_neg_u = np.load('dummy_data/test_neg_u.npy')

train_pos_v = np.load('dummy_data/train_pos_v.npy')
train_neg_v = np.load('dummy_data/train_neg_v.npy')
test_pos_v = np.load('dummy_data/test_pos_v.npy')
test_pos_v=test_pos_v.squeeze()
test_neg_v = np.load('dummy_data/test_neg_v.npy')

node_num2str=pd.read_csv('dummy_data/node_dict.csv')
dict_num2str=dict(zip(node_num2str.int_names, node_num2str.nodes))


sdne_file=open("dummy_data/embed_trainonly_2kepochs.pickle","rb")
sdne_embed=pickle.load(sdne_file)
sdne_file.close()

dim_emb=list(sdne_embed.values())[0].shape[0]


def get_samples(pos_u,pos_v,sdne_embed,dict_num2str):
        
    samples=np.zeros((pos_u.shape[0],dim_emb))
    i=0
    for x in range(pos_u.shape[0]):
        try:
            samples[i] = np.multiply(sdne_embed[str(int(pos_u[x]))],sdne_embed[str(int(pos_v[x]))])
            i+=1
            
        except:
            pass
            
            print("One of the nodes in connection dont have embeddings")

    return samples

train_input_pos = get_samples(train_pos_u,train_pos_v,sdne_embed,dict_num2str)
train_pos_y = np.ones((train_pos_u.shape[0],1))

train_input_neg = get_samples(train_neg_u,train_neg_v,sdne_embed,dict_num2str)
train_neg_y = np.zeros((train_neg_u.shape[0],1))

test_input_pos = get_samples(test_pos_u,test_pos_v,sdne_embed,dict_num2str)
test_pos_y = np.ones((test_pos_u.shape[0],1))

test_input_neg = get_samples(test_neg_u,test_neg_v,sdne_embed,dict_num2str)
test_neg_y = np.zeros((test_neg_u.shape[0],1))

X_final = np.concatenate((test_input_pos,test_input_neg))
y_final=np.concatenate((test_pos_y,test_neg_y))


X = np.concatenate((train_input_pos,train_input_neg))
y=np.concatenate((train_pos_y,train_neg_y))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15, random_state=3)

model = keras.Sequential(
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

opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#model_checkpoint=keras.callbacks.ModelCheckpoint(model_name,monitor='val_loss',verbose=1,save_best_only=True,min_delta=0.001)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,mode='min',min_delta=0.001) # saves only the best ones
red_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5, verbose=1, mode='auto',min_lr=1e-7,min_delta=0.001)            


history=model.fit(X_train, y_train,validation_split=0.2 ,epochs=50, batch_size=500,callbacks=[early_stopping,red_lr])

y_pred=model.predict_classes(X_test)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)

y_pred_final=model.predict_classes(X_final)
y_prob_final = model.predict_proba(X_final)
_, test_acc_final = model.evaluate(X_final, y_final, verbose=0)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_final, y_pred_final)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_final, y_pred_final)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_final, y_pred_final)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_final, y_pred_final)
print('F1 score: %f' % f1)

# ROC AUC
auc = roc_auc_score(y_final, y_prob_final)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_final, y_pred_final)
print(matrix)

fpr, tpr, thresholds = roc_curve(y_final, y_prob_final)
plt.scatter(fpr, tpr, marker='.', label='SDNE embeddings')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

