import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
import timeit

start = timeit.default_timer()

def clean_Dirt_Data(x):
    ret = []
    for i in x:
        i = str(i)
        ret.append(float(i.replace('[', '').replace('.]', '').replace(']', '') ))
    
    return pd.DataFrame(ret)

def prepareData(data_df):
    #Prepare Validation data
    y = data_df['cellType(ALL=1, HEM=-1)'].values
    for i in range(len(y)):
        if y[i]==-1:
            y[i] = 0
        elif y[i]==1:
            y[i] = 1
    y = np.array(y)
    x = data_df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)
    for col in x.columns:
        x[col] = (x[col] - data_df[col].mean()) / data_df[col].std() #mean=0, std=1
    x = x.values
    return x, y

print('Reading Train Dataframe...')
train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1612-Features_40000-images.csv'), index_col=0)
print('Done Read Train Dataframe!')

print('Reading Validation Dataframe...')
valid_df = pd.read_csv(Path('feature-dataframes/PatLvDiv_TEST-AllFeats_1612-Features_1503-images.csv'), index_col=0)
print('Done Read Validation Dataframe!')

print('Preparing Data...')

nrange = valid_df.shape[1]
for i in range(nrange):
    valid_df[valid_df.columns[i]] = clean_Dirt_Data(valid_df[valid_df.columns[i]])
    train_df[train_df.columns[i]] = clean_Dirt_Data(train_df[train_df.columns[i]])

x_train, y_train = prepareData(train_df)
x_valid, y_valid = prepareData(valid_df)

print('Done Read Train and Validation data!')

classifier = LinearSVC()
classifier.fit(x_train, y_train)

prediction = classifier.predict(x_valid)

#Find the prediction
prediction = classifier.predict(x_valid)
prediction = (prediction > 0.5)

#Let's measure the accuracy of the network
precision = f1_score(y_valid, prediction)

final_prediction = pd.DataFrame(prediction)
final_prediction.to_csv('svm_previsoes.csv')

print('F1-Score: ', precision)
stop = timeit.default_timer()
print('Time: ', stop - start)  
