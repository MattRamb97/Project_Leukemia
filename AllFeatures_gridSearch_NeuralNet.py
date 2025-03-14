from keras.layers import Dropout, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from pathlib import Path
import numpy as np
import pandas as pd

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
valid_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_VALIDATION-AllFeats_1612-Features_10000-images.csv'), index_col=0)
print('Done Read Validation Dataframe!')

print('Preparing Data...')

nrange = valid_df.shape[1]
for i in range(nrange):
    valid_df[valid_df.columns[i]] = clean_Dirt_Data(valid_df[valid_df.columns[i]])
    train_df[train_df.columns[i]] = clean_Dirt_Data(train_df[train_df.columns[i]])

x_train, y_train = prepareData(train_df)
x_valid, y_valid = prepareData(valid_df)
print('Done Read Train and Validation data!')
    
def criarRede(optimizer, loos, kernel_initializer, activation,
              neurons, hidden, dropout):
    classifier = Sequential()
    classifier.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer, input_shape = (x_train.shape[1],)))
    classifier.add(Dropout(dropout))
    
    for i in range(hidden):
        classifier.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer))
        classifier.add(Dropout(dropout))
    
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = loos,
                      metrics = ['binary_accuracy'])
    return classifier


classifier = KerasClassifier(build_fn = criarRede)
parameters = {
              'batch_size': [250, 750, 1000, 1500],
              'optimizer': ['adamax', 'adam', 'sgd'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'prelu', 'sigmoid', 'softmax'],
              'hidden' : [1, 2, 3, 4],
              'epochs': [50],
              'loos': ['binary_crossentropy'],
              'dropout' : [0.1, 0.25, 0.3, 0.5],
              'neurons': [1024, 1536, 2048, 2560],
              }
grid_search = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'f1_micro')
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_