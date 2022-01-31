from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_data():
    train = pd.read_csv('D:/mnist/mnist_train.csv')
    test = pd.read_csv('D:/mnist/mnist_test.csv')

    columns = list(set(train.columns) - {'label'})
    train[columns] = train[columns] / 255
    test[columns] = test[columns] / 255
    return train, test

def train_models(train):
    models = {}
    columns = set(train.columns) - {'label'}
    
    # train models
    for i in range(10):
        
        # set the label == i to 1 and the others to 0
        train_data = train.copy()
        cond = train_data['label'] == i
        train_data.loc[cond, 'label'] = 1
        train_data.loc[~cond, 'label'] = 0

        train_data_i = train_data[columns]  # train_data
        train_label_i = train_data[['label']]  # train_label

        lr = LogisticRegression(random_state=np.random.randint(0, 100))  # train model
        lr.fit(train_data_i,train_label_i)
        models[i] = lr  # save models
            
    return models

def predict_test(models, test):
    result = []
    columns = set(train.columns) - {'label'}
    
    # predict test data
    for i in range(10):
        predict = models[i].predict_proba(test[columns])  # predict
        result.append(predict[:,1])  # only save the result of class 1 is enough
            
    return result

def check_result(result):
    
    result = np.vstack(result)
    cnt = 0
    
    # check each test data
    for i in range(result.shape[1]):
        p = result[:,i]
        p = np.where(p == p.max())[0][0]  # select the predict class with the highest probability
        if p == test['label'].iloc[i]:
            cnt += 1

    acc = cnt / len(test)  # calculate accuracy
    return acc

train, test = load_data()
models = train_models(train)
result = predict_test(models, test)
acc = check_result(result)
print(acc)
# 0.9201
