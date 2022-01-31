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
    columns = list(set(train.columns) - {'label'})
    
    # train models
    for i in range(10):
        for j in range(i + 1, 10):

            cond = train['label'].isin([i, j])  # get train_data with labels in (i,j)
            train_data = train[cond]

            train_data_ij = train_data[columns]  # train_data
            train_label_ij = train_data[['label']]  # train_label

            lr = LogisticRegression(random_state=np.random.randint(0, 100))  # train model
            lr.fit(train_data_ij,train_label_ij)
            models['{}_{}'.format(i,j)] = lr  # save models
            
    return models

def predict_test(models, test):
    result = []
    
    # predict test data
    for i in range(10):
        for j in range(i + 1, 10):
            model_name = '{}_{}'.format(i, j)  # get model
            predict = models[model_name].predict_proba(test[columns])  # predict
            result.append(np.where(predict[:,0] > predict[:,1], i, j))  # save result
            
    return result

def check_result(result):
    
    result = np.vstack(result)
    cnt = 0
    
    # check each test data
    for i in range(result.shape[1]):
        p = Counter(result[:,i]).most_common(1)[0][0]  # get the predict result with the highest number of vote
        if p == test['label'].iloc[i]:
            cnt += 1

    acc = cnt / len(test)  # calculate accuracy
    return acc

train, test = load_data()
models = train_models(train)
result = predict_test(models, test)
acc = check_result(result)
print(acc)
# 0.9446
