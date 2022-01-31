import numpy as np
import time
from itertools import islice
import multiprocessing as mp

def LoadData(file):
    fr = open(file, 'r')
    
    data = []
    label = []
    for line in islice(fr,1,None):
        splited = line.strip().split(',')
        data.append([int(num)/255 for num in splited[1:]])
        
        a = int(splited[0])
        if a > 4:
            label.append(1)
        else:
            label.append(0)
    
    SampleNum = len(data)
    for i in range(SampleNum):
        data[i].append(1)
    return data, label


def predict(w, x):
    
    exp_wx = np.exp(-np.dot(w, x))
    P = 1 / (1 + exp_wx)
    if P > 0.5:
        return 1
    return 0


def single_job(p):
    
    # 计算每个样本的梯度
    w = p[0]
    x = p[1]
    y = p[2]
    if predict(w, x) != y:
        exp_wx = np.exp(-np.dot(w,x))
        gradient = 0.001 * (x * y - x / (1 + exp_wxi))
        return 1, gradient
    return 0, gradient


def Logistic(data, label, itertime):
    FeatureNum = np.shape(data)[1]
    w = np.zeros(FeatureNum)
    
    SampleNum = len(data)
    data = np.array(data)
    
    for i in range(itertime):
        param_list = []
        for j in range(data.shape[0]):
            param_list.append((w, data[j], label[j]))
            
        # 多进程
        pool = mp.Pool(24)
        result = pool.map(single_job, param_list)
        pool.start()
        pool.join()
        
        result = [x[1] for x in result if x[0] == 1]
        gradient = np.zeros(FeatureNum)
        for x in result:
            gradient += x
        gradient = gradient / len(result)
        w += gradient
    return w


def Classifier(data, label, w):
    SampleNum = len(data)
    errorCnt = 0
    for i in range(SampleNum):
        result = predict(w, data[i])
        if result != label[i]:
            errorCnt += 1
    
    Acc = 1 - errorCnt / SampleNum
    return Acc


def main():
    train_data, train_label = LoadData('D:/mnist/mnist_train.csv')
    test_data, test_label = LoadData('D:/mnist/mnist_test.csv')

    w = Logistic(train_data, train_label, 400)

    accurate = Classifier(test_data, test_label, w)
    print('Accurate: {}'.format(accurate))
    print('*' * 50)

if __name__ == '__main__':
    main()
