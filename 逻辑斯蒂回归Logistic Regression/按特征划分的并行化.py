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
    
    w = p[0]
    x = p[1]
    return -w * x


def Logistic(data, label, itertime):
    FeatureNum = np.shape(data)[1]
    w = np.zeros(FeatureNum)
    SampleNum = len(data)
    data = np.array(data)
    label = np.array(label)
    
    for i in range(itertime):
        print(i)
        
        # 获得分类错误的样本
        error_list = []
        for j in range(data.shape[0]):
            if predict(w, data[j]) != label[j]:
                error_list.append(j)
        error_data = data[j]
        error_label = label[j]
        
        # 多进程
        param_list = []
        for j in range(FeatureNum):
            param_list.append((w[j], data[:,j]))
            
        pool = mp.Pool(24)
        result = pool.map(single_job, param_list)
        pool.start()
        pool.join()
        
        # 计算梯度
        matrix = np.vstack(result).T
        exp_wx = np.exp(np.apply_along_axis(sum, 1, c))
        gradient = np.zeros(FeatureNum)
        for j in range(error_data.shape[0]):
            gradient += error_label[j] * error_data[j] - error_data[j] / (exp_wx[j] + 1)
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
