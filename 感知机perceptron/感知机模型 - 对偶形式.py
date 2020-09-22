#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
这里训练集数量仅使用了1000条，测试集使用了200条
不仅因为Gram矩阵需要大量内存
如果数据量太大了可能出现memory error
而且因为这个算法实在是太太太太慢了！！！
'''

'''
感知机模型 - 对偶形式
-------------------------------------
数据集：Mnist
训练集数量：1000
测试集数量：200
迭代次数：5
---------------------------------
训练结果：
总时间：75s
训练时间：71s
正确率：0.77
'''


import numpy as np
import time
import pandas as pd
import random
import matplotlib.pyplot as plt


# In[ ]:


'''pandas方法读取训练数据'''
def LoadData(file, sampleN):
    
    # 开始记录读取数据时间
    start = time.time()
    print('start loading')

    # 读取训练集数据
    Alldata = pd.read_csv(file)

    # 将dataframe中的data转换成矩阵
    data = np.mat(list(Alldata.iloc[:,1:].values)[0:sampleN])
    
    # 准备空的List存储label
    label = []
    
    # 因为数据集存在0-9一共10种分类，为了符合二分类的感知机模型，我们将>=5的分为类1，将<5的分为类-1。
    temp = Alldata['label']
    for num in temp:
        if num >= 5:
            label.append(1)
        else:
            label.append(-1)
    
    # 将label转换成矩阵
    label = np.mat(label[0:sampleN]).T
    
    # 结束记录读取数据时间
    print('end loading')
    end = time.time()
    print('loading time: ' , end - start)
    
    # 返回数据的特征部分和标记部分
    return data, label


# In[ ]:


'''计算Gram矩阵'''
def Gram(train_data):
    
    print('start calculating Gram Matrix')
    start = time.time()
    
    # 使用numpy的dot函数计算Gram_Matrix，并将矩阵转换为array格式
    Gram_Matrix = np.array(np.dot(train_data, train_data.T))
    
    print('end calculating')
    end = time.time()
    print('Gram Matrix calculating time: ', end - start)
    
    return Gram_Matrix


# In[ ]:


'''训练部分 - 感知机对偶形式'''
def Perceptron(train_data, train_label, Gram_Matrix, itertime):
    
    # 开始记录训练模型时间
    start = time.time()
    print('start training')
    
    # 获得实例数和特征数
    sampleNum, featureNum = np.shape(train_data)
    
    # 使用特征数为w赋予初值，b也赋予初值，并且给一个学习率
    alpha = np.zeros(sampleNum)
    b = 0
    
    # 迭代开始，t代表迭代数
    for t in range(itertime):
        
        print(t)
        
        # 遍历每一个样本点
        for i in range(sampleNum):
            
            # 误分类判断公式中的求和项
            summary = 0
            
            # 计算求和项
            for j in range(sampleNum):
                
                yj = train_label[j]
                summary += alpha[j] * yj * Gram_Matrix[i][j]
            
            yi = train_label[i]
            
            # 判断是否为误分类点
            if yi * (summary + b) <= 0:
                
                # 如果是，对alpha和b进行迭代
                alpha[i] = alpha[i] + 1
                b = b + yi
    
    # 计算最优w
    w = sum((np.multiply(alpha, np.array(train_label).reshape(sampleNum)) * np.array(train_data).T).T)
    
    # 结束记录训练模型时间
    print('end training')
    end = time.time()
    print('training time: ', end - start)
    
    # 返回参数w和b，得到模型
    return w, b


# In[ ]:


'''测试部分'''
def classifier(test_data, test_label, w, b):
    
    # 开始记录测试模型时间
    start = time.time()
    print('start testing')
    
    # 得到测试集的实例数量
    testNum = len(test_label)
    
    # 误分类的实例的数量
    errorCnt = 0
    
    # 开始分类
    for n in range(testNum):
        
        # 获得测试集中的实例特征和标记
        xi = test_data[n]
        yi = test_label[n]
        
        # 如果误分类，errorCnt+1
        if yi * (w * xi.T + b) < 0:
            errorCnt += 1
    
    # 正确率 = 1 - （错误数 / 总实录数）
    Accurate = 1 - (errorCnt / testNum)
    
    # 结束记录测试时间
    print('end testing')
    print('Accurate: ', Accurate)
    end = time.time()
    
    # 返回正确率
    return Accurate


# In[ ]:


'''测试整个程序'''
if __name__ ==  '__main__':
    
    # 输入训练集
    train_data, train_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_train.csv',1000)
    
    # 输入测试集
    test_data, test_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_test.csv',200)
    
    # 计算Gram矩阵
    Gram_Matrix = Gram(train_data)
    
    # 训练模型
    itertime = 5
    w, b = Perceptron(train_data, train_label, Gram_Matrix, itertime)
    
    # 测试模型
    accurate = classifier(test_data, test_label, w, b)

