#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
感知机模型 - 梯度下降法
-------------------------------------
数据集：Mnist
训练集数量：60000
测试集数量：10000
迭代次数：30
---------------------------------
训练结果：
总时间：79s
训练时间：75s
正确率：0.76
'''

import numpy as np
import time
import pandas as pd
import random
import matplotlib.pyplot as plt


# In[ ]:


'''pandas方法读取训练数据'''
def LoadData(file):
    
    # 开始记录读取数据时间
    start = time.time()
    print('start loading')

    # 读取训练集数据
    Alldata = pd.read_csv(file)

    # 将dataframe中的data转换成矩阵
    data = np.mat(list(Alldata.iloc[:,1:].values))
    
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
    label = np.mat(label).T
    
    # 结束记录读取数据时间
    print('end loading')
    end = time.time()
    print('loading time: ' , end - start)
    
    # 返回数据的特征部分和标记部分
    return data, label


# In[ ]:


'''开始训练 - 梯度下降法'''
def Perceptron(train_data, train_label, itertime):
    
    # 开始记录训练模型时间
    start = time.time()
    print('start training')
    
    # 获得实例数和特征数
    sampleNum, featureNum = np.shape(train_data)
    
    # 使用特征数为w赋予初值，b也赋予初值，并且给一个学习率
    w = np.zeros(featureNum)
    b = 0
    h = 0.0001
    
    # 迭代开始，t代表迭代数
    for t in range(itertime):
        
        # 梯度下降法中每次迭代将遍历所有的误分类点，并加总所有的梯度，因此首先初始化为0
        # 也可以准备两个list用来接收误分类点的梯度，之后再sum一下
        wG = 0
        bG = 0
        
        # 因为要遍历所有的点，判断是否为误分类，所以采用for循环
        for i in range(sampleNum):

            # 获取该样本点的特征和标记
            xi = train_data[i]
            yi = train_label[i]
            
            # 查看该样本点是否为误分类点
            if -1 * yi * (w * xi.T + b) >= 0:
                
                # 如果是，算出w和b对L的偏导数，即梯度
                wG = -1 * yi * xi
                bG = -1 * yi
                
                # 梯度相加
                wGs += wG
                bGs += bG

        # 循环结束后，对w和b进行迭代
        w = w + h * -1 * wGs
        b = b + h * -1 * bGs
                
    # 结束记录训练模型时间
    print('end training')
    end = time.time()
    print('training time: ', end - start)
    
    # 返回参数w和b，得到模型
    return w, b


# In[ ]:


'''开始测试'''
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
    print('training time: ', end - start)
    
    # 返回正确率
    return Accurate


# In[ ]:


'''测试整个程序'''
if __name__ ==  '__main__':
    
    # 输入训练集
    train_data, train_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_train.csv')
    
    # 训练模型
    itertime = 30
    w, b = Perceptron(train_data, train_label, itertime)
    
    # 输入测试集
    test_data, test_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_test.csv')
    
    # 测试模型
    accurate = classifier(test_data, test_label, w, b)


# In[ ]:


'''测试整个程序'''
if __name__ ==  '__main__':
    
    # 输入训练集
    train_data, train_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_train.csv')
    
    # 输入测试集
    test_data, test_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_test.csv')
    
    # 测试不同的itertime
    itertime_group = np.linspace(5,100,20)
    
    accurate_group = []
    
    # 对每一个itertime进行训练
    for itertime in itertime_group:
        itertime = int(itertime)
        w, b = Perceptron(train_data, train_label, itertime)
        
    # 测试模型
        accurate = classifier(test_data, test_label, w, b)
        accurate_group.append(accurate)
    
    plt.plot(itertime_group, accurate_group)
    plt.xlabel('itertime')
    plt.ylabel('accurate')

