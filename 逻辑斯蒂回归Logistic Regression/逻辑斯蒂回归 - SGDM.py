#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
逻辑斯蒂回归 - SGDM
-------------------------------------
数据集：Mnist
训练集数量：60000
测试集数量：10000
---------------------------------
训练结果：
迭代次数：400
训练时间：171s
测试时间：1s
正确率：0.86
'''

import numpy as np
import time
from itertools import islice
import random


# In[ ]:


'''
readlines方法读取训练数据
--------
输入：
file：数据地址
--------
输出：
data：转换后的数据
label：转换后的标记
'''
def LoadData(file):
    
    # 打开数据文件
    fr = open(file, 'r')
    
    # 准备两个list存储data和label
    data = []
    label = []
    
    # 逐行读取数据，使用islice可以直接跳过第一行表头进行逐行读取
    for line in islice(fr,1,None):
        
        # 对每一行消去空格，并根据','进行分割
        splited = line.strip().split(',')
        
        # 逐行存储数据，此处对特征进行归一化
        data.append([int(num)/255 for num in splited[1:]])
        
        # 因为逻辑斯蒂回归是二分类问题，我们将大于4的类归为1，小于4的类归为0
        # 这里如果让等于0的数为1，其它为-1，正确率可以达到98%
        a = int(splited[0])
        if a > 4:
            label.append(1)
        else:
            label.append(0)
    
    # 采用书中的方法，我们将b作为一个元素放在w向量的最后一个
    # 同样也需要在每个特征向量后面加上一个1
    SampleNum = len(data)
    for i in range(SampleNum):
        data[i].append(1)
            
    # 返回数据的特征部分和标记部分
    return data, label


# In[ ]:


'''
预测
-------
输入：
w：目前的权值向量
x：一条样本的特征向量
-----------
输出：类别
'''
def predict(w, x):
    
    exp_wx = np.exp(np.dot(w, x))
    P = exp_wx / (1 + exp_wx)
    if P > 0.5:
        return 1
    return 0


# In[ ]:


'''
逻辑斯蒂回归 - SGDM
--------------
输入：
data：训练集数据
label：训练集标记
itertime：迭代次数
-------------
输出：
w：最优化的权值向量
'''
def Logistic(data, label, itertime):
    
    # 初始化w
    FeatureNum = np.shape(data)[1]
    w = np.zeros(FeatureNum)
    
    # 设定学习率
    h = 0.001
    
    # 样本数
    SampleNum = len(data)
    
    # 将data转换成ndarray格式，方便后续的计算
    data = np.array(data)
    
    # 初始化动量
    momentum = np.zeros(FeatureNum)
    
    # 设置超参数，用于设置动量中对前一次动量和速度的依赖程度
    beita = 0.9
    
    # 迭代
    for i in range(itertime):
        
        # 初始化梯度
        v = np.zeros(FeatureNum)
        
        # 遍历每一个样本点
        for j in range(SampleNum):
            
            # 得到该样本的特征向量和标记
            xi = data[j]
            yi = label[j]
            
            # 如果分类错误，更新w_i
            if predict(w, xi) != yi:
                
                # 计算梯度
                exp_wxi = np.exp(np.dot(w,xi))
                v += xi * yi - (xi * exp_wxi) / (1 + exp_wxi)
                
        # 记录动量，其中beita是超参数
        momentum = beita * momentum + (1 - beita) * v

        # 更新w
        w += h * momentum

    return w


# In[ ]:


'''
模型测试
------------
输入：
data：测试集数据
label：测试集标记
w：最优化的权值向量
-----------
输出：
Acc：正确率
'''

def Classifier(data, label, w):
    
    # 样本数
    SampleNum = len(data)
    
    # 初始化错误分类的数量
    errorCnt = 0
    
    # 遍历每一个样本
    for i in range(SampleNum):
        
        # 对该样本的分类
        result = predict(w, data[i])
        
        # 判断是否分类正确
        if result != label[i]:
            
            # 分类错误，errorCnt+1
            errorCnt += 1
    
    # 计算正确率
    Acc = 1 - errorCnt / SampleNum
    
    return Acc


# In[ ]:


'''测试模型'''
if __name__ == "__main__":

    print('start loading')
    # 输入训练集
    train_data, train_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_train.csv')

    # 输入测试集
    test_data, test_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_test.csv')
    print('end loading')
    
    # 最优化参数
    print('start training')
    start = time.time()
    w = Logistic(train_data, train_label, 400)
    print('end training')
    end = time.time()
    print('training time: ', end - start)

    # 进行分类
    start = time.time()
    print('start testing')
    accurate = Classifier(test_data, test_label, w)
    print('end testing')
    end = time.time()
    print('testing time: ', end - start)
    print(accurate)


# In[ ]:




