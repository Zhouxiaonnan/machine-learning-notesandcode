#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
朴素贝叶斯 - 贝叶斯估计
-------------------------------------
数据集：Mnist
训练集数量：60000
测试集数量：10000
---------------------------------
训练结果：
时间：136s
正确率：0.83
'''


import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import collections as cc


# In[ ]:


'''
pandas方法读取训练数据
file：数据地址
'''

def LoadData(file):
    
    print('loading data')
    
    # 读取训练集数据
    Alldata = pd.read_csv(file)

    print('end loading data')
    
    # 返回数据的特征部分和标记部分
    return Alldata


# In[ ]:


'''
计算先验概率
train：训练集标记和特征一起
'''
def Prior(train):
    
    print('Calculating Prior Probability')
    start = time.time()
    
    # 类别和类别的数量
    train_label = train['label']
    Class = np.sort(np.unique(train['label']))
    ClassNum = len(Class)
    
    # 训练集实例的总数
    TrainNum = len(train_label)
    
    # 使用collections.Counter方法对train_label中不同的类进行计数
    Counter = cc.Counter(train_label)
    
    # 准备一个列表，存储每个类别的先验概率
    Prior_P = [0] * ClassNum
    
    # 准备一个列表，存储每个类别的训练集中实例的数量
    C_Number = [0] * ClassNum
    
    # 遍历每个Class
    for i in range(ClassNum):
        
        # 计算每一个类别的数量，并保存在C_Number中的相应位置
        C_Number[i] = Counter[i]
        
        # 计算每一个类别的先验概率，并保存在Prior_P中的相应位置
        # 因为使用贝叶斯估计的方法，所以分子+1，分母+类别的数量
        Prior_P[i] = (Counter[i] + 1) / (TrainNum + ClassNum)
        
    print('End Calculating')
    end = time.time()
    print('prior calculating time：', end - start)
    
    # 返回每一个类的先验概率，每个类别训练集中实例的数量，以及类
    return Prior_P, C_Number, Class


# In[ ]:


'''
统计训练集中的特征可取值出现的次数
Class：类别
train：训练集
'''
def FeatureStatistic(Class, train):
    
    print('start feature statistic stage')
    start = time.time()
    
    # 准备一个空list，装每个类所对应的特征Array
    Classified_train_data = []

    # 遍历每个类别
    for C in Class:

        # 找到对应于类别C的训练集数据
        train_C = train[train['label'] == C].iloc[:,1:].values

        # 将该数据转换成Matrix，转置，再转换回Array
        # 此时每行都代表在类别C的情况下，该行对应的特征出现的值
        SampleMatrix = np.mat(train_C)
        FeatureMatrix = SampleMatrix.T
        FeatureArr = np.array(FeatureMatrix)

        # 准备一个空list用来存储某个特征中特定值出现的频率
        FeatureCounter = []

        # 遍历一个类别中的所有特征
        for f in FeatureArr:

            # 统计一个类别中的一个特征中各个值出现的次数
            FeatureCounter.append(cc.Counter(f))

        # 将featureCounter存储起来
        # 此时Classified_train_data的结构是
        # 第一层的index代表0-9一共10个类别
        # 第二层代表在一个特定类别内，每一个特征
        # 第三层代表在一个特定类别内的一个特定特征内，对该特征各个值出现的次数的统计
        Classified_train_data.append(FeatureCounter)
        
    print('end feature statistic stage')
    end = time.time()
    print('feature statistic stage time: ', end - start)
    
    # 返回统计结果
    return Classified_train_data


# In[ ]:


'''
计算条件概率，新输入实例进行分类
Class：类
Classified_train_data：对每个类别中的每个特征的可取值出现的次数的统计
C_Number：每个类别训练集中实例的数量
featureNum：特征的数量
test_sample：新输入实例
'''
def Classifier(Class, Classified_train_data, C_Number, featureNum, test_sample):

    # 准备一个空list用来存放该新输入实例点的条件概率
    C_P = []

    # 对于每一个类别
    for C in Class:

        # 因为在python中当一个值太小可能直接为0，即下溢出问题
        # 而在贝叶斯方法中，我们需要对大量的概率进行连乘
        # 因为概率<1，所以大量概率连乘很可能太小，导致其值为0
        # 因此我们将贝叶斯中的条件概率相乘log一下
        # 转变成条件概率相加，对于对比概率来说没有影响
        P = 0

        # 对于每一个特征的输入值
        for i in range(len(test_sample)):

            # 计算条件概率
            # 使用了贝叶斯估计的方法，所以分子+1，分母+256，因为灰度值是0-255，一共256个
            # Classified_train_data[C][i]表示C类的第i个特征
            # test_sample[i]是test_sample的第i个特征的取值
            # 对Counter的说明：
            # 比如a = [1,1,2,3]，那么b=cc.Counter(a)的输出为Counter({1: 2, 2: 1, 3: 1})
            # 代表1出现2次，2和3出现1次
            # 如果需要1出现的次数，只需要b[1]即可
            # 如果需要3出现的次数，只需要b[3]即可
            # log是因为在python中如果数字太小会直接变成0（下溢出）
            # 因此要log一下，把相乘变成相加，规避这个问题
            P = P + np.log((Classified_train_data[C][i][test_sample[i]] + 1) / (C_Number[C] + 256))

        # 将一个类的条件概率存储在C_P
        C_P.append(P)
    
    # 分类
    test_sample_C = C_P.index(max(C_P))
    
    # 输出对新输入实例的分类结果
    return test_sample_C


# In[ ]:


'''运行'''
if __name__ == "__main__":

    # 输入训练集
    train = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_train.csv')
    
    # 输入测试集
    test = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_test.csv')
    
    # 得到先验概率和类别
    Prior_P, C_Number, Class = Prior(train)
    
    # 统计训练集中的特征可取值出现的次数
    Classified_train_data = FeatureStatistic(Class, train)
    
    # 得到特征数量
    featureNum = len(train.columns) - 1

    # 对测试集中的每一个实例进行分类
    
    # 首先将测试集中data和label分开
    test_data = test.iloc[:,1:]
    test_label = test['label']
    
    #得到测试集的实例数量
    testN = len(test_label)
    
    # 初始化errorCnt
    errorCnt = 0
    
    print('start classifying')
    start = time.time()
    
    # 遍历每一个测试集的实例
    for i in range(testN):
        
        # 对实例进行分类
        test_sample_C = Classifier(Class, Classified_train_data, C_Number, featureNum, test_data.iloc[i].values)
        
        # 检查分类是否正确
        if test_label[i] != test_sample_C:
            
            # 若不正确，errorCnt+1
            errorCnt += 1
    
    print('end classifying')
    end = time.time()
    print('classifying time: ', end - start)
    
    # 计算正确率
    Accurate = 1 - (errorCnt / testN)
    
    print('正确率：', Accurate)
    

