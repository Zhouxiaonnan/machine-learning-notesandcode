#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
提升树
-------------------------------------
数据集：Mnist
训练集数量：60000
测试集数量：10000
---------------------------------
训练结果：
基本分类器数量：20
训练时间：3h
测试时间：0.75s
正确率：0.79
'''

import numpy as np
import time
from itertools import islice


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
        
        # 逐行存储数据，此处将特征值0-255分为两个部分
        # >=128的为1，<128的为0
        data.append([int(int(num)) >= 128 for num in splited[1:]])
        
        # 因为提升分类树是二分类问题，我们将大于4的类归为1，小于4的类归为-1
        a = int(splited[0])
        if a > 4:
            label.append(1)
        else:
            label.append(-1)
    
    data = np.array(data)
    label = np.array(label)
            
    # 返回数据的特征部分和标记部分
    return data, label


# In[ ]:


'''
对每一种情况计算误差率
-----------
输入：
feature：所有样本的特定特征的值
divide：切分点
rule：判断类别的规则
train_label：训练集标记
WeightArr：当前的样本权重向量
------------
输出：
PredictArr：在该情况下的所有样本的预测值
error：被错误分类的样本对应权值之和
'''
def Estimate(feature, divide, rule, train_label, WeightArr):
    
    # 用于之后预测在该情况下每个样本属于哪个类
    if rule == 'H1L-1':
        H = 1
        L = -1
    else:
        H = -1
        L = 1
    
    # 初始化分类错误的样本的权值之和
    error = 0
    
    # 准备一个列表记录该情况下的预测值
    PredictArr = []
    
    # 得到样本容量
    SampleNum = len(feature)
    
    # 遍历每个样本，判断样本的该特征值是否大于divide
    # 并预测相应的类
    for i in range(SampleNum):
        if feature[i] > divide:
            predict = H
        else: 
            predict = L
        
        PredictArr.append(predict)
        
        # 如果预测值与标记不符
        # 则error + 该分类错误样本所对应的权值
        if predict != train_label[i]:
            error += WeightArr[i]
        
    return PredictArr, error


# In[ ]:


'''
创建单一的基本分类器
---------
输入：
train_data：训练集数据
train_label：训练集标记
WeightArr：当前训练集样本的权值向量
--------
输出：
SingleTreeDict：一个存储了最佳基本分类器参数的字典
'''
def CreateSingleBoostingTree(train_data, train_label, WeightArr):
    
    # 初始化被分类错误的样本数
    # 我们需要让每一个基本分类器的该误差最小
    error = 1
    
    # m为样本容量，n为特征数量
    m, n = np.shape(train_data)
    
    # 准备一个字典用来存储基本分类器的参数
    SingleTreeDict = {}
    
    # 遍历每一个特征，找到用来分类的最佳特征
    for i in range(n):

        # 得到所有样本的当前特征值
        feature = train_data[:,i]
        
        # 因为在载入训练集时已经将特征值转换为0-1
        # 此时一共三种切分点
        for divide in [-0.5, 0.5, 1.5]:
            
            # 对于一个切分点而言存在两种情况
            # 第一种是大于切分点为1，小于切分点为-1
            # 第二种是大于切分点为-1，小于切分点为1
            for rule in ['H1L-1', 'H-1L1']:
                
                # 在每一种情况下对样本进行判断为1还是-1
                # 相当于评估这个分类器的好坏
                PredictArr, e = Estimate(feature, divide, rule, train_label, WeightArr)
                
                # 如果错误率比之前所有的基本分类器都低
                # 则记录当前基本分类器的参数
                if e < error:
                    error = e
                    SingleTreeDict['feature'] = i
                    SingleTreeDict['divide'] = divide
                    SingleTreeDict['rule'] = rule
                    SingleTreeDict['error'] = e
                    SingleTreeDict['PredictArr'] = PredictArr
                    
    return SingleTreeDict


# In[ ]:


'''
创建提升树
----------
输入：
train_data：训练集数据
train_label：训练集标记
Number：想要得到多少个基本分类器
--------
输出：
SingleTreeArr：基本分类器的集合
AlphaArr：基本分类器权重的集合
'''
def CreateBoostingTree(train_data, train_label, Number):
    
    # m为样本容量，n为特征数量
    m, n = np.shape(train_data)
    
    # 准备一个列表用来存储所有的单一最优基本分类器
    SingleTreeArr = []
    
    # 准备一个列表用来存储所有的单一最优基本分类器的权重
    AlphaArr = []
    
    # 初始化权值
    WeightArr = [1 / m] * m
    
    for N in range(Number):
        print('Creating ', N + 1, ' SingleTree')
    
        # 这里需要createsingleboostingtree
        # SingleTree是当前基本分类器
        # e为基本分类器的误差
        # Gxi是当前基本分类器的预测结果
        SingleTreeDict = CreateSingleBoostingTree(train_data, train_label, WeightArr)

        # 保存当前基本分类器
        SingleTreeArr.append(SingleTreeDict)
        
        # 获得当前基本分类器的分类误差率
        e = SingleTreeDict['error']

        # 计算基本分类器在强分类器中的权重
        alpha = np.log((1 - e) / e) / 2

        # 保存当前基本分类器的权重
        AlphaArr.append(alpha)
        
        # 计算样本的新权重
        # 首先得到当前基本分类器的预测结果
        Gxi = SingleTreeDict['PredictArr']
        
        # 计算exp项
        Exp = np.exp(-1 * alpha * train_label * Gxi)
        
        # 计算规范化因子
        Z = np.dot(WeightArr, Exp)

        # 最后计算样本的新权重
        WeightArr = WeightArr * Exp / Z
    
    return SingleTreeArr, AlphaArr


# In[ ]:


'''
预测
---------
输入：
feature：依据哪个特征进行分类
divide：切分点
rule：分类规则
x：一个样本
----------
输出：
预测值
'''
def predict(feature, divide, rule, x):
    
    # 用于之后预测在该情况下每个样本属于哪个类
    if rule == 'H1L-1':
        H = 1
        L = -1
    else:
        H = -1
        L = 1
    
    # 根据特征值的大小返回相应预测值
    if x[feature] > divide:
        return H
    else:
        return L


# In[ ]:


'''
分类器
--------
输入：
SingleTreeArr：基本分类器的集合
AlphaArr：基本分类器的权重的集合
Number：基本分类器的数量
test_data：测试集数据
test_label：测试集标记
--------
输出：
Acc：正确率
'''
def Classifier(SingleTreeArr, AlphaArr, Number, test_data, test_label):
    
    # 初始化errorCnt
    errorCnt = 0
    
    # 遍历每一个测试集实例
    for i in range(len(test_label)):
        
        xi = test_data[i]
        yi = test_label[i]
        
        # 初始化强分类器的结果
        StrongResult = 0
        
        # 遍历每一个基本分类器
        for j in range(Number):

            # 得到基本分类器
            SingleTree = SingleTreeArr[j]

            # 得到基本分类器的参数
            feature = SingleTree['feature']
            divide = SingleTree['divide']
            rule = SingleTree['rule']

            # 预测得到当前基本分类器的结果
            WeakResult = predict(feature, divide, rule, xi)

            # 得到当前基本分类器的权重
            alpha = AlphaArr[j]
            
            # 将结果和权重相乘，并对每个基本分类器重复以上步骤
            # 最终得到强分类器的结果
            StrongResult += alpha * WeakResult
        
        # 得到强分类器对当前测试实例的分类
        FinalResult = np.sign(StrongResult)
        
        # 如果分类错误，errorCnt + 1
        if FinalResult != yi:
            errorCnt += 1
    
    # 计算正确率
    Acc = 1 - errorCnt / len(test_label)
    
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
    
    # 想要多少个基本分类器
    Number = 20
    
    # 学习强分类器
    print('start training')
    start = time.time()
    SingleTreeArr, AlphaArr = CreateBoostingTree(train_data, train_label, Number)
    print('end training')
    end = time.time()
    print('training time: ', end - start)

    # 进行分类
    start = time.time()
    print('start testing')
    Accuracy = Classifier(SingleTreeArr, AlphaArr, Number, test_data, test_label)
    print('end testing')
    end = time.time()
    print('testing time: ', end - start)
    print('Accuracy: ', Accuracy)

