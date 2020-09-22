#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import time
import pandas as pd
import random
import matplotlib.pyplot as plt


# In[ ]:


'''
读取训练数据
file即数据的地址
'''
def LoadData(file):
    
    # 开始记录读取数据时间
    # 这里通过时间的记录可以查看每一步所消耗的时间，可以针对性的进行优化
    start = time.time()
    print('start loading')

    # 读取训练集数据
    # pandas读取数据较快，但是内存负担较重，如果数据集太多，建议以逐行的方式读取数据
    Alldata = pd.read_csv(file)

    # 将dataframe中的特征值部分转换成矩阵
    # Alldata.iloc[:,1:] 去掉label部分
    # .values 提取其值，不需要表头
    # np.mat() 转换成矩阵形式
    data = np.mat(Alldata.iloc[:,1:].values)
    
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
    # .T为转置
    label = np.mat(label).T
    
    # 结束记录读取数据时间
    print('end loading')
    end = time.time()
    print('loading time: ' , end - start)
    
    # 返回数据的特征部分和标记部分
    return data, label


# In[ ]:


'''
开始训练 - 随机梯度下降法
train_data：训练集的数据部分
train_label：训练集的标记部分
itertime：循环次数
'''
def Perceptron(train_data, train_label, itertime):
    
    # 开始记录训练模型时间
    start = time.time()
    print('start training')
    
    # 获得实例数和特征数
    sampleNum, featureNum = np.shape(train_data)
    
    # 使用特征数为w赋予初值，b也赋予初值，并且给一个学习率
    # 因为w是权值向量，一个权值对应一个特征，因此依据训练集的特征数，初始化每个权值为0
    # 此处可以自行进行调整参数以优化模型
    w = np.zeros(featureNum)
    b = 0
    h = 0.0001
    
    # 迭代开始，t代表迭代数
    for t in range(itertime):
        
        # 考虑到采用随机梯度下降法，使用while循环
        # 每次迭代时找到一个误分类点即开始下一轮迭代
        a = 0
        while a == 0:
            
            # 用random函数在样本实例数之间随机选择一个样本点的下标
            # 之后加了一个[0]是因为random.sample输出为列表，我们只需要其中的值
            i = random.sample(range(0,sampleNum - 1), 1)[0]

            # 获取该样本点的特征和标记
            xi = train_data[i]
            yi = train_label[i]
            
            # 检查该样本点是否为误分类点
            if -1 * yi * (w * xi.T + b) >= 0:
                
                # 如果是，求w和b对L的偏导
                wG = -1 * yi * xi
                bG = -1 * yi
                
                # 更新w和b
                w = w + h * -1 * wG
                b = b + h * -1 * bG
                
                # 跳出循环，如果不是继续用random进行随机选择
                a = 1
    
    # 结束记录训练模型时间
    print('end training')
    end = time.time()
    print('training time: ', end - start)
    
    # 返回参数w和b，得到模型
    return w, b


# In[ ]:


'''
开始测试
test_data：测试集的数据部分
test_label：测试集的标记部分
w，b：模型参数
'''
def classifier(test_data, test_label, w, b):
    
    # 开始记录测试模型时间
    start = time.time()
    print('start testing')
    
    # 得到测试集的实例数量
    testNum = len(test_label)
    
    # 误分类的实例的数量，初始化为0
    errorCnt = 0
    
    # 遍历测试集中的实例，进行分类
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
    print('testing time: ', end - start)
    
    # 返回正确率
    return Accurate


# In[ ]:


'''测试整个程序'''
if __name__ ==  '__main__':
    
    # 输入训练集
    train_data, train_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_train.csv')
    
    # 训练模型
    itertime = 1200
    w, b = Perceptron(train_data, train_label, itertime)
    
    # 输入测试集
    test_data, test_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_test.csv')
    
    # 测试模型
    accurate = classifier(test_data, test_label, w, b)


# In[ ]:


'''测试迭代次数和正确率之间的关系'''
if __name__ ==  '__main__':
    
    # 输入训练集
    train_data, train_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_train.csv')
    
    # 输入测试集
    test_data, test_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_test.csv')
    
    # 测试不同的itertime
    itertime_group = np.linspace(100,2000,40)
    
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
    

