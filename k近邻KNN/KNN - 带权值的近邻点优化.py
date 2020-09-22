#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
k近邻模型 - 线性扫描 - 带权值的近邻点优化
-------------------------------------
数据集：Mnist
训练集数量：60000
测试集数量：200
近邻点：30
---------------------------------
训练结果：
时间：241s
正确率：0.956
'''


import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


'''
pandas方法读取训练数据
file：数据地址
'''

def LoadData(file):
    
    print('loading data')
    
    # 读取训练集数据
    Alldata = pd.read_csv(file)

    # 将dataframe中的特征值部分转换成矩阵
    data = np.mat(list(Alldata.iloc[:,1:].values))

    # 将dataframe中的标记部分转换成矩阵
    label = np.mat(Alldata['label']).T
    
    print('end loading data')
    
    # 返回数据的特征部分和标记部分
    return data, label


# In[ ]:


'''
计算新输入点与所有训练集样本点的距离
train_data：训练集的特征部分
x：需要分类的新输入实例点的特征值
'''
def Distance(train_data, x):
    
    # 将新输入的实例点的特征值转换成矩阵
    x = np.mat(x)
    
    # 计算该点到每个样本点的距离，这里使用的是欧式距离
    d = np.sqrt(np.sum(np.square(train_data - x), axis = 1))
    
    # 样本点的个数就是距离数列中元素的个数
    # 将距离矩阵转换成一个一维的数列
    sampleNum = np.shape(train_data)[0]
    dArr = np.array(d).reshape(sampleNum)

    # 返回距离数列
    return dArr


# In[ ]:


'''
选择距离新输入实例距离最短的K个点的index
distance：距离数列
K：近邻点的个数
'''
def ClosestK(distance, K):
    
    # 得到对距离排序的index
    # 比如np.argsort[1,3,2]得到的是[0,2,1]，输出为原列表中从小到大排列的元素的index
    sort = np.argsort(distance)
    
    # 得到距离最近的K个点的index
    sampleK_index = sort[:K]
    
    # 返回距离最近的K个点的index
    return sampleK_index


# In[ ]:


'''
计算距离最短的K个点中每个类的个数，并得到个数最大的那个类
train_label：训练集的标记
sampleK_index：距离最近的K个点的index
distance：距离
'''
def ClassN(train_label, sampleK_index, dArr):
    
    # 得到类别的个数
    ClassNum = len(np.unique(np.array(train_label)))
    
    # 准备一个列表，用来存放对应类别的个数，因为这里的类别就是0-9
    # 因此可以用列表的index代表其类别
    Class_record = [0] * ClassNum
    
    # 遍历每一个距离最近的样本点，sampleK_index中记录的是该点对应的train_label的index
    # 因此train_label[index]即该点的类别
    # 这里采用了高斯函数根据距离对邻近点赋予不同的权值
    # 距离越小，权值越大，最大权值为1，weighted即计算后带权距离
    # 在对应的class_record的位置加上weighted
    for index in sampleK_index:
        weighted = np.exp(-1 * (dArr[index]) ** 2 / (2 * 500 ** 2))
        Class_record[np.array(train_label)[index][0]] += weighted
    
    # 找到Class_record中最大的值的index，即对应了最多个数的类别
    Class = Class_record.index(max(Class_record))
    
    # 返回新输入实例的类
    return Class


# In[ ]:


'''
分类器，对测试集中的数据进行分类
train_data：训练集的特征部分
train_label：训练集的标记部分
test_data：测试集特征部分
test_label：测试集标记部分
K：近邻的实例的个数
'''
def Classifier(train_data, train_label, test_data, test_label, K):
    
    start = time.time()
    
    # 赋予错误分类数量初值
    errorCnt = 0
    
    # 得到测试集的记录的个数
    testNum = np.shape(test_data)[0]
    
    # 遍历每一个新输入实例
    for i in range(testNum):

        # 显示正在进行第几个新输入实例点的分类
        #print('classifying %d' % i)
        
        # 得到新输入点与所有训练集样本点的距离
        dArr = Distance(train_data, test_data[i])
        
        # 得到距离新输入实例距离最短的K个点的index
        sampleK_index = ClosestK(dArr, K)
        
        # 得到新输入实例的分类
        C = ClassN(train_label, sampleK_index, dArr)
        
        # 如果分类错误，则errorCnt+1
        if C != test_label[i]:
            errorCnt += 1
    
    # 计算正确率
    Accurate = 1 - (errorCnt / testNum)
    
    end = time.time()
    print('Classifying time: ', end - start)
    print('Accurate = ', Accurate)
    
    # 返回正确率
    return Accurate


# In[ ]:


'''考察k值与正确率的关系'''
if __name__ == "__main__":

    # 输入训练集
    train_data, train_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_train.csv')
    
    # 输入测试集
    test_data, test_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_test.csv')
    
    # 由于测试数据太多，这里仅考虑500条
    testN = 500
    test_data = test_data[0:testN]
    test_label = test_label[0:testN]
    
    # 考察不同的K值，即不同的近邻点的个数，对正确率的影响
    K = np.linspace(10,100,10)
    Acc = []
    
    # 开始对每个K值进行测试
    for k in K:
        print('k值为：', k)
        
        # 记录分类所需要的时间
        start = time.time()
        
        # 得到正确率
        Accurate = Classifier(train_data, train_label, test_data, test_label, int(k))
        
        # 得到不同K值对应的一组正确率
        Acc.append(Accurate)
        
        # 记录不同K值所需时间
        end = time.time()
        
    # 画图
    plt.plot(K,Acc)
    plt.xlabel('lowest distance sample dots number')
    plt.ylabel('Accurate')

'''
在考察k值与正确率的关系的时候
因为用于测试的新输入点没有变化
因此这里可以一次算完新输入点到所有训练集样本点的距离
再根据不同的K值得到分类器
而不需要每次K值变化都计算一次距离
这样可以省下很多时间
'''

