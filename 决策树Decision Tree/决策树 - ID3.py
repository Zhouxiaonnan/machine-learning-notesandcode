#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
决策树 - ID3
-------------------------------------
数据集：Mnist
训练集数量：60000
测试集数量：10000
---------------------------------
训练结果：
将训练集的标记分组：2组
训练时间：303s
测试时间：0.06s
正确率：0.85
'''


import numpy as np
import time
import collections as cc
from itertools import islice
import matplotlib.pyplot as plt


# In[2]:


'''
readlines方法读取训练数据
--------
输入：
file：数据地址
n：将连续型变量分成几组
这里的参数n是为了考察对连续型变量分组的组数与正确率的关系
--------
输出：
data：转换后的数据
label：转换后的标记
'''
def LoadData(file, n):

    
    # 打开数据文件
    fr = open(file, 'r')
    
    # 在Mnist数据集中，严格来说，其特征并非是连续型变量
    # 但因为每个特征可取值为0-255，一共256个
    # 如果不对特征值进行分组，决策树将是一个很浅很宽的决策树
    # 这样的决策树是过拟合的（可以查看决策树理论部分）
    # 因此需要对特征值进行分组
    # 这里start用了-0.1是为了方便包含特征值0
    group = np.linspace(-0.1,255,n+1)
    
    # 准备两个list存储data和label
    data = []
    label = []
    
    # 逐行读取数据，使用islice可以直接跳过第一行表头进行逐行读取
    for line in islice(fr,1,None):
        
        # 对每一行消去空格，并根据','进行分割
        splited = line.strip().split(',')
        
        # 准备一个空list用来存储转换后的特征值
        int_line = []
        
        # 分割后的第一个元素是Label，跳过label遍历所有特征值
        for num in splited[1:]:
            
            a = 0
            i = 0
            while a==0 :
                
                # 在第一组内则将其转换为0，在第二组内则将其转换为1，以此类推
                if int(num) > group[i] and int(num) <= group[i+1]:
                    
                    int_line.append(i)
                    
                    a = 1
                
                i += 1
        
        # 逐行存储数据
        data.append(int_line)
        label.append(int(splited[0]))

    # 返回数据的特征部分和标记部分
    return data, label


# In[3]:


'''
找到当前label集的数目最大的一个类
---------
输入：
label：当前结点拥有的label集
---------
输出：
MajorClass：当前结点拥有的label集中数目最大的一个类
'''
def FindMajorClass(label):
    
    # 对label集进行统计
    # .most_common(1)，从Counter中提起出现数目最多的一组
    MajorClass = cc.Counter(label).most_common(1)[0][0]

    return MajorClass


# In[4]:


'''
计算经验熵
-----------
输入：
label：当前结点的样本的标记
-----------
输出：
H：当前结点的数据的标记的经验熵
'''
def Entropy(label):
    
    # 所有的类
    Class = np.unique(label)
    
    # 对每个类统计出现次数
    ClassNum = cc.Counter(label)
    
    # 得到标记的总数
    Classlen = len(label)
    
    # 初始化熵
    H = 0
    
    # 遍历每一个类
    for c in Class:
        
        # 计算每个类出现的概率
        P = ClassNum[c] / Classlen
    
        # 计算经验熵
        # 这里的对数以e为底
        H += -1 * P * np.log(P)
    
    return H


# In[5]:


'''
计算条件熵
-----------
输入：
data：当前结点的样本的数据
label：当前结点的样本的标记
-----------
输出：
C_H_Arr：当前结点的样本的条件熵
'''
def Conditional_Entropy(data, label):

    # 特征的数量
    FeatureNum = len(data[0])
    
    # 得到数据集的样本数量
    dataNum = len(data)
    
    # 将数据集转置
    data = np.transpose(data)
    
    # 准备一个空列表存放每个feature对应的经验条件熵
    C_H_Arr = []
    
    # 遍历每个特征
    for f in range(FeatureNum):
        
        # 所有样本点的特征f的值
        f_data = data[f]
        
        # 特征f的可取值
        f_value = np.unique(f_data)
        
        # 初始化特征f的经验条件熵
        C_H = 0
        
        # 遍历特征f的每一个可取值
        for f_v in f_value:
            
            # 得到f_data中值为f_v的所有index
            index = np.argwhere(f_data == f_v)
            
            # 准备一个空列表存储满足值为f_v的所有标记
            f_label = []
            
            for i in index:
                
                # 得到该特征下满足值为f_v的对应的所有标记
                f_label.append(label[i[0]])
            
            # 计算f_label的经验熵
            f_H = Entropy(f_label)
            
            # 得到在该特征下，值为f_v的概率
            f_P = len(f_label) / dataNum
            
            # 计算经验条件熵
            C_H += f_P * f_H
            
        # 记录每个特征的条件熵
        C_H_Arr.append(C_H)

    return C_H_Arr


# In[6]:


'''
计算各个特征的信息增益
---------
输入：
data：当前结点下的数据
label：当前结点下的标记
-----------
输出：
BestFeature：最适合用来进行分类的特征
IGMax：该特征对应的最大信息熵
'''
def InforGain(data, label):

    # 计算当前结点的经验熵
    H = Entropy(label)
    
    # 计算当前结点的经验条件熵
    C_H_Arr = Conditional_Entropy(data, label)
    
    # 得到最大的信息增益
    IG = [H - num for num in C_H_Arr]
    IGMax = max(IG)
    
    # 得到最大信息增益对应的特征
    BestFeature = IG.index(IGMax)
    
    return BestFeature, IGMax


# In[7]:


'''
在得到最适合用来划分的特征后，对数据集进行划分
----------
输入：
data：目前的数据集
label：目前的标记
feature：根据哪个特征进行划分
-------------
输出：
datasets：划分后得到的多个子集
labelsets：划分后得到的多个子集的标记
'''
def SplitDataSet(data, label, feature):
    
    # 样本数
    SampleNum = len(label)
    
    # 转置data
    data_T = np.transpose(data)
    
    # 获得最佳特征的可取值
    feature_value = np.unique(data_T[feature])
    
    # 准备两个列表，用来存放分割后的子集
    datasets = []
    labelsets = []
    
    # 遍历最佳特征的每个取值
    for f in feature_value:
        
        datasets_sub = []
        labelsets_sub = []
        
        # enumerate不仅遍历元素，同时遍历元素的下标
        # 此处遍历每个样本在最佳特征的取值和下标
        for Index, num in enumerate(data_T[feature]):

            # 当data中的某个样本的该特征=f时，获得它的index
            if num == f:
                
                # 将用于划分该样本点的最佳特征从数据集中去除
                # 去除后在下一次的迭代中将不再考虑这个特征
                data_temp = data[Index]
                del data_temp[feature]
                
                # 存储划分后的子集
                # 此时得到的仅为最佳特征的一个取值下的子集
                datasets_sub.append(data_temp)
                labelsets_sub.append(label[Index])
        
        # 存储根据最佳特征的不同取值划分的子集
        datasets.append(datasets_sub)
        labelsets.append(labelsets_sub)
    
    return datasets, labelsets


# In[8]:


'''
创建决策树
----------
输入：
pre_train_data: 当前训练集数据
pre_train_label：当前训练集标记
epsilon：阈值，如果当前结点的最大信息增益小于该值，则将该结点设为叶节点
-----------
输出：
treeDict：决策树
'''
def CreateTree(pre_train_data, pre_train_label, epsilon):

    # 类别去重
    Class = np.unique(pre_train_label)
    
    # 如果对于当前的标签集合而言，类别只有一个
    # 说明这个结点是叶结点，返回这个类
    if len(Class) == 1:
        return Class[0]
    
    # 如果已经没有特征可以进行分类了，返回当前label集中数目最多的类
    if len(pre_train_data[0]) == 0:
        return FindMajorClass(pre_train_label)
    
    # 其它情况下，需要继续对结点进行分类，计算信息增益
    # 得到信息增益最大的特征，及其信息增益
    BestFeature, IGMax = InforGain(pre_train_data, pre_train_label)
    
    # 如果最佳特征的信息增益小于一个我们自己设定的阈值
    # 则采用当前标记中数目最多的类
    if IGMax < epsilon:
        return FindMajorClass(pre_train_label)
    
    # 构建树
    # 这里使用了dict格式的特点来构建树
    # 比如得到的树是{374 : {0 : 2, 1 : {562 : {0 : 4, 1 : 7}}}}
    # 代表根结点对应特征是第374个特征
    # 如果一个样本的374特征的值为0，则该样本分类为2
    # 如果它的374特征为1，则进入子树
    # 子树的结点对应的特征为第562个特征
    # 如果样本的562特征值为0，则分类为4，如果值为1，则分类为7
    treeDict = {BestFeature:{}}
    
    # 树生成后，对数据集根据最佳特征进行划分
    subdatasets, sublabelsets = SplitDataSet(pre_train_data, pre_train_label, BestFeature)
    
    # 子集的个数
    setsNum = len(sublabelsets)
    
    # 对子集进行迭代，创建子树
    for i in range(setsNum):
        
        # 这里运用的迭代思想
        # 即在一个自定义函数中调用自己
        treeDict[BestFeature][i] = CreateTree(subdatasets[i], sublabelsets[i], epsilon)        

    return treeDict


# In[9]:


'''
对单个测试点进行分类
-----------
输入：
data：测试点数据
label：测试点标记
tree：决策树
--------
输出：
Class：该样本点被分的类
'''
def Predict(data, tree):
    
    # 初始化Class
    Class = -1
    
    # 当Class被赋予了新值，也就是说该样本点被分类，则停止循环
    while Class == -1:
        
        # 获得当前结点的key和value
        # key代表结点中需要对哪一个特征进行判断
        # value代表结点的可取值
        (key, value), = tree.items()
        
        # 该样本在结点所需判断的特征的值
        feature_value = data[key]
        
        # 如果判断下来，其值还是字典
        # 那么就说明还在内部结点，要继续往下分
        if type(value[feature_value]).__name__ == 'dict':

            # 将该内部结点及其子树设为新的树
            tree = value[feature_value]
            
            # 删除该结点所对应的特征
            del data[key]

        # 如果判断下来是不是字典了，说明到达叶节点
        if type(value[feature_value]).__name__ != 'dict':

            # 则返回叶结点对应的分类
            Class = value[feature_value]
    
    return Class


# In[10]:


'''
对测试集进行分类
------------
输入：
test_data：测试集数据
test_label：测试集标记
tree：决策树
-----------
输出：
Acc：正确率
'''
def Classifier(test_data, test_label, tree):
    
    # 测试集测试样本数量
    TestNum = len(test_label)
    
    # 初始化分类错误的个数
    errorCnt = 0
    
    # 遍历每一个测试样本点
    for i in range(TestNum):
        
        # 在对特征值比较细的分组的情况下，可能训练集中有的特征值未出现
        # 而在测试集中出现，会造成分类错误
        # 因此使用try - except跳过错误
        # 并将错误对应的样本分为-1
        # 出现错误的样本也被认为是分类错误的
        try:

            # 得到样本点的类
            Class = Predict(test_data[i], tree)
            
        except:
            Class = -1
            
        # 如果分类错误
        if Class != test_label[i]:
            errorCnt += 1
    
    # 计算正确率
    Acc = 1 - (errorCnt / TestNum)
    
    return Acc


# In[11]:


'''测试模型'''
if __name__ == "__main__":
    
    # 准备一个空列表记录不同分组情况下的正确率
    Accurate_group = []
    
    # 考察五种分组情况
    group = np.linspace(2,6,5)
    
    # 对于每一种分组情况
    for n in group:

        # 输入训练集
        train_data, train_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_train.csv', n)

        # 输入测试集
        test_data, test_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_test.csv', n)

        # 创建决策树
        print('start creating tree')
        start = time.time()

        tree = CreateTree(train_data, train_label, 0.1)

        end = time.time()
        print('end creating tree')
        print('create tree time: ', end - start)
        print(tree)

        # 测试模型

        print('start testing')
        start = time.time()

        accurate = Classifier(test_data, test_label, tree)
        print('Accurate:', accurate)

        end = time.time()
        print('end testing')
        print('test time: ', end - start)
        
        # 存储正确率
        Accurate_group.append(accurate)
    
    # 画图
    plt.plot(group, Accurate_group)
    plt.xlabel('group Num')
    plt.ylabel('Accurate')

