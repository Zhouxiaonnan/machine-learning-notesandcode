#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
决策树 - CART
-------------------------------------
数据集：Mnist
训练集数量：60000
测试集数量：10000
---------------------------------
训练结果：
将训练集的标记分组：3组
训练时间：2.2h
测试时间：0.17s
正确率：0.86
'''


import numpy as np
import time
import collections as cc
from itertools import islice


# In[2]:


'''
readlines方法读取训练数据
--------
输入：
file：数据地址
n：将连续型变量分成几组，在CART树中，为了说明CART树的特性，这里设置为3组
--------
输出：
data：转换后的数据
label：转换后的标记
'''
def LoadData(file, n):

    
    # 打开数据文件
    fr = open(file, 'r')
    
    # 特征值分组
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
计算基尼系数
-----------
输入：
label：当前结点的样本的标记
-----------
输出：
G：当前结点的数据的标记的基尼系数
'''
def Gini(label):
    
    # 所有的类
    Class = np.unique(label)
    
    # 对每个类统计出现次数
    ClassNum = cc.Counter(label)
    
    # 得到标记的总数
    Classlen = len(label)
    
    # 初始化基尼系数
    G = 1
    
    # 遍历每一个类
    for c in Class:
        
        # 计算基尼系数
        G -= (ClassNum[c] / Classlen) ** 2
    
    return G


# In[5]:


'''
遍历每个特征和每个可能的分割点，计算Gini系数
-----------
输入：
data：当前结点的样本的数据
label：当前结点的样本的标记
-----------
输出：
f_GMin_f：最小基尼系数对应的特征
f_GMin_f_v：最小基尼系数对应的特征值
f_GMin：最小基尼系数
'''
def Gini_DA(data, label):

    # 特征的数量
    FeatureNum = len(data[0])
    
    # 得到数据集的样本数量
    dataNum = len(data)
    
    # 将数据集转置
    data_T = np.transpose(data)
    
    # 初始化基尼系数
    # 为什么初始化为1，可以查看决策树理论部分的文章
    f_GMin = 1
    
    # 遍历每个特征
    for f in range(FeatureNum):
        
        # 所有样本点的特征f的值
        f_data = data_T[f]
        
        # 特征f的可取值
        f_value = np.unique(f_data)
        
        # 遍历特征f的每一个可取值
        for f_v in f_value:
            
            # 得到f_data中值为f_v的所有index
            index = np.argwhere(f_data == f_v)
            
            # 准备一个空列表存储满足值为f_v的所有标记
            f_label_r = []
            f_label_l = []
            
            # 遍历每个label，对label进行划分
            for i in range(len(label)):
                if i in index:
                    f_label_l.append(label[i])
                else:
                    f_label_r.append(label[i])
 
            # 计算f_label_r和f_label_f的Gini系数
            G_l = Gini(f_label_l)
            G_r = Gini(f_label_r)
            
            # 得到在特征f和值f_v下的Gini系数
            f_G = len(f_label_l) / dataNum * G_l + len(f_label_r) / dataNum * G_r
            
            # 如果新算出来的基尼系数比之前的更小，更新最小值，特征和特征的值
            if f_G < f_GMin:
                f_GMin = f_G
                f_GMin_f_v = f_v
                f_GMin_f = f
    
    return f_GMin_f, f_GMin_f_v, f_GMin


# In[6]:


'''
在得到最适合用来划分的特征和特征值后，对数据集进行划分
----------
输入：
data：目前的数据集
label：目前的标记
feature：根据哪个特征进行划分
value：根据该特征的哪个值进行划分
-------------
输出：
datasets_r：划分后得到的右子集
labelsets_r：划分后得到的右子集的标记
datasets_l：划分后得到的左子集
labelsets_l：划分后得到的左子集的标记
'''
def SplitDataSet(data, label, feature, value):
    
    # 样本数
    SampleNum = len(label)
    
    # 转置data
    data_T = np.transpose(data)
    
    # 准备列表用来存放分割后的子集
    # l代表左，r代表右
    datasets_l = []
    labelsets_l = []
    datasets_r = []
    labelsets_r = []
        
    # enumerate不仅遍历元素，同时遍历元素的下标
    # 此处遍历每个样本在最佳特征的取值和下标
    for Index, num in enumerate(data_T[feature]):

        # 当data中的某个样本的该特征值=value时，将该样本存储在右侧
        if num == value:

            # 将用于划分该样本点的最佳特征从数据集中去除
            data_temp = data[Index]

            # 存储划分后的子集
            # 此时得到的仅为最佳特征的一个取值下的子集
            datasets_l.append(data_temp)
            labelsets_l.append(label[Index])
        
        # 当data中的某个样本的该特征值不等于value时，将该样本存储在左侧
        else:
            data_temp = data[Index]
            datasets_r.append(data_temp)
            labelsets_r.append(label[Index])

    return datasets_r, labelsets_r, datasets_l, labelsets_l


# In[7]:


'''
创建决策树
----------
输入：
pre_train_data: 当前训练集数据
pre_train_label：当前训练集标记
epsilon：阈值，如果当前结点的基尼系数小于该值，则将该结点设为叶节点
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
    
    # 其它情况下，需要继续对结点进行分类，计算Gini系数
    # 得到最小基尼系数对应的特征，特征值和最小基尼系数
    BestFeature, BestFeatureValue, GMin = Gini_DA(pre_train_data, pre_train_label)
    
    # 如果最佳特征及特征值的Gini系数小于一个阈值
    # 则采用当前标记中数目最多的类
    if GMin < epsilon:
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
    
    # 树生成后，对数据集根据最佳特征和特征值进行划分
    # CART树是二叉树，因此我们可以保存好左子集及标记，和右子集及标记
    subdatasets_r, sublabelsets_r, subdatasets_l, sublabelsets_l =     SplitDataSet(pre_train_data, pre_train_label, BestFeature, BestFeatureValue)
    
    # 这里运用的迭代思想，在函数中调用子集
    # 左子集是最佳特征=最佳特征值的样本的集合，因此使用BestFeatureValue
    # 右子集是其它样本的集合，因此使用'other'
    treeDict[BestFeature][BestFeatureValue] = CreateTree(subdatasets_l, sublabelsets_l, epsilon)
    treeDict[BestFeature]['other'] = CreateTree(subdatasets_r, sublabelsets_r, epsilon)

    return treeDict


# In[8]:


'''
对单个测试点进行分类
-----------
输入：
data：测试集的单个样本
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
        
        # 在value_list中包括判断走左子树还是右子树的值
        # 左子树一定是数字（0或1或2），右子树一定是'other'
        # 当value_list中包含feature_value，则走左子树
        # 如果不包含，则走右子树
        value_list = list(value.keys())
        
        # 判断value_list中是否包含feature_value
        if feature_value in value_list:
            
            # 如果包含，接着判断左子树的根结点是内部结点还是叶结点
            # 如果类型是dict，那么就说明还在内部结点，要继续往下分
            if type(value[feature_value]).__name__ == 'dict':

                # 将该内部结点及其子树设为新的树
                tree = value[feature_value]

            # 如果判断下来是不是字典了，说明到达叶节点
            if type(value[feature_value]).__name__ != 'dict':

                # 则返回叶结点对应的分类
                Class = value[feature_value]
        
        # 如果value_list中不包含feature_value
        # 则走右子树
        else:

            if type(value['other']).__name__ == 'dict':
                tree = value['other']
            if type(value['other']).__name__ != 'dict':
                Class = value['other']
    
    return Class


# In[9]:


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


# In[10]:


'''测试模型'''
if __name__ == "__main__":

    # 输入训练集
    train_data, train_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_train.csv', 3)

    # 输入测试集
    test_data, test_label = LoadData('E:/我的学习笔记/统计学习方法/实战/训练数据/mnist_test.csv', 3)
    
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


# In[ ]:




