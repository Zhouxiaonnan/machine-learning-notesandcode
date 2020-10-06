#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
EM算法
-------------------------------------
数据集：两个生成的正态分布
训练集数量：10000
---------------------------------
训练结果：
时间：0.04s
真实参数：mu1 1.20  sigma1 0.70  alpha1 0.30  mu2 2.30  sigma2 1.20  alpha2 0.70
模型参数：mu1 1.18  sigma1 0.65  alpha1 0.26  mu2 2.24  sigma2 1.21  alpha2 0.74
'''

import numpy as np
import time


# In[ ]:


'''生成数据集'''
def CreateData(mu1, sigma1, alpha1, mu2, sigma2, alpha2, length):
    
    # 生成第一个分模型的数据
    First = np.random.normal(loc = mu1, scale = sigma1, size = int(length * alpha1))
    
    # 生成第二个分模型的数据
    Second = np.random.normal(loc = mu2, scale = sigma2, size = int(length * alpha2))
    
    # 混合两个数据
    ObservedData = np.concatenate((First, Second), axis = 0)
    
    # 打乱顺序(打不打乱其实对算法没有影响)
    np.random.shuffle(ObservedData)
    
    return ObservedData


# In[ ]:


'''根据高斯密度函数计算'''
def Cal_Gaussian(Data, mu, sigma):
    
    GaussianP = 1 / (sigma * np.sqrt(2 * np.pi))                 * np.exp(-1 * np.square(Data - mu) / (2 * np.square(sigma)))
    
    return GaussianP


# In[ ]:


'''E步'''
def E_step(Data, mu1, sigma1, alpha1, mu2, sigma2, alpha2):
    
    # 计算分模型对每个样本的响应度
    # 这里Data使用的是ndarray格式
    # 因此得到的数列结果中，包含该分模型对每个样本的计算响应度公式的分子项
    Gamma1 = alpha1 * Cal_Gaussian(Data, mu1, sigma1)
    Gamma2 = alpha2 * Cal_Gaussian(Data, mu2, sigma2)
    
    # 计算响应度公式的分母项
    Summary = Gamma1 + Gamma2

    # 计算响应度
    Gamma1 = Gamma1 / Summary
    Gamma2 = Gamma2 / Summary
    
    return Gamma1, Gamma2   


# In[ ]:


'''M步'''
def M_step(Data, mu1_old, mu2_old, Gamma1, Gamma2):
    
    # 计算新的参数mu
    mu1_new = np.dot(Gamma1, Data) / np.sum(Gamma1)
    mu2_new = np.dot(Gamma2, Data) / np.sum(Gamma2)
    
    # 计算新的参数sigma
    sigma1_new = np.sqrt(np.dot(Gamma1, np.square(Data - mu1_old)) / np.sum(Gamma1))
    sigma2_new = np.sqrt(np.dot(Gamma2, np.square(Data - mu2_old)) / np.sum(Gamma2))
    
    # 计算新的参数alpha
    m = len(Data)
    alpha1_new = np.sum(Gamma1) / m
    alpha2_new = np.sum(Gamma2) / m
    
    return mu1_new, sigma1_new, alpha1_new, mu2_new, sigma2_new, alpha2_new


# In[ ]:


'''EM算法'''
def EM(Data, mu1, sigma1, alpha1, mu2, sigma2, alpha2, itertime):
    
    # 迭代
    for i in range(itertime):
        
        # 计算响应度，E步
        Gamma1, Gamma2 = E_step(Data, mu1, sigma1, alpha1, mu2, sigma2, alpha2)
    
        # 更新参数，M步
        mu1, sigma1, alpha1, mu2, sigma2, alpha2         = M_step(Data, mu1, mu2, Gamma1, Gamma2)
    
    return mu1, sigma1, alpha1, mu2, sigma2, alpha2


# In[ ]:


'''测试模型'''
if __name__ == '__main__':

    # 生成数据
    mu1 = 1.2
    sigma1 = 0.7
    alpha1 = 0.3
    mu2 = 2.3
    sigma2 = 1.2
    alpha2 = 0.7
    length = 10000
    Data = CreateData(mu1, sigma1, alpha1, mu2, sigma2, alpha2, length)

    # 初始化数据
    init_mu1 = 0.9
    init_sigma1 = 0.4
    init_alpha1 = 0.2
    init_mu2 = 2
    init_sigma2 = 1.4
    init_alpha2 = 0.8
    itertime = 100

    # 训练模型
    start = time.time()
    print('start training')
    model_mu1, model_sigma1, model_alpha1, model_mu2, model_sigma2, model_alpha2 =     EM(Data, init_mu1, init_sigma1, init_alpha1, init_mu2, init_sigma2, init_alpha2, itertime)
    print('end training')
    end = time.time()
    print('training time: ', end - start)


    print('真实参数：mu1 ', mu1, 'sigma1 ', sigma1, 'alpha1 ', alpha1, 'mu2 ', mu2, 'sigma2 ', sigma2, 'alpha2 ', alpha2)
    print('模型参数：mu1 ', model_mu1, 'sigma1 ', model_sigma1, 'alpha1 ', model_alpha1, 'mu2 ', model_mu2, 'sigma2 ', model_sigma2, 'alpha2 ', model_alpha2)


# In[ ]:




