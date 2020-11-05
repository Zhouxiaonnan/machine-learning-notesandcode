# 机器学习：学习笔记&代码复现

如果觉得对您有帮助，还请帮忙点个star，谢谢各位大佬 ^_^

email：zxndd1996@163.com

微信：zxn059

知乎专栏：[舟晓南](https://zhuanlan.zhihu.com/c_1274454587772915712)

## 目录
- 《统计学习方法》学习笔记及复现代码
- 《tensorflow2.0》学习笔记及代码

## 以下内容为《统计学习方法》第二版学习笔记和复现代码

### 感知机（Perceptron）

- 理论：[感知机模型解读](https://zhuanlan.zhihu.com/p/213772724)
- 理论：[感知机模型的收敛性解读](https://zhuanlan.zhihu.com/p/213905084)
- 代码：[随机梯度下降法](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E6%84%9F%E7%9F%A5%E6%9C%BAperceptron/%E6%84%9F%E7%9F%A5%E6%9C%BA%E6%A8%A1%E5%9E%8B%20-%20%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95.py)
- 代码：[梯度下降法](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E6%84%9F%E7%9F%A5%E6%9C%BAperceptron/%E6%84%9F%E7%9F%A5%E6%9C%BA%E6%A8%A1%E5%9E%8B%20-%20%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95.py)
- 代码：[adagrad](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E6%84%9F%E7%9F%A5%E6%9C%BAperceptron/%E6%84%9F%E7%9F%A5%E6%9C%BA%E6%A8%A1%E5%9E%8B%20-%20adagrad.py)
- 代码：[对偶形式](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E6%84%9F%E7%9F%A5%E6%9C%BAperceptron/%E6%84%9F%E7%9F%A5%E6%9C%BA%E6%A8%A1%E5%9E%8B%20-%20%E5%AF%B9%E5%81%B6%E5%BD%A2%E5%BC%8F.py)

### k近邻（KNN）

- 理论：[k近邻（KNN）模型解读](https://zhuanlan.zhihu.com/p/214165992)
- 代码：[线性扫描](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/k%E8%BF%91%E9%82%BBKNN/KNN%20-%20%E7%BA%BF%E6%80%A7%E6%89%AB%E6%8F%8F.py)
- 代码：[带权值的近邻点优化](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/k%E8%BF%91%E9%82%BBKNN/KNN%20-%20%E5%B8%A6%E6%9D%83%E5%80%BC%E7%9A%84%E8%BF%91%E9%82%BB%E7%82%B9%E4%BC%98%E5%8C%96.py)

### 朴素贝叶斯（Bayes）

- 理论：[朴素贝叶斯模型解读](https://zhuanlan.zhihu.com/p/215721959)
- 理论：[朴素贝叶斯之后验概率最大化的含义](https://zhuanlan.zhihu.com/p/215897132)
- 代码：[贝叶斯估计](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AFBayes/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%20-%20%E8%B4%9D%E5%8F%B6%E6%96%AF%E4%BC%B0%E8%AE%A1.py)

### 决策树（Decision Tree）

- 理论：[决策树，CART决策树解析](https://zhuanlan.zhihu.com/p/222724664)
- 代码：[ID3](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91Decision%20Tree/%E5%86%B3%E7%AD%96%E6%A0%91%20-%20ID3.py)
- 代码：[C4.5](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91Decision%20Tree/%E5%86%B3%E7%AD%96%E6%A0%91%20-%20C4.5.py)
- 代码：[CART](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91Decision%20Tree/%E5%86%B3%E7%AD%96%E6%A0%91%20-%20CART.py)

### 逻辑斯蒂回归（Logistic Regression）

- 理论：[逻辑斯蒂回归解析](https://zhuanlan.zhihu.com/p/231627246)
- 理论：[最大熵模型解析](https://zhuanlan.zhihu.com/p/234442747)
- 理论：[改进的迭代尺度法IIS解析](https://zhuanlan.zhihu.com/p/234553402)
- 理论：[改进的迭代尺度法(IIS)中f#(x,y)的意义](https://zhuanlan.zhihu.com/p/265299086)
- 代码：[随机梯度下降法](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92Logistic%20Regression/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92%20-%20%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95.py)
- 代码：[SGDM](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92Logistic%20Regression/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92%20-%20SGDM.py)
- 代码：[RMSProp](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92Logistic%20Regression/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92%20-%20RMSProp.py)
- 代码：[Adam](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92Logistic%20Regression/%E9%80%BB%E8%BE%91%E6%96%AF%E8%92%82%E5%9B%9E%E5%BD%92%20-%20Adam.py)

### 支持向量机（SVM）

- 理论：[线性可分支持向量机解析](https://zhuanlan.zhihu.com/p/235266761)
- 理论：[线性支持向量机解析](https://zhuanlan.zhihu.com/p/237540358)
- 理论：[非线性支持向量机解析](https://zhuanlan.zhihu.com/p/240659581)
- 理论：[序列最小最优化算法（SMO）解析](https://zhuanlan.zhihu.com/p/248862271)
- 代码：[支持向量机 - SMO - 高斯核函数](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BASVM/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%20-%20SMO%20-%20%E9%AB%98%E6%96%AF%E6%A0%B8%E5%87%BD%E6%95%B0.py)

### 提升算法（Boosting）

- 理论：[提升算法Adaboost解析](https://zhuanlan.zhihu.com/p/250458152)
- 理论：[提升树算法解析](https://zhuanlan.zhihu.com/p/252398216)
- 代码：[提升树](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/%E6%8F%90%E5%8D%87%E6%A0%91Boosting%20Tree/Adaboost.py)

### EM算法

- 理论：[EM算法解析](https://zhuanlan.zhihu.com/p/254871111)
- 理论：[高斯混合模型的EM算法解析](https://zhuanlan.zhihu.com/p/262243267)
- 代码：[EM算法 - 高斯混合模型](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/EM%E7%AE%97%E6%B3%95/EM%E7%AE%97%E6%B3%95.py)

### 隐马尔可夫模型

- 理论：[隐马尔可夫模型解析](https://zhuanlan.zhihu.com/p/263493585)

### 条件随机场模型

- 理论：[条件随机场模型解析](https://zhuanlan.zhihu.com/p/265341256)

### 聚类方法

- 理论：[层次聚类，k均值模型](https://zhuanlan.zhihu.com/p/266163764)

### 奇异值分解(SVD)

- 理论：[奇异值分解](https://zhuanlan.zhihu.com/p/266554196)

### 主成分分析(PCA)

- 理论：[主成分分析](https://zhuanlan.zhihu.com/p/269204488)

### 潜在语义分析(LSA)

- 理论：[潜在语义分析](https://zhuanlan.zhihu.com/p/270553039)

### 概率潜在语义分析(PLSA)

- 理论：[概率潜在语义分析](https://zhuanlan.zhihu.com/p/270889121)

### 马尔可夫链蒙特卡罗法(MCMC)

- 理论：[马尔可夫链蒙特卡罗法](https://zhuanlan.zhihu.com/p/271682379)

----------
## 《tensorflow》学习笔记及代码

- [tensorflow2.0的特点](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/tensorflow/tensorflow2.0%20-%20tensorflow2.0%E7%9A%84%E7%89%B9%E7%82%B9.ipynb)
- [tensorflow2.0 - 基础1 - 创建和数据类型及其应用](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/tensorflow/tensorflow2.0%20-%20%E5%9F%BA%E7%A1%801%20-%20%E5%88%9B%E5%BB%BA%E5%92%8C%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%BA%94%E7%94%A8.ipynb)
- [tensorflow2.0 - 基础2 - 索引切片;维度转换;数学运算](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/tensorflow/tensorflow2.0%20-%20%E5%9F%BA%E7%A1%802%20-%20%E7%B4%A2%E5%BC%95%E5%88%87%E7%89%87%3B%E7%BB%B4%E5%BA%A6%E8%BD%AC%E6%8D%A2%3B%E6%95%B0%E5%AD%A6%E8%BF%90%E7%AE%97.ipynb)
- [tensorflow2.0 - 神经网络构建的一般步骤](https://github.com/Zhouxiaonnan/machine-learning-notesandcode/blob/master/tensorflow/tensorflow2.0%20-%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E7%9A%84%E4%B8%80%E8%88%AC%E6%AD%A5%E9%AA%A4.ipynb)
