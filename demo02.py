from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

import jieba
import numpy as np


def dictvec():
    """
    字典数据抽取
    :return:None
    """
    # 实例化
    dic = DictVectorizer(sparse=False)
    data = dic.fit_transform([{'name': '小明', 'age': '20', 'sex': '男'}, {'name': '小美', 'age': 21, 'sex': '女'}, {'name': '小h', 'age': 25, 'sex': '男'}])
    print(data)
    print(dic.get_feature_names())
    return None


def textvec():
    """
    文本数据抽取
    :return:None
    """
    obj = CountVectorizer()
    # data = obj.fit_transform(["Life is is short, i like python!", "Life is too long, i dislike python"])
    par1 = " ".join(jieba.cut("Anaconda指的是一个开源的Python发行版本，其包含了conda、Python等180多个科学包及其依赖项。 \
    [1]  因为包含了大量的科学包，Anaconda 的下载文件比较大（约 531 MB），如果只需要某些包，或者需要节省带宽或存储空间，也可以使用Miniconda这个较小的发行版（仅包含conda和 Python）。"))
    par2 = " ".join(jieba.cut("Anaconda是一个开源的包、环境管理器，可以用于在同一个机器上安装不同版本的软件包及其依赖，并能够在不同的环境之间切换"))
    data = obj.fit_transform([par1, par2])
    print(obj.get_feature_names())
    print(data.toarray())

    return None


def tfidfvec():
    """
    中文特征值化（重要性）
    :return:None
    """
    obj = TfidfVectorizer()
    # data = obj.fit_transform(["Life is is short, i like python!", "Life is too long, i dislike python"])
    par1 = " ".join(jieba.cut("Anaconda指的是一个开源的Python发行版本，其包含了conda、Python等180多个科学包及其依赖项。 \
    [1]  因为包含了大量的科学包，Anaconda 的下载文件比较大（约 531 MB），如果只需要某些包，或者需要节省带宽或存储空间，也可以使用Miniconda这个较小的发行版（仅包含conda和 Python）。"))
    par2 = " ".join(jieba.cut("Anaconda是一个开源的包、环境管理器，可以用于在同一个机器上安装不同版本的软件包及其依赖，并能够在不同的环境之间切换"))
    data = obj.fit_transform([par1, par2])
    print(obj.get_feature_names())
    print(data.toarray())

    return None


def minmaxvec():
    """
    归一化
    算式: x' = (x-min)/(max-min)  x'' = x'*(mx-mi)+mi
    作为每一列， max为每一列的最大值，min为每一列的最小值，
    那么x'为最终结果， mx，mi分别为指定区间值（默认mx为1，mi为0）
    缺点：容易受异常值的影响
    :return:None
    """
    obj = MinMaxScaler(feature_range=(2, 3))
    data = obj.fit_transform([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(data)

    return None


def standvec():
    """
    算式：x' = (x-mean)/y   方差var = [(x1-mean)^2+(x2-mean)^2+...]/n(每个特征的样本数) 标准差y=根号var
    对于归一化：如果异常值影响了最大值或最小值则会直接影响最终结果值
    对于标准化：如果出现异常值，由于具有一定数据量，少量的异常值对于平均值的影响不大，方差改变较小
    :return:None
    """
    obj = StandardScaler()
    data = obj.fit_transform([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(data)
    return None


def dealna():
    """
    缺失值处理
    :return:None
    """
    obj = Imputer(missing_values='NaN', strategy='mean', axis=0)
    data = obj.fit_transform([[1, np.nan, 3], [4, np.nan, 6], [7, 8, 9]])
    print(data)
    return None

# 数据降维
def var():
    """
    特征选择：去除低方差的特征
    （方差越低，数据越相似就失去了作为特征的意义）
    :return:None
    """
    obj = VarianceThreshold()
    data = obj.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)
    return None


def pca():
    """
    主成分分析降维
    :return:None
    """
    obj = PCA(n_components=0.9)
    data = obj.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)
    return None


if __name__ == '__main__':
    from hashlib import md5
    data = 'hujie'
    print(md5(b'hujie').hexdigest())
