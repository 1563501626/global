import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups, load_boston
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


def kneighbors():
    """
    k近邻
    :return:
    """
    df = pd.read_csv(r'D:\crawl_datasource\facebook-v-predicting-check-ins\train.csv').query("x > 1.0 & x < 2.25 & y > 2.5 & y < 3.75")
    time_value = pd.to_datetime(df['time'], unit='s')
    time_value = pd.DatetimeIndex(time_value)
    df.loc[:, 'day'] = time_value.day
    df.loc[:, 'hour'] = time_value.hour
    df.loc[:, 'weekday'] = time_value.weekday
    df = df.drop(['time'], axis=1)
    place_count = df.groupby('place_id', as_index=False).count()
    place_count = place_count[place_count['row_id'] > 3]
    df = df[df['place_id'].isin(place_count['place_id'])]
    df = df.drop('row_id', axis=1)
    # 提取特征值和目标值
    x = df.drop('place_id', axis=1)
    y = df['place_id']
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # k-近邻
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    # 预测结果
    y_pre = knn.predict(x_test)
    # 准确率
    score = knn.score(x_test, y_test)
    print(score)


def mul():
    """
    朴素贝叶斯
    :return:
    """
    news = fetch_20newsgroups(subset='all')
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    # 词的重要性
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    print(x_train.toarray())
    x_test = tf.transform(x_test)
    # 朴素贝叶斯
    nb = MultinomialNB(alpha=1.0)
    nb.fit(x_train, y_train)
    y_pre = nb.predict(x_test)
    print('预测结果为',y_pre)
    print('准确率', nb.score(x_test, y_test))


# 信息熵与不确定性相关， 信息熵越大越不确定 信息熵为：H(x):对[p(x)logp(x)]求和
# 信息增益：在得知一个特征之后，对信息熵减少程度的大小
# 决策树的分类依据之一为：信息增益
def decision():
    """
    决策树
    :return:
    """
    df = pd.read_csv(r"C:\Users\Administrator\Desktop\titanic.txt")
    x = df.loc[:, ['pclass', 'age', 'sex']]
    y = df.loc[:, 'survived']
    x.age = x.age.fillna(x.age.mean())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    dicts = DictVectorizer(sparse=False)
    x_train = dicts.fit_transform(x_train.to_dict(orient='record'))
    print(x_train, '\n', dicts.get_feature_names())
    x_test = dicts.transform(x_test.to_dict(orient='record'))
    # 决策树
    deci = DecisionTreeClassifier()
    deci.fit(x_train, y_train)
    print(deci.score(x_test, y_test))


def forest():
    """
    随机森林
    :return:
    """
    df = pd.read_csv(r"C:\Users\Administrator\Desktop\titanic.txt")
    x = df.loc[:, ['pclass', 'age', 'sex']]
    y = df.loc[:, 'survived']
    x.age = x.age.fillna(x.age.mean())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    dicts = DictVectorizer(sparse=False)
    x_train = dicts.fit_transform(x_train.to_dict(orient='record'))
    print(x_train, '\n', dicts.get_feature_names())
    x_test = dicts.transform(x_test.to_dict(orient='record'))
    # 随机森林 (超参数调优) n_estimators决策树的最大棵数
    rf = RandomForestClassifier()
    # 网格搜索和交叉验证
    params = {"n_estimators": [10,200,300,500,800,1200], 'max_depth': [5,8,15,25,30]}
    gc = GridSearchCV(rf, param_grid=params)
    gc.fit(x_train, y_train)
    print(gc.score(x_test, y_test))
    print('最佳参数模型为', gc.best_params_)


def lineas():
    """
    线性回归
    :return:
    """
    lb = load_boston()
    x_train, x_test, y_train, y_test= train_test_split(lb.data, lb.target, test_size=0.25)
    # 标准化处理
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))  # 需要将一维数组转换为二维数组
    y_test = std_y.transform(y_test.reshape(-1, 1))
    # 线性回归
    # 正规方程
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pre = lr.predict(x_test)
    y_pre = std_y.inverse_transform(y_pre)
    # print("result", y_pre)
    # print("权重", lr.coef_)
    print("均方误差1", mean_squared_error(std_y.inverse_transform(y_test), y_pre))

    # 梯度下降
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    y_pre2 = std_y.inverse_transform(sgd.predict(x_test))
    # print('权重', sgd.coef_)
    print("均方误差2", mean_squared_error(std_y.inverse_transform(y_test), y_pre2))


def logistic():
    """
    逻辑回归
    :return:
    """
    columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
               'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv('D:\crawl_datasource\cancer.data', names=columns)
    data = data.replace(to_replace=['?'], value=np.nan)
    data = data.dropna()
    x_train, x_test, y_train, y_test = train_test_split(data[columns[1:-1]], data[columns[-1]])
    # 标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # 逻辑回归
    lg = LogisticRegression(penalty='l2', C=1.0)
    lg.fit(x_train, y_train)
    y_pre = lg.predict(x_test)
    print(y_pre)
    print("准确率", lg.score(x_test, y_test))
    print("召回率", classification_report(np.array(y_test).reshape(-1, 1), y_pre.reshape(-1, 1), labels=[2, 4], target_names=['良性', '恶性']))


if __name__ == '__main__':
    logistic()