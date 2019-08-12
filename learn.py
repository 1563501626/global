import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier


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
    df = pd.read_csv("titanic.csv")
    x = df.loc[:, ['pclass', 'age', 'sex']]
    y = df.loc[:, 'survived']
    x.age = x.age.fillna(x.age.mean())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    dicts = DictVectorizer(sparse=False)
    x_train = dicts.fit_transform(x_train.to_dict(orient='record'))
    print(x_train, '\n', dicts.get_feature_names())
    x_test = dicts.transform(x_test.to_dict(orient='record'))
    # 决策树
    deci = DecisionTreeClassifier(max_depth=5)
    deci.fit(x_train, y_train)
    print(deci.score(x_test, y_test))


if __name__ == '__main__':
    decision()