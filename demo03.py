from sklearn.datasets import fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

import pandas as pd


def knncls():
    """
    k近邻
    :return:None
    """
    df = pd.read_csv(r'D:\workeplace\datas\facebook-v-predicting-check-ins\train.csv')
    df = df.query("x>1.0 & x<1.25 & y>2.5 & y<2.75")
    time_data = pd.to_datetime(df['time'], unit='s')
    time_data = pd.DatetimeIndex(time_data)
    df.loc[:, 'hour'] = time_data.hour
    df.loc[:, 'day'] = time_data.day
    df.loc[:, 'weekday'] = time_data.weekday
    df = df.drop(['time'], axis=1)
    grouped = df.groupby(by='place_id').count()
    grouped = grouped[grouped['row_id'] > 3].reset_index()
    df = df[df['place_id'].isin(grouped['place_id'])]
    y = df['place_id']
    x = df.drop(['place_id'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    knn = KNeighborsClassifier()
    # 网格搜索
    param = {'n_neighbors': [3, 5, 9]}
    gc = GridSearchCV(knn, param_grid=param, cv=2)  # 2折验证
    gc.fit(x_test, y_test)
    print("在测试集上准确率：", gc.score(x_test, y_test))
    print("在交叉验证当中最好的结果：", gc.best_score_)
    print("选择最好的模型是：", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果：", gc.cv_results_)
    # knn.fit(x_train, y_train)
    # y_predict = knn.predict(x_test)
    # print(knn.score(x_test, y_test))


def knncls2():
    """
    K-近邻预测用户签到位置
    :return:None
    """
    data = pd.read_csv(r'D:\workeplace\datas\facebook-v-predicting-check-ins\train.csv')
    data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")
    time_value = pd.to_datetime(data['time'], unit='s')
    time_value = pd.DatetimeIndex(time_value)
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday
    data = data.drop(['time'], axis=1)
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    print("预测的目标签到位置为：", y_predict)
    print("预测的准确率:", knn.score(x_test, y_test))

    # # 构造一些参数的值进行搜索
    # param = {"n_neighbors": [3, 5, 10]}
    #
    # # 进行网格搜索
    # gc = GridSearchCV(knn, param_grid=param, cv=2)
    #
    # gc.fit(x_train, y_train)
    #
    # # 预测准确率
    # print("在测试集上准确率：", gc.score(x_test, y_test))
    #
    # print("在交叉验证当中最好的结果：", gc.best_score_)
    #
    # print("选择最好的模型是：", gc.best_estimator_)
    #
    # print("每个超参数每次交叉验证的结果：", gc.cv_results_)

    return None

def naviebayes():
    """
    朴素贝叶斯进行文本分类
    :return:None
    """
    news = fetch_20newsgroups(subset='all')
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    tf = TfidfVectorizer()
    # 以训练集中的词列表进行词的重要性统计
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    print("所有词", tf.get_feature_names())
    # print("训练集特征值", x_train.toarray())
    # 朴素贝叶斯算法
    mul = MultinomialNB(alpha=1.0)
    mul.fit(x_train, y_train)
    y_predict = mul.predict(x_test)
    print("预测值为", y_predict)
    print("准确率", mul.score(x_test, y_test))


def linear():
    """
    线性回归
    :return:
    """
    lb = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    # print(x_train, y_train)
    # 标准化处理(对于目标值也需要进行标准化处理)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))
    # 正规方程求解预测
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pre = lr.predict(x_test)
    # print("预测结果为：", lr.predict(x_test))
    # print("各个特征的权重值为：", lr.coef_)
    # print("预测结果：", std_y.inverse_transform(lr.predict(x_test))
    print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), std_y.inverse_transform(y_pre.reshape(-1, 1))))
    # 梯度下降求解预测
    sr = SGDRegressor()
    sr.fit(x_train, y_train)
    y_pre = sr.predict(x_test)
    print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), std_y.inverse_transform(y_pre.reshape(-1, 1))))
    # 岭回归 当出现过拟合时候采用l2正则化来降低权重值
    # r = Ridge()
    # r.fit(x_train, y_train)
    # joblib.dump(r, './models/test.pkl')  # 模型保存
    model = joblib.load('./models/test.pkl')
    y_pre = model.predict(x_test)
    print(y_pre)
    # y_pre = r.predict(x_test)
    print("岭回归的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), std_y.inverse_transform(y_pre.reshape(-1, 1))))


if __name__ == '__main__':
    # linear()
    import tensorflow as tf

    a = tf.constant(1.0)
    b = tf.constant(2.0)

    with tf.Session() as session:
        print(session.run(tf.add(a, b)))
