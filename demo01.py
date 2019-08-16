import random
import string

import numpy as np
import pandas as pd

# a = np.zeros((3, 4))
# a = np.empty((3, 4))  # 随机的数字
# a = np.arange(12).reshape((3, 4))
# a = np.linspace(1, 10, 6).reshape(2, 3)  # 平均将1-9之间的数字分为6个
# a = np.arange(12, 0, -1).reshape((3, 4))
# b = np.array([1, 2, 3, 4])
# print(a+b**2)
# print(a == b)
# a = np.random.random((2, 4))  # 2行4列
# print(np.min(a, axis=0))  # 轴是反的
# print(np.nonzero(a))  # 返回非0的数字的索引
# print(np.sort(a, axis=0))  # 默认按行从小到大排序（坑爹的轴是反的）
# print(np.transpose(a))  # 按对称轴翻转（行变列， 列变行）
# print(a.T)  # 同上
# print(np.clip(a, 5, 9))  # 大于9的数等于9小于5的数等于5（指标刘5-9之间的数）
# print(a)
# print('***************')
# print(a[0, 1:])
# for row in a:
#     print(row)  # 返回每一行
# for col in a.T:
#     print(col)  # 返回每一列
# print(a.flatten())  # 展开为一行
# for i in a.flat:
#     print(i)  # 返回每一个数（a.flat为迭代器）
# a = np.array([1, 1, 1])
# b = np.array([2, 2, 2])
# print(np.vstack((a, b)))  # 竖直拼接（列数不变）
# print(np.hstack((a, b)))  # 水平拼接（行数不变）
# print(np.concatenate((a, b), axis=0))  # 同上
# print(np.split(a, 2, axis=0))  # 对列进行分割， 每一列分割为2部分（行不受影响， 如果行数不是2的倍数会报错）
# print(np.array_split(a, 2, axis=0))  # 不等量分割， 上面会报错但本次操作不会报错
# print(np.hsplit(a, 2))  # ...
# b = a.copy()  # b是a的深拷贝

# __________________________________________________________________________________
# df = pd.Series([1, np.nan, 2, 'a'])
# dates = pd.date_range('20190728', periods=6)
# df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])  # 生成6行4列的数组
# df = pd.DataFrame(np.arange(12).reshape((3, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
# df = pd.DataFrame({'A': 1, 'B': pd.Timestamp('20190728'), 'C': pd.Series(1, index=list(range(4)), dtype='float64'),
# 'D': np.array([3]*4, dtype='int32'), 'E': pd.Categorical(['test', 'train', 'test', 'train']), 'F':'fool'})
# df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
# print(df.sort_index(axis=1, ascending=False))  # 对列索引排序（ascending=False为逆序）
# print(df.A, df['A'])
# print(df['20190728': '20190801'])
# print(df.loc['20190728', 'A'])
# print(df.loc[:, ['A', 'C']])
# print(df.iloc[[0, 3], [1, 3]])
# print(df>8)
# print(df[df['A'] > 8])
# df.columns = list('哈或黑哼')
# print(df)
# print(df.哈)
# print(df[df.loc[:,'A']>8]['A'])

# __________________________________________________________________________________
# 数据的离散化
# df = pd.DataFrame(np.array(['%s,%s,%s' % (random.choice('abcd'), random.choice('abcd'), random.choice\
#     ('abcd')) for i in range(100)]).reshape(100, 1), index=[i for i in range(100)], columns=['A'])
# zeros_df = pd.DataFrame(np.zeros((df.shape[0], len('abcd'))), columns=list('abcd'))
# for i in range(df.shape[0]):
#     for j in df.loc[i, 'A'].split(','):
#         zeros_df.loc[i, j] += 1
# print('a:%s, b:%s, c:%s, d:%s' %(zeros_df['a'].sum(), zeros_df['b'].sum(), zeros_df['c'].sum(), zeros_df['d'].sum()))

# __________________________________________________________________________________
# 数据的合并
# df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=[1, 2, 3], columns=['a', 'b', 'c'])
# df2 = pd.DataFrame([[10, 2, 30], [40, 50, 60], [70, 80, 90]], index=[1, 2, 3], columns=['a', 'b', 'c'])
# print(df1.join(df2))
# print(df2)

# __________________________________________________________________________________
# 数据分组
# df = pd.DataFrame(np.array(['%s' % 'abcdef'[i % 6] for i in range(400)]).reshape(100, 4), index=[i for i in range(100)
# ],columns=['A', 'B', 'C', 'D'])
# grouped = df.groupby(by='A')
# print(grouped.count())

# __________________________________________________________________________________
# 数据分组聚合
df = pd.DataFrame(np.array(['%s' % 'abcdef'[i % 6] for i in range(400)]).reshape(100, 4), index=[i for i in range(100)],
                  columns=['A', 'B', 'C', 'D'])
# print(df[df['A']=='b'])
grouped = df.groupby(by=[df['A'], df['B']])
for i, j in grouped:
    print(j)

    