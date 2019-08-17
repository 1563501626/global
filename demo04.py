import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义一张新的图
g = tf.Graph()
c = tf.constant(1.0)
with g.as_default():
    d = tf.constant(0.1)
    print(c.graph)
    print(d.graph)

print("*********")
a = tf.constant(5.0)
b = tf.constant(6.0)
sum = tf.add(a, b)
graph = tf.get_default_graph()
print(graph)

# 在同一个session下只能运行一张图 tf.Session(graph=g)graph制定使用哪张图
# 交互式工具tf.InteractiveSession()
tf.InteractiveSession()
sum.eval()
print('------------')

# tf.placeholder() 占位符用于实时提供数据
plt = tf.placeholder(tf.float32, [2, 3])  # 参数一为数据类型， 参数二为数据形状  [None, 3] 为行数不确定可以为任意值列为3列

# 变量
var = tf.Variable(tf.random_normal([3, 4], mean=0.0, stddev=1.0))
# 需要添加一个初始化所有变量的op
init_op = tf.global_variables_initializer()
print(var)

# tf.Session(config=tf.ConfigProto(log_device_placement=True)) 输出这张图所有用到的资源
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    # 必须运行初始化op
    session.run(init_op)
    print(session.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 6]]}))
    # print(session.run(sum))
    print(sum.eval())
    print(a.graph)
    print(sum.graph)
    print(session.graph)

    # 把图写入事件文件中
    w = tf.summary.FileWriter(r'D:\workeplace\dataAnalysis\models\summary\test', graph=session.graph)
    print(w)