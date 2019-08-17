import tensorflow as tf

# 实现一元一次线性回归


def linrear():
    """
    线性回归预测
    :return: None
    """
    with tf.variable_scope("data"):
        # 特征值
        x = tf.random_normal([100, 1], mean=2.0, stddev=0.5, name='x')
        # 指定权重0.7， 偏置0.8
        y_true = tf.matmul(x, [[0.7]]) + 0.8  # 忽略本行
    with tf.variable_scope("model"):
        # 指定随机的权重值 矩阵必须为二维 1个特征对应一个权重和一个偏置
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=0.5), name='w')
        bias = tf.Variable(0.0, name='b')
        # 矩阵相乘 预测目标值
        y_predict = tf.matmul(x, weight) + bias
    with tf.variable_scope("loss"):
        # 计算相应的均方误差 [(x_1-x_1`)^2+(x_1-x_1`)^2+...+(x_n-x_n`)^2]/n x_1:实际值 x_1`:预测值
        loss = tf.reduce_mean(tf.square(y_true - y_predict))
    with tf.variable_scope("optimizer"):
        # 采用梯度下降来调优参数 learn_rate: 0-1
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    init_op = tf.global_variables_initializer()  # 这一行最好放在最后欧一行，如果放在tf.Variable之前就会报错
    # 收集tensor
    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weights", weight)
    merged = tf.summary.merge_all()
    with tf.Session() as session:
        # 初始化变量op
        session.run(init_op)
        # 打印初始化的权重和偏置 op必须eval()后才能打印
        print("初始化权重为%f，偏置为%f" % (weight.eval(), bias.eval()))
        file = tf.summary.FileWriter("./models/summary/test", graph=session.graph)
        # 循环运行优化
        for i in range(1, 400):
            session.run(train_op)
            print("第%d次优化 权重为%f，偏置为%f" % (i, weight.eval(), bias.eval()))
            summary = session.run(merged)
            file.add_summary(summary, i)

    return None


def testqueue():
    """
    管道
    :return:
    """
    # 定义queue
    q = tf.FIFOQueue(3, tf.float32)

    # 存入数据
    enq_many = q.enqueue_many([[1, 2, 3], ])
    # 取出数据做+1操作在存入数据
    out_q = q.dequeue()
    data = out_q + 1  # 运算符的重载
    en_q = q.enqueue(data)

    with tf.Session() as session:
        # 初始化队列
        session.run(enq_many)
        # 处理数据 tensorflow运算具有依赖性
        for i in range(100):
            session.run(en_q)
        # 训练数据
        for i in range(q.size().eval()):
            print(session.run(out_q))


def testthread():
    """
    多线程
    :return:
    """
    # 初始化管道
    q = tf.FIFOQueue(100, tf.float32)
    # 创建变量接受数据
    var = tf.Variable(0.0, tf.float32)
    # 实现自增
    data = tf.assign_add(var, tf.constant(1.0))
    en_q = q.enqueue(data)
    # 定义队列管理器op 多少个线程？ 做什么？
    qr = tf.train.QueueRunner(q, enqueue_ops=[en_q] * 2)  # 定义两个线程
    #初始化变量op
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        # 初始化变量op
        session.run(init_op)
        # 开启线程管理器
        coord = tf.train.Coordinator()
        # 开启线程
        threads = qr.create_threads(session, coord=coord, start=True)
        # 主线程读取数据模拟训练
        for i in range(300):
            print(session.run(q.dequeue()))
        # 回收线程
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    testthread()
