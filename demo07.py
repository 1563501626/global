import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 简单神经网络识别手写数字图片
# mnist = input_data.read_data_sets(r'D:\workeplace\datas\mnist', one_hot=True)
# mnist.train.images  图片特征
# mnist.train.labels  图片目标值
# mnist.train.next_batch(50)  图片的批处理 一次输出50张图

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("is_train", 1, "aa")


def easyShenJin():
    """
    简单神经网络
    :return:  None
    """
    mnist = input_data.read_data_sets(r'D:\workeplace\datas\mnist', one_hot=True)  # 数据集

    # 1、建立数据占位符 特征值(采用placeholder一次输入图片数量动态更改) [None, 784]   目标值 [None, 10]
    with tf.variable_scope("data"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')

        y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_true')

    # 2、建立一个全连接层的神经网络  权重w shape:[748, 10]  偏置b shape:[10]
    with tf.variable_scope("fc_model"):
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name='w')

        bias = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), name='b')

        y_predict = tf.matmul(x, weight) + bias

    # 3、计算损失  交叉损失熵  labels=None(真实值), logits=None(预测值), 返回损失值列表并求出平均值
    with tf.variable_scope("loss"):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)

        loss = tf.reduce_mean(loss)

    # 4、优化  梯度下降优化  指定学习率为0.1
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    # 5、计算准确率  将全连接层输出的最大概率的位置的预测值与相同位置的对应的真实值进行比较
    # def equal(x, y, name=None) Returns the truth value of (x == y) element-wise.  x:真实值 y:预测值
    # 计算 equal_list的平均值
    # def argmax(input,
    #            axis=None,
    #            name=None,
    #            dimension=None,
    #            output_type=dtypes.int64)  给出最大值的位置
    # tf.cast  Casts a tensor to a new type
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    var_init = tf.global_variables_initializer()

    # 收集变量（图里显示） 一维变量
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

    # 收集高维变量
    tf.summary.histogram("weight", weight)
    tf.summary.histogram("bias", bias)

    # 创建合并变量的Op
    merged = tf.summary.merge_all()

    # 创建saver
    saver = tf.train.Saver()

    # 开启会话
    with tf.Session() as sess:
        sess.run(var_init)

        # 创建events文件并写入
        filewriter = tf.summary.FileWriter("./models/summary/test", graph=sess.graph)

        if FLAGS.is_train == 1:
            # 迭代训练
            for i in range(2000):
                # 取出数据
                mnist_x, mnist_y = mnist.train.next_batch(50)
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

                # 写入每步执行的值
                summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
                filewriter.add_summary(summary, i)

                print("第%d次训练准确率为：%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))

            # 保存模型
            saver.save(sess, "./models/shenjing_01/")
        else:
            # 加载模型
            saver.restore(sess, "./models/shenjing_01/")

            # 进行预测  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(100):
                mnist_x, mnist_y = mnist.test.next_batch(1)

                print("第%d次预测，预测值为：%d，真实值为：%d" % (
                    i,
                    tf.argmax(sess.run(y_predict, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval(),
                    tf.argmax(mnist_y, 1).eval()
                ))

    return None


if __name__ == '__main__':
    easyShenJin()
