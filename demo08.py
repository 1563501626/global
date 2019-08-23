import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 卷积神经网络识别手写数字图

def weight_variable(shape):
    """
    权重变量
    """
    w = tf.Variable(tf.random_normal(shape=shape))
    return w


def bias_variable(shape):
    """
    偏置变量
    """
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def model():
    """
    卷积神经网络模型
    :return: None
    """
    # x、y_true 占位符 x:[None, 784] y_true:[None, 10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])

    # 第一层卷积  卷积 激活 池化
    # conv2d  input:[None, 28, 28, 1], filter:[5, 5, 1, 32], strides:[1, 1, 1, 1], padding:`SAME`,
    with tf.variable_scope("conv1"):
        w1 = weight_variable([5, 5, 1, 32])
        b1 = bias_variable([32])
        x_reshaped = tf.reshape(x, [-1, 28, 28, 1])

        # [None, 28, 28, 1] ----> [None, 28, 28, 32]  卷积+激活函数处理
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshaped, w1, [1, 1, 1, 1], padding='SAME') + b1)

        # 池化 value, ksize:[1, ksize, ksize, 1], strides:[1, strides, strides, 1], padding,
        # [None, 28, 28, 32] ----> [None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第二层卷积
    with tf.variable_scope("conv2"):
        # 现在有32张表（一张图通过第一步filter计算有32个结果） filter:64 size:5*5
        w2 = weight_variable([5, 5, 32, 64])
        b2 = bias_variable([64])

        # 卷积+激活函数处理  [None, 14, 14, 32] ----> [None, 14, 14, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w2, [1, 1, 1, 1], padding='SAME')) + b2

        # 池化 [None, 14, 14, 64] ----> [None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 全连接层
    # [None, 7*7*64] * [7*7*64, 10] + [10] = [None, 10]
    with tf.variable_scope("full_connect"):
        w3 = weight_variable([7*7*64, 10])
        b3 = bias_variable([10])
        x3 = tf.reshape(x_pool2, shape=[-1, 7*7*64])

        y_predict = tf.matmul(x3, w3) + b3

    # 计算损失  交叉损失熵
    with tf.variable_scope("loss"):
        loss_li = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
        loss = tf.reduce_mean(loss_li)

    # 优化调参  梯度下降
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.0025).minimize(loss)

    # 计算准确率
    with tf.variable_scope("accuracy"):
        equal_li = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_li, tf.float32))

    # 初始化变量op
    var_init = tf.global_variables_initializer()

    return train_op, var_init, x, y_true, accuracy, loss


def conv_fc():
    # 获取数据集
    mnist = input_data.read_data_sets(r'D:\crawl_datasource\yzm\data', one_hot=True)

    # 获取训练op
    train_op, var_init, x, y_true, accuracy, loss = model()

    # 开始训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(var_init)

        # 迭代训练
        for i in range(1000):
            # 批处理
            x_mnist, y_mnist = mnist.train.next_batch(50)
            sess.run(train_op, feed_dict={x: x_mnist, y_true: y_mnist})
            print("第%d次训练，准确率为：%f" % (i, sess.run(accuracy, feed_dict={x: x_mnist, y_true: y_mnist})))
            print(sess.run(loss, feed_dict={x: x_mnist, y_true: y_mnist}))


if __name__ == '__main__':
    conv_fc()
