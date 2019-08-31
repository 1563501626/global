import tensorflow as tf


# 验证码数据训练

TFRECORDS_DIR = r'./train.tfrecords'
BATCH_SIZE = 50


def read_data():
    """
    读取image、label API
    :return: image_batch label_batch
    """
    # 构造文件队列
    q = tf.train.string_input_producer([TFRECORDS_DIR])

    # 构造阅读器  tfrecord文件
    reader = tf.TFRecordReader()

    # 读取
    key, value = reader.read(q)

    # 解析、解码
    feature = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.string)
    })
    # 特征值 bytes --> uint8
    image = tf.decode_raw(feature["image"], tf.uint8)
    # 真实值
    label = tf.decode_raw(feature["label"], tf.uint8)

    # 批处理 x [?, 32, 90, 3] y [?, 3]
    x = tf.reshape(image, [32, 90, 3])
    y_true = tf.reshape(label, [5])

    print(x)
    print(y_true)

    image_batch, label_batch = tf.train.batch([x, y_true], batch_size=BATCH_SIZE, num_threads=1, capacity=BATCH_SIZE)
    # print(image_batch, label_batch)

    return image_batch, label_batch


def fc_shenjing(image_batch, label_batch):
    """
    训练 API
    image_batch: [50, 32, 90, 3],label_batch: [50, 3]
    """
    # label --> one_hot
    y_true = tf.one_hot(label_batch, depth=14, axis=2, on_value=1.0)

    # x: [50, 32*90*3] y_true: [50, 3] w: [32*90*3, 100+4+100] b: [100+4+100]
    x = tf.reshape(image_batch, [-1, 32*90*3])
    w = tf.Variable(tf.random_normal([32*90*3, 14*5]))
    b = tf.constant(0.0, shape=[14*5])

    # 矩阵计算
    y_predict = tf.matmul(tf.cast(x, tf.float32), w) + b

    # softmax、交叉熵损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(y_true, [BATCH_SIZE, 14*5]),
        logits=y_predict
    ))

    # 梯度下降
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)

    # 计算准确率
    # 要将y_predict[50, 3*104] --> [50, 3, 104]
    equal_list = tf.equal(tf.argmax(y_true, axis=2), tf.argmax(tf.reshape(y_predict, [BATCH_SIZE, 5, 14]), axis=2))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    var_init = tf.global_variables_initializer()

    return y_predict, var_init, train_op, accuracy, loss


def main():
    """
    主要逻辑
    """
    # 读取数据
    image_batch, label_batch = read_data()

    # 全连接层神经网络
    y_predict, var_init, train_op, accuracy, loss = fc_shenjing(image_batch, label_batch)

    with tf.Session() as sess:
        sess.run(var_init)

        # 定义线程协调器和开启线程（有数据在文件当中读取提供给模型）
        coord = tf.train.Coordinator()

        # 开启线程去运行读取文件操作
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 训练识别程序
        for i in range(5000):
            sess.run(train_op)

            print("第%d批次的准确率为：%f" % (i, accuracy.eval()))

        # 回收线程
        coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    main()
