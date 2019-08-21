import os

import tensorflow as tf


def testfile():
    """
    读取文件
    :return:
    """
    file_list = [os.path.join('./files', i) for i in os.listdir('./files')]
    # 构造文件队列
    q = tf.train.string_input_producer(file_list)
    # 构造csv文件阅读器读取队列数据 一次读取一行
    reader = tf.TextLineReader()
    key, value = reader.read(q)  # key：文件名  value：文件类容
    # 对每行内容解码 record_defaults指定每一行的类型以及默认值
    first_line, second_line = tf.decode_csv(value, record_defaults=[['nan'], ['nan']])
    return first_line, second_line


if __name__ == '__main__':
    first_line, second_line = testfile()
    with tf.Session() as sess:
        print(sess.run([first_line, second_line]))