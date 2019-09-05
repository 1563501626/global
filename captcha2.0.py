import tensorflow as tf
from PIL import Image
import numpy as np
import random
import os


class TrainError(Exception):
    pass


class TrainCap:
    def __init__(self, file_path, file_width, file_height):
        self.train_images_list = list(map(lambda x: os.path.join(file_path, x), os.listdir(file_path)))
        self.file_width = file_width
        self.file_height = file_height
        self.max_captcha = 4
        self.char_set_len = 26
        self.cap_text = 'abcdefghijklmnopqrstuvwxyz'

    @staticmethod
    def gen_data(img_path):
        # 标签
        label = img_path.split('_')[1].split('.')[0]

        # 图片
        captcha_img = Image.open(img_path)
        captcha_array = np.array(captcha_img)
        return label, captcha_array

    @staticmethod
    def convert2gray(img):
        """
        图片灰度化
        :param img:
        :return:
        """
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    def text2vec(self, label):
        """
        转换标签为one_hot编码
        :param label:
        :return:
        """
        vector = np.zeros(self.max_captcha * self.char_set_len)

        for i, v in enumerate(label):
            ids = i * self.char_set_len + self.cap_text.index(v)
            vector[ids] = 1
        return vector

    def get_batch1(self, n, size=50):
        batch_x = np.zeros([size, self.file_height * self.file_width])  # 初始化
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])  # 初始化

        max_batch = int(len(self.train_images_list) / size)
        # print(max_batch)
        if max_batch - 1 < 0:
            raise TrainError("训练集图片数量需要大于每批次训练的图片数量")
        if n > max_batch - 1:
            n = n % max_batch
        s = n * size
        e = (n + 1) * size
        this_batch = self.train_images_list[s:e]
        # print("{}:{}".format(s, e))

        for i, img_path in enumerate(this_batch):
            label, image_array = self.gen_data(img_path)
            image_array = self.convert2gray(image_array)  # 灰度化图片
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.text2vec(label)  # 生成 oneHot
        return batch_x, batch_y

    def get_batch(self, size=50):
        batch_x = np.zeros([size, self.file_height * self.file_width])  # 初始化
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])  # 初始化

        max_batch = int(len(self.train_images_list) / size)
        # print(max_batch)
        if max_batch - 1 < 0:
            raise TrainError("训练集图片数量需要大于每批次训练的图片数量")

        this_batch = random.sample(self.train_images_list, size)

        for i, img_path in enumerate(this_batch):
            label, image_array = self.gen_data(img_path)
            image_array = self.convert2gray(image_array)  # 灰度化图片
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.text2vec(label)  # 生成 oneHot
        return batch_x, batch_y


class CNN:
    def __init__(self, file_width, file_height, train_cap):
        self.file_width = file_width
        self.file_height = file_height
        self.max_captcha = 4
        self.char_set_len = 26
        self.train_capt = train_cap

    def cnn_train(self):
        """
        全连接层神经网络
        :return:
        """
        x_train = tf.placeholder(tf.float32, [None, self.file_width * self.file_height])
        y_true = tf.placeholder(tf.float32, [None, self.max_captcha * self.char_set_len])

        weight = tf.Variable(
            tf.random_normal([self.file_width * self.file_height, self.max_captcha * self.char_set_len]), name='w')
        bias = tf.constant(0.0, tf.float32, [self.max_captcha * self.char_set_len], name='b')

        y_predict = tf.add(tf.matmul(x_train, weight), bias)

        # 计算交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

        # 梯度下降优化损失
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

        # 计算准确率
        # loss_rate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true), tf.argmax(y_predict)), tf.float32))
        equal_li = tf.equal(tf.argmax(y_true), tf.argmax(y_predict))
        loss_rate = tf.reduce_mean(tf.cast(equal_li, tf.float32))

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            for i in range(1000):
                x_batch, y_batch = self.train_capt.get_batch()
                sess.run(train_op, feed_dict={x_train: x_batch, y_true: y_batch})
                print("第%s次训练，准确率为：%s" % (i, sess.run(loss_rate, feed_dict={x_train: x_batch, y_true: y_batch})))


if __name__ == '__main__':
    # file_path, file_width, file_height, train_cap
    file_paths = 'train/'
    file_widths = 100
    file_heights = 30
    train_caps = TrainCap(file_paths, file_widths, file_heights)

    cnn = CNN(file_widths, file_heights, train_caps)
    cnn.cnn_train()
