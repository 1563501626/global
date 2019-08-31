import tensorflow as tf
from PIL import Image
import numpy as np


class TrainCap:
    def __init__(self, file_path, file_width, file_height, label_path):
        self.file_li = [file_path.format(str(j)) for j in range(500)]
        self.file_width = file_width
        self.file_height = file_height
        self.label_path = label_path

    def get_text(self, img_path):
        captcha_img = Image.open(img_path)
        captcha_array = np.array(captcha_img)
        return captcha_array

    def get_batch(self, n, size=50):
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

        for i, img_name in enumerate(this_batch):
            label, image_array = self.gen_captcha_text_image(self.train_img_path, img_name)
            image_array = self.convert2gray(image_array)  # 灰度化图片
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.text2vec(label)  # 生成 oneHot
        return batch_x, batch_y