import tensorflow as tf
import os, re


# 数据输入

IMAGE_PATH = r'D:\workeplace\datas\train_data\caption\image'
LABEL_PATH = r'D:\workeplace\datas\train_data\caption\text'
TFRECORDS_SAVE_PATH = r'D:\workeplace\dataAnalysis\models\tfrecords\\'


def image():
    """
    获取验证码图片数据
    :return:
    """
    # 存储文件名
    file_name = []

    for i in range(1, 501):
        file_name.append(str(i) + '.jpg')

    # 构造文件路径+文件名
    file_list = [os.path.join(IMAGE_PATH, file) for file in file_name]

    # 构造文件队列 shuffle=False 不打乱顺序
    file_queue = tf.train.string_input_producer(file_list, shuffle=False)

    # 构造阅读器
    reader = tf.WholeFileReader()

    # 读取图片数据内容  key:文件名 value:图片数据
    key, value = reader.read(file_queue)

    # 解码
    image = tf.image.decode_jpeg(value)

    # 32 * 90 * 3
    image.set_shape([32, 90, 3])

    # 批处理  capacity 队列最多可存储的样例数 [500, 32, 90, 3]
    image_batch = tf.train.batch([image], batch_size=500, capacity=500, num_threads=1)

    return image_batch


def label():
    """
    获取图片标签数据
    :return:
    """
    # 构造文件队列
    file_queue = tf.train.string_input_producer([LABEL_PATH], shuffle=False)

    # 构造阅读器
    reader = tf.TextLineReader()

    # 读取内容 一次读取一行
    key, value = reader.read(file_queue)

    # 解码 records=[['None']] 读取出来是字符串形式
    file = tf.decode_csv(value, record_defaults=['None'])

    # 批处理 [500, 1]
    file_batch = tf.train.batch([file], batch_size=500, capacity=500, num_threads=1)

    return file_batch


def deal_label(labels):
    """
    标签值数值化 1 ÷ 1
    :return:
    """
    # LETTER = "0123456789+-×÷"
    #
    # # {...0: '+', 1: '-', 2: '×', 3: '÷'}
    # letter = dict(enumerate(LETTER))
    # # {...'+': 0, '-': 1, '×': 2, '÷': 3}
    # letter = dict(zip(letter.values(), letter.keys()))

    temp = []
    for i in range(100):
        temp.append(str(i))
    for i in "+-×÷":
        temp.append(i)
    # {...0: '+', 1: '-', 2: '×', 3: '÷'}
    letter = dict(enumerate(temp))
    # {...'+': 0, '-': 1, '×': 2, '÷': 3}
    letter = dict(zip(letter.values(), letter.keys()))

    # 储存值化后的标签 [[1,3,6], [2,5,2]...]
    label_letter = []

    # 数值化
    for i in labels:
        string_i = i[0].decode()
        num = re.search(r'(\d+)([\+\-×÷]+)(\d+)', string_i)
        first = num.group(1)
        second = num.group(2)
        third = num.group(3)
        label_letter.append([letter[first], letter[second], letter[third]])

    print(label_letter)

    # 将列表转换为tensor
    label_letter = tf.constant(label_letter)

    return label_letter


def write_to_tfrecords(image_batch, label_batch):
    """
    写入tfrecords文件
    :return:
    """
    # 转换类型
    label_batch = tf.cast(label_batch, tf.uint8)
    print(label_batch)

    # 构造TFRecords存储器 path tfrecords文件存储地址
    writer = tf.python_io.TFRecordWriter(path=TFRECORDS_SAVE_PATH + 'train.tfrecords')

    # 循环将每一张图片序列化后写入
    for i in range(1, 501):
        # 取出第i张图片， 将其特征值转换为string
        image_str = image_batch[i].eval().tostring()

        # 标签值转换为string
        label_str = label_batch[i].eval().tostring()

        # 构造协议块
        example = tf.train.Example(features=tf.train.Feature(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_str])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_str]))
        }))
        writer.write(example.SerializeToString())
        writer.close()

    return None


def caption():
    # 获取文件当中的数据 (x)
    image_batch = image()

    # 获取验证码中的标签数据（y_true）
    label_batch = label()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        label_banary = sess.run(label_batch)

        label_batch = deal_label(label_banary)

        # 数值化后
        label_batchs = sess.run(label_batch)

        print(label_batchs)

        # 将图片数据和标签值写入tfrecords文件中
        write_to_tfrecords(image_batch, label_batchs)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    caption()
