import tensorflow as tf
import numpy as np
import os


def get_files(file_dir):
    """
    输入： 存放训练照片的文件地址
    返回:  图像列表， 标签列表
    """
    # 建立空列表
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    # 读取标记好的图像和加入标签
    for file in os.listdir(file_dir):   # file就是要读取的照片
        name = file.split(sep='.')      # 因为照片的格式是cat.1.jpg/cat.2.jpg
        if name[0] == 'cat':            # 所以只用读取 . 前面这个字符串
            cats.append(file_dir + file)
            label_cats.append(0)        # 把图像和标签加入列表
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' % (len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))  # 在水平方向平铺合成一个行向量
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])  # 生成一个两行数组列表，大小是2 X 25000
    temp = temp.transpose()   # 转置向量，大小变成25000 X 2
    np.random.shuffle(temp)   # 乱序，打乱这25000个例子的顺序

    image_list = list(temp[:, 0])  # 所有行，列=0
    label_list = list(temp[:, 1])  # 所有行，列=1
    label_list = [int(float(i)) for i in label_list]  # 把标签列表转化为int类型

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    输入：
    image,label ：要生成batch的图像和标签
    image_W，image_H: 图像的宽度和高度
    batch_size: 每个batch（小批次）有多少张图片数据
    capacity: 队列的最大容量
    返回：
    image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
    label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
    # 将列表转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成队列(牵扯到线程概念，便于batch训练）
    """
    队列的理解：每次训练时，从队列中取一个batch送到网络进行训练，
               然后又有新的图片从训练库中注入队列，这样循环往复。
               队列相当于起到了训练库到网络模型间数据管道的作用，
               训练数据通过队列送入网络。
    """
    input_queue = tf.train.slice_input_producer([image, label])

    # 图像的读取需要tf.read_file()，标签则可以直接赋值
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)  # 解码彩色的.jpg图像
    label = input_queue[1]

    # 统一图片大小
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)  # 标准化图片，因为前两行代码已经处理过了，所以可要可不要

    # 打包batch的大小
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,  # 涉及到线程，配合队列
                                              capacity=capacity)

    # 下面两行代码应该也多余了，放在这里确保一下格式不会出问题
    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.cast(label_batch, tf.int32)

    return image_batch, label_batch
