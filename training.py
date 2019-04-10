import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import input_data
import model


N_CLASSES = 2  # 猫和狗
IMG_W = 208  # resize图像，太大的话训练时间久
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000  # 一般5K~10k
learning_rate = 0.0001  # 一般小于0.0001

train_dir = 'D:/python/deep-learning/CatVsDog/Project/data/train/'
logs_train_dir = 'D:/python/deep-learning/CatVsDog/Project/log/'  # 记录训练过程与保存模型

train, train_label = input_data.get_files(train_dir)
train_batch, train_label_batch = input_data.get_batch(train,
                                                      train_label,
                                                      IMG_W,
                                                      IMG_H,
                                                      BATCH_SIZE,
                                                      CAPACITY)

train_logits = model.cnn_inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)
train_op = model.training(train_loss, learning_rate)
train__acc = model.evaluation(train_logits, train_label_batch)

summary_op = tf.summary.merge_all()  # 这个是log汇总记录

# 可视化为了画折线图
step_list = list(range(100))  # 因为后来的cnn_list加了200个
cnn_list1 = []
cnn_list2 = []
fig = plt.figure()  # 建立可视化图像框
ax = fig.add_subplot(1, 1, 1)  # 子图总行数、列数，位置
ax.yaxis.grid(True)
ax.set_title('cnn_accuracy ', fontsize=14, y=1.02)
ax.set_xlabel('step')
ax.set_ylabel('accuracy')
bx = fig.add_subplot(1, 2, 2)
bx.yaxis.grid(True)
bx.set_title('cnn_loss ', fontsize=14, y=1.02)
bx.set_xlabel('step')
bx.set_ylabel('loss')


# 初始化，如果存在变量则是必不可少的操作
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 产生一个writer来写log文件
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    # 产生一个saver来存储训练好的模型
    saver = tf.train.Saver()

    # 队列监控
    # batch训练法用到了队列，不想用队列也可以用placeholder
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # 执行MAX_STEP步的训练，一步一个batch
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            # 启动以下操作节点，这里不能用train_op，因为它在第二次迭代是None，会导致session出错，改为_
            _op, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
            # 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            # 每隔100步画个图
            if step % 100 ==0:
                cnn_list1.append(tra_acc)
                cnn_list2.append(tra_loss)
            # 每隔5000步，保存一次训练好的模型
            if step % 5000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        ax.plot(step_list, cnn_list1)
        bx.plot(step_list, cnn_list2)
        plt.show()

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()


