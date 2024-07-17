import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
# 每张图片分辨率为28*28
IMAGE_SIZE = 28
# Mnist数据集为灰度图，故输入图片通道数NUM_CHANNELS取值为1
NUM_CHANNELS = 1
# 第一层卷积核大小为5
CONV1_SIZE = 5
# 卷积核个数为32
CONV1_KERNEL_NUM = 32
# 第二层卷积核大小为5
CONV2_SIZE = 5
# 卷积核个数为64
CONV2_KERNEL_NUM = 64
# 全连接层第一层为 512 个神经元
FC_SIZE = 512
# 全连接层第二层为 10 个神经元
OUTPUT_NODE = 10
# batch的数量
BATCH_SIZE = 1024
# 初始学习率
LEARNING_RATE_BASE = 0.005
# 学习率衰减率
LEARNING_RATE_DECAY = 0.99
# 正则化
REGULARIZER = 0.0001
# 最大迭代次数
STEPS = 100
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99
# 模型保存路径
MODEL_SAVE_PATH = "./model/"
# 模型名称
MODEL_NAME = "mnist_model"

# 权重w计算
def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
# 偏置b计算
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b
# 卷积层计算
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
# 最大池化层计算
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def forward(x, train, regularizer):
    # 实现第一层卷积
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    # 非线性激活
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 最大池化
    pool1 = max_pool_2x2(relu1)
    # 实现第二层卷积
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)
    # 获取一个张量的维度
    pool_shape = pool2.get_shape().as_list()
    # pool_shape[1] 为长 pool_shape[2] 为宽 pool_shape[3]为高
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 得到矩阵被拉长后的长度，pool_shape[0]为batch值
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    # 实现第三层全连接层
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 如果是训练阶段，则对该层输出使用dropout
    if train: fc1 = tf.nn.dropout(fc1, 0.5)
    # 实现第四层全连接层
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y 

def backward(mnist):
    # 卷积层输入为四阶张量
    # 第一阶表示每轮喂入的图片数量，第二阶和第三阶分别表示图片的行分辨率和列分辨率，第四阶表示通道数
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
    # 前向传播过程
    y = forward(x, True, REGULARIZER)
    # 声明一个全局计数器
    global_step = tf.Variable(0, trainable=False)
    # 对网络最后一层的输出y做softmax，求取输出属于某一类的概率
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 向量求均值
    cem = tf.reduce_mean(ce)
    # 正则化的损失值
    loss = cem + tf.add_n(tf.get_collection('losses'))
    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    # 梯度下降算法的优化器
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss, global_step=global_step)
    # train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32), name="accuracy")
    # 采用滑动平均的方法更新参数
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    # 将train_step和ema_op两个训练操作绑定到train_op上
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    # 实例化一个保存和恢复变量的saver
    saver = tf.train.Saver()
    # 创建一个会话
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 通过 checkpoint 文件定位到最新保存的模型，若文件存在，则加载最新的模型
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(STEPS):
            # 读取一个batch数据，将输入数据xs转成与网络输入相同形状的矩阵
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS))
            # 读取一个batch数据，将输入数据xs转成与网络输入相同形状的矩阵
            _, loss_value, step, acc_value = sess.run([train_op, loss, global_step, accuracy], feed_dict={x: reshaped_xs, y_: ys})
            if i % 1 == 0:
                print("After %d training step(s), loss on training batch is %g, accuracy on training %g" % (step, loss_value, acc_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
def main():
    mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)
    backward(mnist)
if __name__ == '__main__':
    main()

