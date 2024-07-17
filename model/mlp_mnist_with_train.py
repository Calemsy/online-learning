# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import numpy as np

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, [X_train.shape[0], -1])
X_train, y_train = X_train / 255., np.reshape(y_train, [-1, 1])
test_x, test_y = np.reshape(X_test, [X_test.shape[0], -1]) / 255., np.reshape(y_test, [-1, 1])

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
LEARNING_RATE = 0.01
#学习率衰减率
LEARNING_RATE_DECAY = 0.99
total_batch = 101

def get_weight(shape, regularizer, name):
    # 参数满足截断正态分布，并使用正则化，
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
    # 将每个参数的正则化损失加到总损失中
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape, name):
    # 初始化的一维数组，初始化值为全 0
    b = tf.Variable(tf.zeros(shape), name=name)
    return b

def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer, "w1")
    b1 = get_bias([1, LAYER1_NODE], "b1")
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer, "w2")
    b2 = get_bias([1, OUTPUT_NODE], "b2")
    y = tf.add(tf.matmul(y1, w2), b2, name="res")
    return y

def backward():
    x = tf.compat.v1.placeholder(tf.float32, [None, INPUT_NODE], name="input_x")
    y_ = tf.compat.v1.placeholder(tf.int32, [None, 1], name="input_y")
    y = forward(x, None)
    correct_prediction = tf.equal(tf.expand_dims(tf.argmax(y, 1), -1), tf.cast(y_, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="test_accuracy")
    #当前计算轮数计数器赋值，设定为不可训练类型
    global_step = tf.Variable(0, trainable=False)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.one_hot(y_, depth=10)), name="loss")
    #使用梯度衰减算法对模型优化，降低损失函数
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss, global_step=global_step, name="train_step")
    for var in tf.trainable_variables():
        print(var.op.name, var.shape)
    with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name='train')
    with tf.Session() as sess:
        #所有参数初始化
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        for i in range(total_batch):
            batch_x, batch_y = X_train, y_train
            # batch_y_one_hot = np.eye(OUTPUT_NODE)[batch_y]
            _, loss_value, step, accuracy_ = sess.run([train_op, loss, global_step, accuracy], feed_dict={x: batch_x, y_: batch_y})
            #accuracy_test = sess.run(accuracy, feed_dict = {x: test_x, y_: test_y})
            if i % 1 == 0:
                #print("After %d training step(s), loss on training batch is %g, accuracy is %g, test accuracy is %g" % (step, loss_value, accuracy_, accuracy_test))
                print("After %d training step(s), loss on training batch is %g, accuracy is %g." % (step, loss_value, accuracy_))
        # tf.io.write_graph(tf.compat.v1.get_default_graph().as_graph_def(), './model_pb/', 'mnist_graph.pb', as_text=False)
        # tf.io.write_graph(sess.graph_def, '/data0/users/shuaishuai3/wt/t/t1_5/src/py/model_pb/', 'mnist_graph.pb', as_text=False)

def main():
    backward()

if __name__ == '__main__':
    main()

