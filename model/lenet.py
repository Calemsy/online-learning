#-- coding:UTF-8 --
import tensorflow as tf
import os, json
import numpy as np

BATCH_SIZE 	= 1024  	# batch的数量
LEARNING_RATE 	= 0.05  	# 初始学习率
IMAGE_SIZE 	= 28 	 	# 每张图片分辨率为28*28
NUM_CHANNELS 	= 1  		# Mnist数据集为灰度图，故输入图片通道数NUM_CHANNELS取值为1
CONV1_SIZE 	= 5  		# 第一层卷积核大小为5
CONV1_KERNEL_NUM= 32  		# 卷积核个数为32
CONV2_SIZE 	= 5  		# 第二层卷积核大小为5
CONV2_KERNEL_NUM= 64  		# 卷积核个数为64
FC_SIZE 	= 512  		# 全连接层第一层为 512 个神经元
CLASS_NUM 	= 10  		# 全连接层第二层为 10 个神经元
REGULARIZER 	= 0.0001  	# 正则化
PB_SAVE_PATH	= './model_pb/'
CONF_SAVE_PATH	= './model_conf/'
MODEL_NAME	= 'lenet'

# 权重w计算
def get_weight(shape, regularizer, name):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

# 偏置b计算
def get_bias(shape, name):
    b = tf.Variable(tf.zeros(shape), name=name)
    return b

# 卷积层计算
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# 最大池化层计算
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def forward(x, train, regularizer):
    # 第一层卷积
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer, "conv1_w")
    conv1_b = get_bias([CONV1_KERNEL_NUM], "conv1_b")
    conv1   = conv2d(x, conv1_w)
    relu1   = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1   = max_pool_2x2(relu1)
    # 第二层卷积
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer, "conv2_w")
    conv2_b = get_bias([CONV2_KERNEL_NUM], "conv2_b")
    conv2   = conv2d(pool1, conv2_w)
    relu2   = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2   = max_pool_2x2(relu2)
    # 获取一个张量的维度
    pool_shape = pool2.get_shape().as_list()
    # pool_shape[1] 为长 pool_shape[2] 为宽 pool_shape[3]为高
    nodes   = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 得到矩阵被拉长后的长度，pool_shape[0]为batch值
    reshaped= tf.reshape(pool2, [-1, nodes])
    # 实现第三层全连接层
    fc1_w   = get_weight([nodes, FC_SIZE], regularizer, "fc1_w")
    fc1_b   = get_bias([FC_SIZE], "fc1_b")
    fc1     = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 如果是训练阶段，则对该层输出使用dropout
    if train: fc1 = tf.nn.dropout(fc1, 0.5)
    # 实现第四层全连接层
    fc2_w   = get_weight([FC_SIZE, CLASS_NUM], regularizer, "fc2_w")
    fc2_b   = get_bias([CLASS_NUM], "fc2_b")
    y       = tf.add(tf.matmul(fc1, fc2_w), fc2_b, name="logits")
    return y 

def backward():
    x 		= tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name="input_x")
    y_ 		= tf.placeholder(tf.float32, [None, 1], name="input_y")
    y 		= forward(x, True, REGULARIZER)
    ce 		= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.cast(tf.squeeze(y_), tf.int32))
    cem 	= tf.reduce_mean(ce)
    # 正则化的损失值
    loss 	= tf.add(cem, tf.add_n(tf.get_collection('losses')), name="loss")
    train_step 	= tf.train.GradientDescentOptimizer(0.05).minimize(loss, name="train_step")
    accuracy 	= tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.squeeze(tf.reshape(y_, (1, -1))), tf.int64), tf.argmax(y, 1)), tf.float32), name="accuracy")

    for var in tf.trainable_variables():
        dvar_ = tf.gradients(loss, var)
        dvar = tf.identity(dvar_[0], name="d-" + var.op.name)
        print(dvar_[0], "d-" + var.op.name)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name="ws_init")
        sess.run(init_op)
        tf.io.write_graph(sess.graph_def, PB_SAVE_PATH, MODEL_NAME+".pb", as_text=False)

def main():
    backward()

if __name__ == '__main__':
    main()

    with open(CONF_SAVE_PATH + MODEL_NAME + ".json", 'w') as f:
        conf = {}
        shape_conf, param_list, d_param_list = {}, [], []
        for var in tf.trainable_variables():
            shape_conf[var.op.name] = var.shape.as_list()
            shape_conf["d-" + var.op.name] = var.shape.as_list()
            param_list.append(var.op.name)
            d_param_list.append("d-" + var.op.name)
        other_conf = {
                        'dense': 	param_list,
                        'gradient': 	d_param_list,
                        'batch_size': 	BATCH_SIZE,
                        'input_size': 	[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS],
                        'output_size': 	[BATCH_SIZE, 1],
			'class_num': 	CLASS_NUM,
                        'observe': 	['loss', 'accuracy'],
                        'model_name': 	MODEL_NAME,
                        'learning_rate':LEARNING_RATE,
			'pb_path': 	'../model/model_pb/' + MODEL_NAME + '.pb'
                     }
        conf['other'] = other_conf
        conf['dense_shape'] = shape_conf
        json.dump(conf, f, indent=4, ensure_ascii=False)


