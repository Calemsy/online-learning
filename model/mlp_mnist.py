# -*- coding: UTF-8 -*-
import tensorflow as tf
import os, json
import numpy as np

BATCH_SIZE = 1024
FEATURE_DIM = 784
CLASS_NUM = 10
LABEL_DIM = 1
LAYER1_NODE = 500
LEARNING_RATE = 0.05
PB_SAVE_PATH='./model_pb/'
CONF_SAVE_PATH='./model_conf/'
MODEL_NAME='mlp_mnist'
ABSOLUT_PATH = "/data0/users/shuaishuai3/wt/t/t1_8/model/"

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
    w1 = get_weight([FEATURE_DIM, LAYER1_NODE], regularizer, "w1")
    b1 = get_bias([1, LAYER1_NODE], "b1")
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    w2 = get_weight([LAYER1_NODE, CLASS_NUM], regularizer, "w2")
    b2 = get_bias([1, CLASS_NUM], "b2")
    y = tf.add(tf.matmul(y1, w2), b2, name="logits")
    return y

def backward():
    x = tf.compat.v1.placeholder(tf.float32, [None, FEATURE_DIM], name="input_x")
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name="input_y")
    y = forward(x, None)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.expand_dims(tf.argmax(y, 1), -1), tf.cast(y_, tf.int64)), tf.float32), name="accuracy")
    #当前计算轮数计数器赋值，设定为不可训练类型
    global_step = tf.Variable(0, trainable=False)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.one_hot(tf.cast(y_, tf.int32), depth=10)), name="loss")
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step, name="train_step")

    for var in tf.trainable_variables():
        dvar_ = tf.gradients(loss, var)
        dvar = tf.identity(dvar_[0], name="d-" + var.op.name)
        print(dvar_[0], "d-" + var.op.name)
    with tf.Session() as sess:
        #所有参数初始化
        #init_op = tf.compat.v1.global_variables_initializer()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name="ws_init")
        sess.run(init_op)
        #tf.io.write_graph(tf.compat.v1.get_default_graph().as_graph_def(), './model_pb/', 'mnist_graph.pb', as_text=False)
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
			'input_size': 	[BATCH_SIZE, FEATURE_DIM], 
			'output_size': 	[BATCH_SIZE, LABEL_DIM],
			'class_num': 	CLASS_NUM, 
			'observe': 	['loss', 'accuracy'],
			'model_name': 	MODEL_NAME,
			'learning_rate': LEARNING_RATE,
			'pb_path': 	ABSOLUT_PATH + './model_pb/' + MODEL_NAME + '.pb'
		     }
        conf['other'] = other_conf
        conf['dense_shape'] = shape_conf
        json.dump(conf, f, indent=4, ensure_ascii=False)

