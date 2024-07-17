import tensorflow as tf
import os, json
import numpy as np

BATCH_SIZE = 1024  	# batch的数量
LEARNING_RATE = 0.05  	# 初始学习率
PB_SAVE_PATH='./model_pb/'
CONF_SAVE_PATH='./model_conf/'
MODEL_NAME='xxxxxxxxxx'
ABSOLUT_PATH = "/data0/users/shuaishuai3/wt/t/t1_8/model/"

# 权重w计算
def get_weight(shape, regularizer, name):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

# 偏置b计算
def get_bias(shape, name):
    b = tf.Variable(tf.zeros(shape), name=name)
    return b

# 前向传播过程
def forward(x, ...):
    y = ...
    y_ = tf.identity(y, name="logits")
    return y 

# 反向传播
def backward():
    x = tf.placeholder(tf.float32, [None, `other dims`], name="input_x")
    y_ = tf.placeholder(tf.float32, [None, `other dims`], name="input_y")
    y = forward(x, ...)
    loss = ...
    loss_ = tf.identity(loss, name="loss")
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, name="train_step")
    accuracy = ...
    accuracy = tf.identity(accuracy, name="accuracy")

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
                        'dense': param_list,
                        'gradient': d_param_list,
                        'batch_size': BATCH_SIZE,
                        'input_size': [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS],
                        'output_size': [BATCH_SIZE, 1],
                        'observe': ['loss', 'accuracy'],
                        'model_name': MODEL_NAME,
                        'learning_rate': LEARNING_RATE,
			'pb_path': './model_pb/' + MODEL_NAME + '.pb'
                     }
        conf['other'] = other_conf
        conf['dense_shape'] = shape_conf
        json.dump(conf, f, indent=4, ensure_ascii=False)


