import tensorflow as tf
import os, json
import numpy as np

BATCH_SIZE 	= 1024 
IMAGE_SIZE 	= 28  
NUM_CHANNELS 	= 1 
LEARNING_RATE 	= 0.05
CLASS_NUM 	= 27
PB_SAVE_PATH	= './model_pb/'
CONF_SAVE_PATH	= './model_conf/'
MODEL_NAME	= 'vgg'

# 权重w计算
def get_weight(shape, name):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
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

# 前向传播过程
def forward(x):
    conv1_w 	= get_weight([3, 3, 1, 64], "conv1_w")
    conv1_b 	= get_bias([64], "conv1_b")
    conv1 	= conv2d(x, conv1_w)
    relu1 	= tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 	= max_pool_2x2(relu1)

    conv2_w 	= get_weight([3, 3, 64, 128], "conv2_w")
    conv2_b 	= get_bias([128], "conv2_b")
    conv2 	= conv2d(pool1, conv2_w)
    relu2 	= tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 	= max_pool_2x2(relu2)

    conv3_w 	= get_weight([3, 3, 128, 256], "conv3_w")
    conv3_b 	= get_bias([256], "conv3_b")
    conv3 	= conv2d(pool2, conv3_w)
    relu3 	= tf.nn.relu(tf.nn.bias_add(conv3, conv3_b))
    pool3 	= max_pool_2x2(relu3)

    pool_shape 	= pool3.get_shape().as_list()
    nodes 	= pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped 	= tf.reshape(pool3, [-1, nodes])
    fc1_w 	= get_weight([nodes, 512], "fc1_w")
    fc1_b 	= get_bias([512], "fc1_b")
    fc1 	= tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    fc1 	= tf.nn.dropout(fc1, 0.5)

    fc2_w 	= get_weight([512, CLASS_NUM], "fc2_w")
    fc2_b 	= get_bias([CLASS_NUM], "fc2_b")
    y 		= tf.add(tf.matmul(fc1, fc2_w), fc2_b, name="logits")
    return y

# 反向传播
def backward():
    x 		= tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name="input_x")
    y_ 		= tf.placeholder(tf.float32, [None, 1], name="input_y")
    y 		= forward(x)
    ce 		= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.cast(tf.squeeze(y_), tf.int32))
    loss 	= tf.reduce_mean(ce, name="loss")
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


