import tensorflow as tf
import os, json
import numpy as np

BATCH_SIZE      = 1024
FEATURE_SIZE    = 39
FEATURE_DIM     = 8
V_SIZE          = 65536
LABEL_DIM       = 1
LEARNING_RATE   = 0.05
PB_SAVE_PATH    = './model_pb/'
CONF_SAVE_PATH  = './model_conf/'
MODEL_NAME      = 'ffm'

def get_weight(shape, name):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
    return w

def get_bias(shape, name):
    b = tf.Variable(tf.zeros(shape), name=name)
    return b

def forward(x):
    w0              = get_weight([1, 1], "w0")
    w               = get_weight([V_SIZE, 1], "w")
    v               = get_weight([V_SIZE, FEATURE_SIZE, FEATURE_DIM], "v")
    x               = tf.cast(x, tf.int32)
    first_order     = tf.reduce_sum(tf.nn.embedding_lookup(w, x), axis=1) + w0
    second_input    = tf.nn.embedding_lookup(v, x)
    second_order    = 0
    for i in range(FEATURE_SIZE):
        for j in range(i + 1, FEATURE_SIZE):
            second_order += tf.reduce_sum(second_input[:, i, j] * second_input[:, j, i], axis=-1, keepdims=True)
    y               = first_order + second_order
    y_              = tf.identity(y, name="logits")
    return y 

def backward():
    x           = tf.placeholder(tf.float32, [None, FEATURE_SIZE], name="input_x")
    y_          = tf.placeholder(tf.float32, [None, LABEL_DIM], name="input_y")
    y           = forward(x)
    loss        = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=tf.cast(y_, tf.float32)), name="loss")
    loss_       = tf.identity(loss, name="loss")
    auc_v,auc_op= tf.metrics.auc(tf.cast(y_, tf.int32), tf.sigmoid(y))
    auc_op_     = tf.identity(auc_op, name="auc_value")
    train_step  = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, name="train_step")

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
                        'input_size': 	[BATCH_SIZE, FEATURE_SIZE],
                        'output_size': 	[BATCH_SIZE, 1],
                        'observe': 	['loss', 'auc_value'],
                        'model_name': 	MODEL_NAME,
                        'learning_rate':LEARNING_RATE,
			'pb_path': 	'../model/model_pb/' + MODEL_NAME + '.pb'
                     }
        conf['other'] = other_conf
        conf['dense_shape'] = shape_conf
        json.dump(conf, f, indent=4, ensure_ascii=False)


