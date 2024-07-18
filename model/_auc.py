import tensorflow as tf

y_true = [1, 1, 1, 0]
y_pred = [0.8, 0.2, 0.7, 0.3]

auc_value, auc_op = tf.metrics.auc(labels=tf.convert_to_tensor(y_true), predictions=tf.convert_to_tensor(y_pred))

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    auc_result = sess.run(auc_op)
    print("AUC:", auc_result)
