#!/bin/python

import sys
import h5py
import numpy as np
import tensorflow as tf

input_height = 84
input_width = 110
n_action = 6

x = tf.placeholder(tf.float32, [None, input_width, input_height, 1],
        name='X')
xn = tf.placeholder(tf.float32, [None, input_width, input_height, 1],
        name='Xn')
a = tf.placeholder(tf.float32, [None],
        name='a')
r = tf.placeholder(tf.float32, [None, 1],
        name='r')

def conv2d(x, kernel_shape, bias_shape, strides):
    W = tf.get_variable('weights', kernel_shape)
    b = tf.get_variable('biases', bias_shape)
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def dense(x, dim):
    W = tf.get_variable('weights', [x.get_shape().as_list()[-1], dim])
    b = tf.get_variable('biases', [dim])
    x = tf.add(tf.matmul(x, W), b)
    return tf.nn.relu(x)

def state_vec(x, conv_shapes, strides, dense_dims):
    for i, (conv_shape, stride) in enumerate(zip(conv_shapes, strides)):
        with tf.variable_scope('conv%d' % i):
            x = conv2d(x, conv_shape, [conv_shape[-1]], stride)
    x_n = reduce(lambda x, y: x * y, x.get_shape().as_list()[1:])
    x = tf.reshape(x, [-1, x_n])
    for i, dense_dim in enumerate(dense_dims):
        with tf.variable_scope('dense%d' % i):
            x = dense(x, dense_dim)
    return x

vec_conv_shapes = [[8, 8, 1, 16], [4, 4, 16, 32]]
vec_conv_strides = [4, 2]
vec_dense_shapes = [256]
with tf.variable_scope('vec', initializer=tf.random_normal_initializer()) as scope:
    with tf.name_scope('x_vec'):
        x_vec = state_vec(x,
                vec_conv_shapes, vec_conv_strides, vec_dense_shapes)
    scope.reuse_variables()
    with tf.name_scope('xn_vec'):
        xn_vec = state_vec(xn,
                vec_conv_shapes, vec_conv_strides, vec_dense_shapes)

ahot = tf.one_hot(tf.cast(a, 'int32'), n_action)
pred_in = tf.concat((x_vec, ahot), 1, name='pred_in')
actual = tf.concat((xn_vec, r), 1, name='actual')
with tf.variable_scope('pred'):
    with tf.variable_scope('dense1'):
        pred_out = dense(pred_in, 256)
    with tf.variable_scope('dense2'):
        pred_out = dense(pred_out, 256 + 1)

loss = tf.reduce_mean(tf.squared_difference(pred_out, actual))
tf.summary.scalar('loss', loss)
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train', sess.graph)
test_writer = tf.summary.FileWriter('./test')
init = tf.global_variables_initializer()
sess.run(init)

batch_size = 128
dataf = h5py.File(sys.argv[1], 'r')
obs = dataf['obs']
acts = dataf['acts']
rews = dataf['rews']
dones = dataf['dones']

def feed_dict(train):
    choice = np.random.choice(len(obs) - 1, batch_size, replace=False)
    choice = sorted(choice)
    choice_next = list(np.array(choice) + 1)
    return {
        x: np.expand_dims(obs[choice], -1),
        xn: np.expand_dims(obs[choice_next], -1),
        a: acts[choice],
        r: np.expand_dims(rews[choice], -1)}

for i in range(10000):
    #if i % 10 == 0:  # Record summaries and test-set accuracy
    #    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    #    test_writer.add_summary(summary, i)
    #    print('Accuracy at step %s: %s' % (i, acc))
    #else:  # Record train set summaries, and train
    if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, loss_val, _ = sess.run([merged, loss, train_step],
                 feed_dict=feed_dict(True),
                 options=run_options,
                 run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for: %d loss: %d' % (i, loss_val))
    else:  # Record a summary
        summary, _ = sess.run([merged, train_step],
            feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()
