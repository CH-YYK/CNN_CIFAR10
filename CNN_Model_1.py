#
import tensorflow as tf
import numpy as np
from tf4_load_data import *

# --- Model functions ---
# create tf.Variable for 'Weights'
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial)

# create tf.Variable for 'bias'
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
	return tf.Variable(initial)

# convolution function for input x with filter W.
def conv2d(x, W):
	# full strides convolutional layer
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# max pooling
def max_pool_2x2(x):
	# kernel size: 2*2, strides: 2*2
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# create convolutional layer
def conv_layer(Input, shape):
	W = weight_variable(shape)		# shape: width * height * channels * n_maps
	b = bias_variable([shape[3]]) 	# shape: n_maps
	return tf.nn.relu(conv2d(Input, W) + b)

# create a standard full connected layer (without activation function)
def full_layer(Input, size):
	in_size = int(Input.get_shape()[1])
	W = weight_variable([in_size, size])	# n_input * n_neurons
	b = bias_variable([size])				# n_neurons
	return tf.matmul(Input, W)+b

# --- Apply Model to CIFAR10 --
X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(dtype=tf.int32, shape=[None, 10])

keep_prob = tf.placeholder(tf.float32)

C1, C2, C3 = 30, 50, 80
F1 = 500

# -- 1st convolutional layer -- 
conv1_1 = conv_layer(X, shape=[3, 3, 3, C1])			# 32*32*3 --> 32*32*C1
conv1_2 = conv_layer(conv1_1, shape=[3, 3, C1, C1])		# 32*32*C1 --> 32*32*C1
conv1_3 = conv_layer(conv1_2, shape=[3, 3, C1, C1])		# 32*32*C1 --> 32*32*C1
conv1_pool = max_pool_2x2(conv1_3)						# 32*32*C1 --> 16*16*C1
conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=keep_prob)


# -- 2nd convolutional layer --
conv2_1 = conv_layer(conv1_drop, shape=[3, 3, C1, C2])	# 16*16*C1 --> 16*16*C2
conv2_2 = conv_layer(conv2_1, shape=[3, 3, C2, C2])		# 16*16*C2 --> 16*16*C2
conv2_3 = conv_layer(conv2_2, shape=[3, 3, C2, C2])		# 16*16*C2 --> 16*16*C2
conv2_pool = max_pool_2x2(conv2_3)						# 16*16*C2 --> 8*8*C2
conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)

# -- 3rd convolutional layer --
conv3_1 = conv_layer(conv2_drop, shape=[3, 3, C2, C3])	# 8*8*C2 --> 8*8*C3
conv3_2 = conv_layer(conv3_1, shape=[3, 3, C3, C3])		# 8*8*C3 --> 8*8*C3
conv3_3 = conv_layer(conv3_2, shape=[3, 3, C3, C3])		# 8*8*C3 --> 8*8*C3
conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1,8,8,1], strides=[1,8,8,1],
							padding='SAME')				# 8*8*C3 --> 1*1*C3
conv3_flat = tf.reshape(conv3_pool, [-1, C3])			#  
conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

full1 = tf.nn.relu(full_layer(conv3_flat, F1))
full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)			# output vector

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- Import Data ---
cifar = CifarDataManager()
def test(sess):
	X1 = cifar.test.images.reshape(10, 1000, 32, 32, 3)
	Y2 = cifar.test.labels.reshape(10, 1000, 10)
	acc = np.mean([sess.run(accuracy, feed_dict={X: X1[i][:50].astype(np.float32), y_: Y2[i][:50].astype(np.float32), keep_prob: 0.5}) for i in range(10)])
	print("Accuracy: {:.4}%".format(acc * 100))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(10000):
		batch = cifar.train.next_batch(100)
		sess.run(train_op, feed_dict={X: batch[0], y_: batch[1], keep_prob: 0.5})
		if i % 200 == 0:
			print("accuracy is:", sess.run(accuracy, feed_dict={X: batch[0], y_: batch[1], keep_prob: 0.5}))
	test(sess)
