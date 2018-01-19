from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# Import data
print(" begin ")
from tensorflow.examples.tutorials.mnist import input_data

print(input_data)
import tensorflow as tf

# 不存在会下载，建议提前下载好 http://yann.lecun.com/exdb/mnist/
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 60000张图片，每张28*28=784px[60000, 784]
x = tf.placeholder(tf.float32, [None, 784])

# 0-9一共10个数字，任意一个像素归为0-9中一个
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()

# 运行在没有指定会话对象的情况下运行变量,with下也必须调用close
sess = tf.InteractiveSession()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    if (i % 50 == 0):
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
