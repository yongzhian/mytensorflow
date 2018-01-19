import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[1, 5]], name="m1", dtype=tf.float32, shape=[1, 2])
matrix2 = tf.constant([[3], [4]], name="m2", dtype=tf.float32)
print(matrix2.get_shape())


r1 = tf.multiply(13, 2)

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    print(sess.run(r1))
    print(sess.run(product))
