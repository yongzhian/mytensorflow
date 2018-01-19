import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))

# y = x*w1*w2 ,1x2*2x3*3x1
x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
y = tf.matmul(tf.matmul(x, w1), w2)

sess = tf.Session()
sess.run(tf.initialize_all_variables()) # 对variable进行初始化

print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
