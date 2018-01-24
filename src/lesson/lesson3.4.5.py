import tensorflow as tf
from numpy.random import RandomState

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
_y = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

y = tf.matmul(tf.matmul(x, w1), w2)

print(1.0 + 2e12)

# 定义损失函数，经典损失函数
cross_entropy = -tf.reduce_mean(_y * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=_y, name=None)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  # 反向传播函数

# 生成模拟训练数据集
rdm = RandomState(1)  # 定义局部种子
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 0为负样本 1为正样本
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

batch_size = 8

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 训练之前
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 10000
    for i in range(STEPS):
        # 每次训练batch_size个
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], _y: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, _y: Y})
            print("经过%d步，损失熵是%g" % (i, total_cross_entropy))
    # 训练之后
    print(sess.run(w1))
    print(sess.run(w2))
