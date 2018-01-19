import tensorflow as tf

# 所有的计算均在一个图中

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", shape=[1],initializer=tf.zeros_initializer())

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v",shape=[1], initializer=tf.ones_initializer())

print(g1 == tf.get_default_graph())
print(g1 == g2)

with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

# Tensorflow是一个通过计算图的形式来表述计算的编程系统，计算图也叫数据流图，
# 可以把计算图看做是一种有向图，Tensorflow中的每一个计算都是计算图上的一个节点，
# 而节点之间的边描述了计算之间的依赖关系。
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))  # Variable v does not exist