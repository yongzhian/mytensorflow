import tensorflow as tf

ones = tf.ones(shape=[2, 2], dtype=tf.float32, name="x")
sess = tf.Session()
print("ones:", sess.run(ones))
# [[ 1.  1.]
#  [ 1.  1.]]
zeros = tf.zeros(shape=[1, 2], dtype=tf.float32, name="y")
print("zeros:", sess.run(zeros))
# [[ 0.  0.]]

ones_like = tf.ones_like([[1, 9], [1, 9]])  # 2*2矩阵
print("ones_like:", sess.run(ones_like))
zeros_like = tf.ones_like(zeros)
print("zeros_like:", sess.run(zeros_like))

print("fill:", sess.run(tf.fill([2, 3], 2)))  # [[2 2 2],[2 2 2]]

print("constant:", sess.run(tf.constant([1, 2, 3], shape=[3, 2])))  # [[2 2 2],[2 2 2]]

print("random_normal:",
      sess.run(tf.random_normal(shape=[1, 5], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)))
# [[-0.46344429  0.01086242  0.33792984 -0.07529864  0.31897786]]

print("truncated_normal:",
      sess.run(tf.truncated_normal(shape=[1, 5], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)))
# [[-1.60640168  1.64168751 -0.04454968 -1.91379106  0.98286909]]

print("random_uniform:",
      sess.run(
          tf.random_uniform(shape=[1, 5], minval=0.0, maxval=2.0, dtype=tf.float32, seed=None, name="random_uniform")))
# [[ 0.95096731  1.02182341  1.76656771  1.3821938   1.08712864]]

# print("get_variable :",
#       sess.run(tf.get_variable("random_uniform", shape=[1, 5], initializer=tf.random_uniform_initializer(minval=-1,maxval=1))))

shape = tf.shape([2, 3])
print("shape:", shape)  # shape: Tensor("Shape:0", shape=(1,), dtype=int32)
print(sess.run(shape))  # 1X2 [2]

expand_dims = tf.expand_dims(shape, 1)
print("expand_dims:", expand_dims)  # expand_dims: Tensor("ExpandDims:0", shape=(1, 1), dtype=int32)
print(sess.run(expand_dims))  # [[2]]

a = tf.get_variable(name='a',
                    shape=[3, 4],
                    dtype=tf.float32,
                    initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
b = tf.argmax(input=a, dimension=0)
c = tf.argmax(input=a, dimension=1)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print(sess.run(a))
# [[ 0.04261756 -0.34297419 -0.87816691 -0.15430689]
# [ 0.18663144  0.86972666 -0.06103253  0.38307118]
# [ 0.84588599 -0.45432305 -0.39736366  0.38526249]]
print(sess.run(b))
# [2 1 1 2]
print(sess.run(c))
# [0 1 0]

a = tf.Variable([1, 0, 0, 1, 1])
b = tf.cast(a, dtype=tf.bool)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print(sess.run(b))
# [ True False False  True  True]

x = tf.linspace(start=1.0, stop=15.0, num=5, name=None)  # 注意1.0和5.0
y = tf.range(start=1, limit=5, delta=2)
print(sess.run(x))
print(sess.run(y))
# ===>[ 1.  2.  3.  4.  5.]
# ===>[1 2 3 4]

log = tf.log(2.0)
print("log:", sess.run(log)) # log: 0.693147 自然数e为底的对数
