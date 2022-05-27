import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(777)

x_data = [[73,80,75],
[93,88,93],
[89,91,90,],
[96,98,100],
[73,65,70]]
y_data = [152,185,180,196,142]

x = tf.placeholder(dtype=tf.float32, shape=[5,3])
y = tf.placeholder(dtype=tf.float32)

w = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
update = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(2000):
    _, _c, _w, _b = sess.run([update, cost, w, b], feed_dict={x:x_data, y:y_data})
    print('epoch:{}\ncost:{}\nw:{}\nb:{}\n\n'.format(epoch, _c, _w, _b))

