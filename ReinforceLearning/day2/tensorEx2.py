
import tensorflow as tf

# vdata1 = tf.Variable(10.)
# print(vdata1)
#
# vdata2 = tf.Variable(tf.ones([3,4]))
# print()
# print(vdata2)
#
# print(vdata2 * 2)
# print(vdata2.assign(tf.zeros([3,4])))
# print()
# print(vdata2.assign_add(tf.ones([3,4])))

x = tf.Variable(tf.constant(1.0))
with tf.GradientTape() as tape:
y = tf.multiply(5, x)

gradient = tape.gradient(y, x)
print(gradient.numpy())
print()

x1 = tf.Variable(tf.constant(3.))
x2 = tf.Variable(tf.constant(5.))

with tf.GradientTape() as tape:
y = tf.multiply(x1, x2)

gradients = tape.gradient(y, [x1, x2])
print(gradients)
print(gradients[0].numpy())
print()

x2 = tf.Variable(tf.constant(7.))
a = tf.constant(3.)

with tf.GradientTape() as tape:
tape.watch(a)
y = tf.multiply(a, x2)

gradient = tape.gradient(y, a)
print(gradient)









