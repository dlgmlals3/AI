
import tensorflow as tf

a = tf.constant([1,2,3])
print(a)

b = tf.constant([[1,2,3],[4,5,6]], dtype=tf.float32)
print(b)

c = tf.range(5)
print(c)
print()

d = tf.linspace(0., 5., 10)
print(d)
print()
print(tf.zeros([2,3]))
print()
print(tf.fill([2,3], 5))
print()
print(tf.ones_like([[1,2,3],[5,6,7]]))

tf.random.set_seed(111)
print()
print(tf.random.normal([2,3], seed=777))

e = tf.constant([[2,4,7],[8,9,11]])
print()
print(e.shape)
print(e.numpy())
print()

f = tf.range(6, dtype=tf.int32)
rdata = tf.reshape(f, [2,3])
print(rdata)

edata = tf.expand_dims(rdata, axis=1)
print()
print(edata)

print()
print(tf.squeeze(edata))

print()
g = tf.range(6, dtype=tf.int32) * 2
print(g)
print()

print(tf.reduce_sum(g).numpy())
print()

h = tf.constant([[2,5,3],[4,6,8]])
print(tf.reduce_mean(h, axis=1))



