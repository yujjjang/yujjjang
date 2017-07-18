import tensorflow as tf

hello = tf.constant("HELLO")
sess = tf.Session()

print(sess.run(hello))

node1 = tf.constant(4.0, tf.float32)
node2 = tf.constant(3.0)
node3 = tf.add(node1,node2)

print(node1)
print(node2)
print(node3)

print(sess.run([node1,node2]))
print(sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder = a+b

print(sess.run(adder, feed_dict={a:3, b:4.5}))

