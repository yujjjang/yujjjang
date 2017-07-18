import tensorflow as tf
x_d = [1,2,3]
y_d = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Mini

learning_rate = 0.1
gradient = tf.reduce_mean((W*X - Y)*W)
descent = W - learning_rate * gradient
update = W.assign(descent)


sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
	sess.run(update, feed_dict={X : x_d, Y: y_d})
	print(step, sess.run(cost, feed_dict={X:x_d, Y:y_d}), sess.run(W))

