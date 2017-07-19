import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

X=tf.placeholder(tf.float32, [None,3])
Y=tf.placeholder(tf.float32, [None,3])
W=tf.Variable(tf.random_normal([3,3]))
b=tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer=tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.arg_max(hypothesis,1)
is_correct = tf.equal(prediction, tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(201):
		cost_val, W_val, b_val, _ = sess.run([cost, W, b, optimizer], feed_dict={X:x_data, Y:y_data})
		if step % 40 == 0:
			print(step, cost_val, W_val)
	print(sess.run(prediction, feed_dict={X:x_test}))
	print(sess.run(accuracy, feed_dict={X:x_test, Y:y_test}))
