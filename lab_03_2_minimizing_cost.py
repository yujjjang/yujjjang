import tensorflow as tf

tf.set_random_seed(777)

x_data=[1,2,3]
y_data=[1,2,3]

W=tf.Variable(tf.random_normal([1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

hypothesis=X*W+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))

learning_rate =0.01
gradient=tf.reduce_mean((W*X+b-Y)*X)
descent = W-learning_rate*gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
	update_val, cost_val,W_val, b_val=sess.run([update, cost, W, b], feed_dict={X:x_data, Y:y_data})
	if step%200 == 0:
		print(step, W_val,b_val, cost_val)

