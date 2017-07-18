import tensorflow as tf

x1_data = [7,8,7,9,8]
x2_data = [6,7,8,9,10]
x3_data = [3,7,8,8,8]
#############
y_data = [[15.],[18.],[13.],[12.],[10.]]
x_data = [[7,6,5],[7,8,9],[8,8,8],[9,9,9],[5,7,8]]
################3
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

######################
X= tf.placeholder(tf.float32, shape=[None, 3])
Y= tf.placeholder(tf.float32, shape=[None, 1])
########################

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')


W = tf.Variable(tf.random_normal([3,1]), name ='weight')

hypothesis1 = x1*w1 + x2*w2 + x3*w3 + b
hypothesis = tf.matmul(X,W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
	cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
	if step % 10 == 0:
		print(step, cost_val, hy_val)
