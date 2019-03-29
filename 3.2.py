#简单神经网络
import tensorflow as tf
#传入参数x
x = tf.constant([[0.7,0.5]])
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#定义前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#用会话计算结果
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	#这里的这句话的意思是对所有变量的初始化（赋初值）
	sess.run(init_op)
	print("y in 3.2.py is:\n",sess.run(y))
