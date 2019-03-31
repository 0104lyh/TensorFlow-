import tensorflow as tf
#1.定义变量及滑动平均类

w1 = tf.Variable(0,dtype=tf.float32)

#定义num_updates(NN的迭代轮数)，初始值为0，不可被优化（训练）
global_step=tf.Variable(0,trainable=False)
#实例滑动平均类，给删减率为0.99，当前轮数为global_step
MOVING_AVERAGE_DECAY=0.99
ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
#ema.apply后的括号是更新列表，每次运行sess.run(ema_op)时，对更新列表中的元素求滑动平均值
#在实际应用中会使用tf.Variable_variables()自动将所有待训练的参数汇总为列表
#ema_op=ema.apply([w1])
ema_op=ema.apply(tf.trainable_variables())

#2.查看不同迭代中变量取值的变化
with tf.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)
	print(sess.run([w1,ema.average(w1)]))

	#参数w1的值赋为1
	sess.run(tf.assign(w1,1))
	sess.run(ema_op)
	print(sess.run([w1,ema.average(w1)]))
	#更新step和w1的值，模拟出100轮迭代后，参数w1变为10
	sess.run(tf.assign(global_step,100))
	sess.run(tf.assign(w1,10))
	sess.run(ema_op)
	print(sess.run([w1,ema.average(w1)]))
	sess.run(ema_op)
	print(sess.run([w1,ema.average(w1)]))
	sess.run(ema_op)
	print(sess.run([w1,ema.average(w1)]))
	sess.run(ema_op)
	print(sess.run([w1,ema.average(w1)]))
	sess.run(ema_op)
	print(sess.run([w1,ema.average(w1)]))

