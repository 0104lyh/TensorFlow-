#3.7 实例：预测酸奶的日销量
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
 
#基于seed产生随机数
rdm = np.random.RandomState(SEED)
#随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
X = rdm.rand(32,2)
#从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如果和不小于1 给Y赋值0 
#作为输入数据集的标签（正确答案） 
Y_=[[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

#定义神经网络的输入，参数输出，前向传播过程
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))
w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y=tf.matmul(x,w1)

#定义损失函数及反向传播的过程
loss_mse=tf.reduce_mean(tf.square(y_-y))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

#生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=20000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%32
        end=(i*BATCH_SIZE)%32+BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i%500==0:
            print("after %d training steps,w1 is:\n"%(i))
            print(sess.run(w1),"\n")
    print("final is :\n",sess.run(w1))        
