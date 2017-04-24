# -*- coding=utf-8 -*-
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 查看 mnist 数据集的基本信息
print (mnist.train.images.shape, mnist.train.labels.shape)
print (mnist.test.images.shape, mnist.test.labels.shape)
print (mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf

# 创建一个新的 InteractiveSession，这个命令会将这个session注册为默认的session
# 不同的 session 之间的数据和运算都是互相独立的
session = tf.InteractiveSession()
# 创建输入数据的地方，第一个参数是数据类型，第二个参数代表 tensor 的shape（数据的尺寸）
x = tf.placeholder(tf.float32, [None, 784])

# Variable 用于存储模型参数，不同书存储数据的 tensor，一旦使用掉就会消失
# Variable 在模型训练迭代中是持久化的（比如一直放在显存中）
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# tf.nn 包含了大量的神经网络组件
# tf.matmul 是矩阵乘法
# 下面一行我们定义了 soft max regression
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数 loss function
# y_ 是真实的概率分布（即 label 的 one-hot coding）
# tf.reduce_sum 用于求和，tf.reduce_mean 用于对每个 batch 数据结果求均值
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

# 定义训练步骤，使用 SGD 进行优化
# 前面定义的各个公式已经自动构成了计算图
# tf 会根据我们定义的整个计算图自动求导，并根据反向传播进行训练
# 设置学习速率为 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 使用 TensorFlow 的全局参数初始化器，并直接执行它的 run 方法
tf.global_variables_initializer().run()

# 每次从训练集中抽取 100 条样本构成一个 mini-batch，并 feed 给 placeholder，然后调用 train_step 对这些样本进行训练
# 对大部分机器学习问题，我们都使用一小部分数据进行随机梯度下降，这种做法绝大多数时候会比全样本训练的收敛速度快很多
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# tf.argmax 用于寻找一个 tensor 中最大值的序号，tf.argmax(y,1) 是求各个预测的数字中，概率最大的那个
# tf.equal 用于判断数字类别是否是正确的，最后返回计算分类是否正确的操作 correct_prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 统计全部样本预测的 accuracy， 先用 tf.cast 将之前 correct_prediction 输出的 bool 值转换为 float32 再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print (accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))


'''
1. 定义算法公式，也就是神经网络forward时的计算
2. 定义loss，选定优化器，并指定优化器优化的loss
3. 迭代地对数据进行训练
4. 在测试集或验证集上对准确率进行评测

定义的各个公式只是 computation graph，在执行这些代码时，计算还没有实际发生，只有等到调用 run 并 feed 数据时，计算才真正执行
'''



