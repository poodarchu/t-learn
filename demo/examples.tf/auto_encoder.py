# -*- coding=utf-8 -*-

"""
    自编码器的学习目标是 使用少量稀疏的高阶特征重构输入：
        1. 限制中间隐层节点的数量，比如让中级那隐含层节点的数量小于输入/输出节点的数量，就相当于一个降维
            再给中间隐含层的权重加一个 L1 正则，则可以根据隐含系数控制隐含节点的稀疏程度，惩罚系数越大，学到的特征组合越稀疏，实际使用（非零权重）
            的特征数量越少。
        2. 如果给数据加入噪声，那么就是 Denoising Auto Encoder（去噪自编码器）
            唯有学习数据频繁出现的模式和结构，将无规律的噪声略去，才可以复原数据。
            常使用的噪声是 加性高斯噪声（Additive Gaussian Noise）
            
    Hinton 提出的 DBN 模型有多个隐含层，每个隐含层都是限制性玻尔兹曼机 RBM。
    
    下面实现 去噪自编码机（Denoising AutoEncoder）
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 如果深度学习模型的权重初始化得太小，那信号将在每层间传递时逐渐缩小而难以产生作用
# 如果权重初始化得太大，那么信号将在每层间传递时逐渐方法并导致发散和失效
# 使用 Xavier Initialization 作为参数初始化方法
# Xaiver 初始化器做的事情是： 让深度学习模型的权重呗初始化得不大不小，正好合适。
# 从数学的角度分析，Xaiver 就是让权重满足 0 均值，同时方差为 2/(n_in+n_out), 分布可以用均匀分布或者高斯分布。
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class AdditiveGaussianNoiseAutoEncoder(object):
    """
        n_input: 输入变量数
        n_hidden: 隐含层节点的数目
        transfer_function: 隐含层激活函数，默认为 softplus
        optimizer: 优化器，默认为 Adam
        scale: 高斯噪声系数
        class内的scale做成了一个placeholder
        这里只使用了一个隐含层
    """
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 将输入 x 加上噪声，然后乘上隐含层的权重 w1，再加上偏置 b1
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x + scale*tf.random_normal((n_input,)),
            self.weights['w1']), self.weights['b1']))
        # 在输出层进行数据复原、重建操作，这里不需要激活曾，直接将隐含层的输出乘上输出层的权重再加上偏置即可
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 定义自编码器的损失函数
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))

        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()

        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

        return all_weights

    # 定义计算损失 cost 以及执行一步训练的函数 partial_fit
    # 函数 partial_fit 做的就是用一个 batch 数据进行训练并返回当前的 cost
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 该函数在自编码器训练完毕后，在测试集上对模型性能进行评估，因此不会像 partial_fit 一样触发训练操作
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 该函数用户返回自编码器隐含层的输出结果
    # 他的目的是提供一个接口来获取抽象后的特征，自编码器的隐含层的最主要的功能就是学习出数据中的高阶特征
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # 该函数将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBias(self):
        return self.sess.run(self.weights['b1'])

# 读取数据集
mnist = input_data.read_data.read_data_sets('MNIST_data', one_hot=True)

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnits.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1 # 设置每隔一轮就显示一次 cost

autoEncoder = AdditiveGaussianNoiseAutoEncoder(
                n_input=784,
                n_hidden=200,
                transfer_function = tf.nn.softplus,
                optimizer=tf.train。AdamOptimizer(learning_rate=0.001),
                scale=0.01)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoEncoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples*batch_size
    if epoch % display_step == 0:
        print('Epoch: ', '%40d' %(epoch+1), 'cost=', '{:.9f}'.format(avg_cost))

print("Total cost: " + str(autoEncoder.calc_total_cost(X_test)))




