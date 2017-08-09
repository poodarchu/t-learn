import tensorflow as tf
from tensorflow.contrib import rnn

learning_rate = 0.01
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

weights = {
    "out" : tf.Varibale(tf.random_normal([n_hidden, n_classes]))
}

biases = {
    "out" : tf.Varibale(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    # Unstack to get a list of 'n_steps' tensors of shape(batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step*batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            loss = sess.run(cost, feed_dict={x:batch_x, y:batch_y})
            print "iter " + str(step*batch_size) + ", Minibatch Loss=" + "{.6f}".format(loss) + ", Training Accuracy= " + "{.5f}".format(acc)
        step += 1
    print "Optimization Finished."

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x:test_data, y:test_label})
