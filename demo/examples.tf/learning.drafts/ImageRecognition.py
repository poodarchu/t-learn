from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
# labels is a tensor which contains a list of classes, e.g. [1, 9, 8, 3, 7, ...]
def cnn_model_fn(features, lables, mode):
    """Model function for CNN."""
    # Input layer
    # batch_size==-1 indicates that this dimension should be dynamically computed based on the number of input values in features
    # this allows us to treat batch_size as a hyperparameter that we can tune.
    input_layer = tf.reshape(features, [-1, 28, 28, 1]) #[batch_size, image_width, image_height, channels]

    # Convolutional Layer #1
    # In conv1 we apply 32 5x5 filters to the input layer, with relu activation function.
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5], #if filter width and height have the same value, you can specify a single integer 'kernel_size' instead: kernel_size=5
        padding='same', #same specifies the output tensor have the same width and hight height values as the input tensor
        # indicates that instructs TF to add 0 values to the edges of the output tensor to preserve width and height of 28
        activation=tf.nn.relu
    )
    # the output tensor produced by conv2d() has a shape of [batch_size, 28, 28, 32], with 32 channels holding the output from each filters

    # Pooling layer #1
    # if pool filter's width=height: 'pool_size=2' is also valid.
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    # the output tensor produced by max_pooling2d() has a shape of [batch_size, 14, 14, 32]


    # Convolutional layer #2 & Pooling layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    # the output tensor is [batch_size, 14, 14, 64]

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2) # the output tensor is [batch_size, 7, 7, 64]

    # Dense layer
    # here we add a dense layer with 1024 neurons.
    # before we connect the dense layer, we flatten the feature map(pool2) to shape of [batch_size, features], so that the tensor has only two dimensions.
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    # the units argument specifies the number of neurons in the dense layer(1024).
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # to help improve the results of our model, we also apply dropout regularization to our dense layer
    # dropout will only be performed if training is True.
    # here we check if the mode passed to our model function cnn_model_fn is TRAIN mode.
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == learn.ModeKeys.TRAIN) # the training argument takes a boolean specifying whether or not the model is currently being run in training mode.
    )

    # Logits Layer
    # the logits layer return the raw values for our predicitons.
    # we create a dense layer with 10 neurons (one for each target class 0-9), with linear activation(the default)
    logits = tf.layers.dense(inputs=dropout, units=10)
    # our final output tensor of the CNN, logits, has shape [batch_size, 10]


    # for multi-class classification problems like MNIST, cross entropy is typically used ad the loss metric.
    loss = None
    train_op = None

    # Calculate Loss (For both TRAIN ad EVAL modes)
    # for multi-class classification, cross-entropy is typically used as the loss metric
    if mode != lean.ModeKeys.INFER:
        # our labels tensor contains a list of predictions for our examples, e.g. [1, 9, ...]
        # in order to calculate cross-entropy, first we need to convert lables to the corresponging one-hot encoding
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels, logits=logits)

    # Configure the Training Op(for training mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss = loss,
            global_step = tf.contrib.framework.get_global_step(),
            learning_rate = 0.001,
            optimizer = 'SGD'
        )

    # Generate Predictions
    predictions = {
        'classes': tf.argmax(inputs=logits, axis=1), # our logits tensor has shape [batch_size, 10]
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor') # we use the name argument to explicitly name this operation softmax_tensor, so we can reference it later(e.g. we can set up logging for the softmax values)
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op
    )
# the above created the CNN model function, now we're ready to train and evaluate it
def main(unused_argv):
    # Load training and eval data
    mnist = learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images # return np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # return np.asarray
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator object
    # the model_fn specifies the model function to use for training, evaluation and inference
    # the model_dir specifies the directory where model data(checkpoints) will be saved
    mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir='/tmp/mnist_convnet_model')

    # Set up logging for predictions
    tensors_to_log = {'probabilities':'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    # Train the model
    mnist_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=20000,
        monitors=[logging_hook] # we pass the logging_hook to the monitor argument, so that it will be triggered during training
    )

    # Configure the accuracy metric for evaluation
    # to setup the accuracy metric for our model, we need to create a metrics dict with a tf.contrib.learn.MetricSpec that calculates accuracy
    metrics = {
        'accuracy':learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key='classes'),
    }

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        x=eval_data, y=eval_labels, metrics=metrics
    )

    print(eval_results)
    

if __name__='__main__':
    tf.app.run()
