from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train', """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10, """How often to log results to the console.""")

def train():
	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()
	with tf.device('/cpu:0'):
		images, labels = cifar10.distorted_inputs()

	logits = cifar10.inference(images)

	loss = cifar10.loss(logits, lables)

	train_op = cifar10.train(loss, global_step)

	class _LoggerHook(tf.train.SessionRunHook):
		def begin(self):
			self._step = -1
			self._start_time = time.time()

		def before_run(self, run_context):
			self._step += 1
			return tf.train.SessionRunArgs(loss)
		def after_run(self, run_context, run_values):
			if self._step % FLAGS.log_frequency == 0:
				current_time = time.time()
				duration = current_time - self._start_time
				self._start_time = current_time

				loss_value = run_values.results
				examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
				sec_per_batch = float(duration / FLAGS.log_frequency)

				format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch')
				print(format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))
	with tf.train.MonitoredTrainingSession(
			checkpoint_dir=FLAGS.train_dir,
			hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps), tf.train.NanTensorHook(loss), _LoggerHook()],
			config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
	) as mon_sess:
			mon_sess.run(train_op)

def main(argv=None):
	cifar10.maybe_download_and_extract()
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)
	train()

if __name__=='__main__':
	tf.app.run()



