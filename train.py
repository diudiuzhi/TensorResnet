# coding=utf-8
import time

import tensorflow as tf
import numpy as np

import resnet_model
import input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_root',
                           'temp',
                           'directory of checkpoint')

# 训练过程数据的存放路劲
tf.app.flags.DEFINE_string('train_dir',
                           'temp/train',
                           'Directory to keep training outputs.')


def train(hps):
    images, labels = input.get_train_batch_data(128)
    model = resnet_model.ResNet(hps, images, labels, "train")
    model.build_graph()

    # 计算准确率
    train_labels = tf.one_hot(model._labels, depth=10)
    train_correct_pred = tf.equal(tf.argmax(model.predictions, 1), tf.argmax(train_labels, 1))
    precision = tf.reduce_mean(tf.cast(train_correct_pred, tf.float32))

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        def begin(self):
            #初始学习率
            self._lrn_rate = 0.1

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(model.global_step,
                                           feed_dict={model.lrn_rate: self._lrn_rate})

        def after_run(self, run_context, run_values):
            train_step = run_values.results
            if train_step < 40000:
                self._lrn_rate = 0.1
            elif train_step < 60000:
                self._lrn_rate = 0.01
            elif train_step < 80000:
                self._lrn_rate = 0.001
            else:
                self._lrn_rate = 0.0001

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss': model.cost,
                 'precision': precision},
        every_n_iter=100)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=FLAGS.train_dir,
        summary_op=tf.summary.merge(
            [model.summaries,
             tf.summary.scalar('Precision', precision)]))

    stop_hook = tf.train.StopAtStepHook(last_step=100000)

    with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=FLAGS.log_root,
                    hooks=[logging_hook, _LearningRateSetterHook(), stop_hook],
                    chief_only_hooks=[summary_hook],
                    save_summaries_steps=0,
                    config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:

        while not mon_sess.should_stop():
            mon_sess.run(model.train_op)


def main(_):
    hps = resnet_model.HParams(batch_size=128,
                               num_classes=10,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom')
    train(hps)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
