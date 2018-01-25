import time

import tensorflow as tf
import numpy as np

import resnet_model
import input

FLAGS = tf.app.flags.FLAGS


def train(hps):
    images, labels = input.get_train_batch_data(256)
    model = resnet_model.ResNet(hps, images, labels, "train")
    model.build_graph()
    
    # 计算准确率
    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
    
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
    with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=FLAGS.log_root,
                    hooks=[logging_hook, _LearningRateSetterHook()],
                    chief_only_hooks=[summary_hook],
                    save_summaries_steps=0,
                    config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
        
        while not mon_sess.should_stop():
            mon_sess.run(model.train_op)
            print mon_sess.run(precision)
        

def main():
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