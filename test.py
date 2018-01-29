 # coding=utf-8
import time

import tensorflow as tf
import numpy as np

import resnet_model
import input

def evaluate(hps):
    # 构建输入数据(读取队列执行器）
    images, labels = input.get_test_batch_data(100)

    # 构建残差网络模型
    model = resnet_model.ResNet(hps, images, labels, 'test')
    model.build_graph()
    # 模型变量存储器
    saver = tf.train.Saver()
    # 总结文件 生成器
    summary_writer = tf.summary.FileWriter("/home/ocean1100/tensorflow/TensorResnet/test/")

    # 执行Session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # 启动所有队列执行器
    tf.train.start_queue_runners(sess)

    best_precision = 0.0
    while True:
        # 检查checkpoint文件
        try:
            ckpt_state = tf.train.get_checkpoint_state("/home/ocean1100/tensorflow/TensorResnet/temp/")
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', "/home/ocean1100/tensorflow/TensorResnet/temp/")
            continue

        # 读取模型数据(训练期间生成)
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        # 逐Batch执行测试
        total_prediction, correct_prediction = 0, 0
        for _ in range(100):
            # 执行预测
            (loss, predictions, truth, train_step) = sess.run(
                [model.cost, model.predictions,
                 model._labels, model.global_step])
            # 计算预测结果
            test_labels = tf.one_hot(truth, depth=10)
            test_correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_labels, 1))
            precision = tf.reduce_mean(tf.cast(test_correct_pred, tf.float32))

            total_prediction += precision

        # 计算准确率
        precision = 1.0 * total_prediction / 100

        # 添加准确率总结
#        precision_summ = tf.Summary()
#        precision_summ.value.add(
#            tag='Precision', simple_value=precision)
#        summary_writer.add_summary(precision_summ, train_step)

        # 添加测试总结
        #summary_writer.add_summary(summaries, train_step)

        rtval = sess.run(precision)

        # 打印日志
        tf.logging.info('loss: %.3f, precision: %.3f' %
                        (loss, rtval))

        # 执行写文件
        summary_writer.flush()


def main(_):
    # 残差网络模型参数
    hps = resnet_model.HParams(batch_size=100,
                             num_classes=10,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')
    evaluate(hps)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
