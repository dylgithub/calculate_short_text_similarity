#encoding=utf-8
import tensorflow as tf
import numpy as np
from embedding_flag_cnntext_sim import sim_data_helper,cnn_textsim_model
'''
进行反向传播，优化模型
'''
# 模型的超参数
tf.flags.DEFINE_integer("embedding_size", 80, "每个单词向量的维度")
tf.flags.DEFINE_integer("sentence_length", 100, "句子的长度")
tf.flags.DEFINE_integer("n_hidden", 128, "全连接层节点的个数")
tf.flags.DEFINE_integer("num_filters_A", 20, "模块A卷积核的个数")
tf.flags.DEFINE_integer("num_filters_B", 20, "模块B卷积核的个数")
tf.flags.DEFINE_integer("num_classes", 6, "类别种类数")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2正则化系数的比率")
filter_size=[1,2,100]
# 训练参数
tf.flags.DEFINE_float("keep_prob", 0.5, "丢失率")
tf.flags.DEFINE_integer("batch_size", 64, "每个批次的大小")
tf.flags.DEFINE_integer("num_epochs", 100, "训练的轮数")
tf.flags.DEFINE_integer("num_steps", 1000, "学习率衰减的步数")
tf.flags.DEFINE_float("init_learning_rate", 0.01, "初始学习率")
FLAGS = tf.flags.FLAGS
#注意这里必须是tf.int32类型的，因为是词的索引为整型
input_1 = tf.placeholder(tf.int32, [None, FLAGS.sentence_length], name='input_x1')
input_2 = tf.placeholder(tf.int32, [None, FLAGS.sentence_length], name='input_x2')
input_y = tf.placeholder(tf.float32, [None, FLAGS.num_classes], name='output')
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
def backward_propagation():
    sim_label,train_sen1_list,train_sen2_list = sim_data_helper.get_train_data()
    with tf.name_scope("embedding"):
        word_vecs, _ = sim_data_helper.word_pro(FLAGS.embedding_size)
        input_x1 = tf.nn.embedding_lookup(word_vecs, input_1)
        input_x2 = tf.nn.embedding_lookup(word_vecs, input_2)
        #注意此处只能用tf.expand_dims()不能用np.expand_dims()，因为此处还没feed进去值
        input_x1 = tf.expand_dims(input_x1, -1)
        input_x2 = tf.expand_dims(input_x2, -1)
    #获得one_hot的label
    label=sim_data_helper.label_process(sim_label)
    # 初始化模型
    sim_cnn = cnn_textsim_model.MPCNN_Layer(FLAGS.num_classes,FLAGS.embedding_size,filter_size,[FLAGS.num_filters_A, FLAGS.num_filters_B], FLAGS.n_hidden,input_x1, input_x2, input_y,keep_prob,FLAGS.l2_reg_lambda)
    out=sim_cnn.similarity_measure_layer()
    # 计算损失值,和以往的不同,并没有使用交叉熵函数而是均方误差
    #这是为所有需要训练的变量加上l2正则化
    reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
    losses=tf.reduce_sum(tf.square(tf.subtract(input_y,out))) + reg
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=out)
    # losses = tf.reduce_mean(loss)
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # optimizer = tf.train.AdamOptimizer(FLAGS.init_learning_rate)
    # grads_and_vars = optimizer.compute_gradients(losses)
    # train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    train_step=tf.train.AdamOptimizer(FLAGS.init_learning_rate).minimize(losses)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 划分训练集和测试集，注意此处是单词的索引并不是单词对应的向量
        sen1_index = sim_data_helper.get_index_array(train_sen1_list,FLAGS.sentence_length, FLAGS.embedding_size)
        sen2_index = sim_data_helper.get_index_array(train_sen2_list,FLAGS.sentence_length, FLAGS.embedding_size)
        #批量获得数据
        data_num=int(len(label))
        num_inter = int(data_num / FLAGS.batch_size)
        for ite in range(FLAGS.num_epochs):
            for i in range(num_inter):
                start = i * FLAGS.batch_size
                end = (i + 1) * FLAGS.batch_size
                feed_dict = {input_1:sen1_index[start:end],input_2:sen2_index[start:end],input_y: label[start:end], keep_prob: FLAGS.keep_prob}
                # 生成summary
                if i % 30 == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict={input_1: sen1_index[start:end], input_2: sen2_index[start:end],
                                   input_y: label[start:end], keep_prob: 1.0})
                    print("epoch %d Step %d accuracy is %f" % (ite,i, train_accuracy))
                sess.run(train_step, feed_dict=feed_dict)
        # print("test accuracy %g" % accuracy.eval(feed_dict={x: X_test, y: y_test, keep_prob: 1.0}))
if __name__ == '__main__':
    backward_propagation()