#encoding=utf-8
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from embedding_flag_cnntext_sim.utils import *
def init_weight(shape,name):
    init=tf.truncated_normal(shape=shape)
    return tf.Variable(init,name=name)
class MPCNN_Layer():
    def __init__(self,num_classes,embedding_size,filter_sizes,num_filters,n_hidden,
                 input_x1,input_x2,input_y,dropout_keep_prob,l2_reg_lambda):
        '''
           :param num_classes:  6,代表6种类别。即输出y的维度
           :param embedding_size: 词向量维度
           :param filter_sizes: 卷积窗口大小。此处为列表【1,2,100】100表示对整个句子直接卷积。
           :param num_filters: 卷积核数量，这里为列表【num_filters_A，num_filters_B】分别为20,20.论文中A为300
           :param n_hidden:全连接层的神经元个数
           :param input_x1:输入句子矩阵。shape为【batch_size,sentence_length， embed_size，1】
           :param input_x2:同inpt_x1
           :param input_y:输出6维的array。one-hot编码
           :param dropout_keep_prob:dropout比率
        '''
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes

        self.poolings = [tf.reduce_max, tf.reduce_min, tf.reduce_mean]

        self.input_x1 = input_x1
        self.input_x2 = input_x2
        self.input_y = input_y
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = l2_reg_lambda
        # Block_A的参数。因为有三种窗口尺寸，所以初始化三个参数
        self.W1 = [init_weight([filter_sizes[0], embedding_size, 1, num_filters[0]], "W1_0"),
                   init_weight([filter_sizes[1], embedding_size, 1, num_filters[0]], "W1_1"),
                   init_weight([filter_sizes[2], embedding_size, 1, num_filters[0]], "W1_2")]
        self.b1 = [tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_0"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_1"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_2")]
        # Block_B的参数。这里只需要两种窗口尺寸（舍弃100的窗口），所以初始化两个参数
        self.W2 = [init_weight([filter_sizes[0], embedding_size, 1, num_filters[1]], "W2_0"),
                   init_weight([filter_sizes[1], embedding_size, 1, num_filters[1]], "W2_1")]
        self.b2 = [tf.Variable(tf.constant(0.1, shape=[num_filters[1], embedding_size]), "b2_0"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[1], embedding_size]), "b2_1")]
        # 卷积层经过句子相似计算之后的输出flatten之后的尺寸。用于生成隐藏层的参数。具体会在后面介绍
        self.h = num_filters[0] * len(self.poolings) * 2 + \
                 num_filters[1] * (len(self.poolings) - 1) * (len(filter_sizes) - 1) * 3 + \
                 len(self.poolings) * len(filter_sizes) * len(filter_sizes) * 3
        #全连接层参数
        self.Wh = tf.Variable(tf.random_normal([604, n_hidden], stddev=0.01), name='Wh')
        self.bh = tf.Variable(tf.constant(0.1, shape=[n_hidden]), name="bh")
        #输出层参数
        self.Wo = tf.Variable(tf.random_normal([n_hidden, num_classes], stddev=0.01), name='Wo')
        self.bo = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bo")
    #模块A通过对1个2个所有单词进行卷积来提取特征，提取特征之后再分别全进行最大，最小以及平均池化操作
    def bulid_block_A(self,x):
        out=[]
        with tf.name_scope('bulid_block_A'):
            for pooling in self.poolings:
                pools=[]
                for i,ws in enumerate(self.filter_sizes):
                    conv=tf.nn.conv2d(x,self.W1[i],strides=[1,1,1,1],padding='VALID')
                    conv=slim.batch_norm(conv,activation_fn=tf.nn.tanh)
                    pool=pooling(conv,axis=1)
                    pools.append(pool)
                out.append(pools)
            return out
    #要对单词的每个维度进行卷积，应该把数据在单词维度的那维数据上进行拆分
    def per_dim_conv_layer(self,x,w,b,pooling):
        # 为了实现per_dim的卷积。所以我们要将输入和权重偏置参数在embed_size维度上进行unstack
        input_unstack=tf.unstack(x,axis=2)
        w_unstack=tf.unstack(w,axis=1)
        b_unstack=tf.unstack(b,axis=1)
        convs=[]
        # 对每个embed_size维度进行卷积操作
        for i in range(x.get_shape()[2]):
            # conv1d要求三维的输入，三维的权重（没有宽度，只有长度。所以称为1d卷积）
            conv=tf.nn.conv1d(input_unstack[i],w_unstack[i],stride=1,padding='VALID')
            conv=slim.batch_norm(conv,activation_fn=tf.nn.tanh)
            # [batch_size, sentence_length-ws+1, num_filters_A]
            convs.append(conv)
        # 将embed_size个卷积输出在第三个维度上进行进行stack。所以又获得了一个4位的tensor
        conv=tf.stack(convs,axis=2)# [batch_size, sentence_length-ws+1, embed_size, num_filters_A]
        # 池化。即对第二个维度的sentence_length-ws+1个值取最大、最小、平均值等操作
        pool=pooling(conv,axis=1)# [batch_size, embed_size, num_filters_A]
        return pool
    def bulid_block_B(self,x):
        out = []
        with tf.name_scope("bulid_block_B"):
            for pooling in self.poolings[:-1]:
                pools = []
                with tf.name_scope("conv-pool"):
                    for i, ws in enumerate(self.filter_sizes[:-1]):
                        with tf.name_scope("per_conv-pool-%s" % ws):
                            pool = self.per_dim_conv_layer(x, self.W2[i], self.b2[i], pooling)
                        pools.append(pool)
                    out.append(pools)
            return out

    def similarity_sentence_layer(self):
        # 对输入的两个句子进行构建block_A。
        # sent1,2都是3*3*[batch_size，1， num_filters_A]的嵌套列表
        sent1 = self.bulid_block_A(self.input_x1)
        sent2 = self.bulid_block_A(self.input_x2)
        fea_h = []
        # 实现算法1
        with tf.name_scope("cal_dis_with_alg1"):
            for i in range(3):
                # 将max，men，mean三个进行连接
                regM1 = tf.concat(sent1[i], 1)
                regM2 = tf.concat(sent2[i], 1)
                # 按照每个维度进行计算max，men，mean三个值的相似度。可以参考图中绿色框
                for k in range(self.num_filters[0]):
                    # comU2计算两个tensor的距离，参见上篇博文，得到一个（batch_size，2）的tensor。2表示余弦距离和L2距离
                    fea_h.append(comU2(regM1[:, :, k], regM2[:, :, k]))
        # 得到fea_h是一个长度3*20=60的list。其中每个元素都是（batch_size，2）的tensor
        fea_a = []
        # 实现算法2的2-9行
        with tf.name_scope("cal_dis_with_alg2_2-9"):
            for i in range(3):
                for j in range(len(self.filter_sizes)):
                    for k in range(len(self.filter_sizes)):
                        # comU1计算两个tensor的距离，参见上篇博文，上图中的红色框。得到一个（batch_size，3）的tensor。3表示余弦距离和L2距离，L1距离
                        fea_a.append(comU1(sent1[i][j][:, 0, :], sent2[i][k][:, 0, :]))
        # 得到fea_a是一个长度为3*3*3=27的list。其中每个元素是（batch_size，3）的tensor

        # 对输入的两个句子进行构建block_B。
        # sent1,2都是2*2*[batch_size，50， num_filters_B]的嵌套列表
        sent1 = self.bulid_block_B(self.input_x1)
        sent2 = self.bulid_block_B(self.input_x2)

        fea_b = []
        # 实现算法2的剩余行
        with tf.name_scope("cal_dis_with_alg2_last"):
            for i in range(len(self.poolings) - 1):
                for j in range(len(self.filter_sizes) - 1):
                    for k in range(self.num_filters[1]):
                        fea_b.append(comU1(sent1[i][j][:, :, k], sent2[i][j][:, :, k]))
        ##得到fea_b是一个长度为2*2*20=80的list。其中每个元素是（batch_size，3）的tensor
        return tf.concat(fea_h + fea_a + fea_b, 1)

    def similarity_measure_layer(self):
        # 调用similarity_sentence_layer函数获得句子的相似性向量
        fea = self.similarity_sentence_layer()
        # fea_h.extend(fea_a)
        # fea_h.extend(fea_b)
        # print len(fea_h), fea_h
        # fea = tf.concat(fea_h+fea_a+fea_b, 1)
        # print fea.get_shape()
        with tf.name_scope("full_connect_layer"):
            h = tf.nn.tanh(tf.matmul(fea, self.Wh) + self.bh)
            h = tf.nn.dropout(h, self.dropout_keep_prob)
            o = tf.matmul(h, self.Wo)+self.bo
            o=tf.nn.softmax(o)
            return o