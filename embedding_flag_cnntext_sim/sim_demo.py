#encoding=utf-8
import numpy as np
import tensorflow as tf
a=np.arange(60).reshape([3,4,1,5])
b=tf.unstack(a,axis=1)
with tf.Session() as sess:
    print(sess.run(b))