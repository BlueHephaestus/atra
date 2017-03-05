from keras import backend as K
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.int64, shape=[3,])
n = tf.placeholder(tf.int64)
#n = tf.reshape(n, (1,))
#n = tf.pack([n])#len(n) = 1 = len(shape(x))
#y = n
n = tf.pack([n])
y = tf.tile(x, n)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print sess.run(y, feed_dict={x: np.array([0,1,2]), n: 3})
