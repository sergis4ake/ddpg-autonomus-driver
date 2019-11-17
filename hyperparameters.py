import tensorflow as tf
from pattern import Singleton

class Hyperparameters(metaclass=Singleton):
   def __init__(self):
      self.kernel_init  = tf.contrib.layers.xavier_initializer()
      self.bias_init    = tf.constant_initializer(0.01)
      self.uniform_init = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)
      self.regularizer  = tf.contrib.layers.l2_regularizer(scale=0.0)
      self.optimizer    = tf.train.AdamOptimizer
      self.layer1 = 400
      self.layer2 = 300
      self.layerout = 1
