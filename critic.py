import tensorflow as tf


kernel_init  = tf.contrib.layers.xavier_initializer()
bias_init    = tf.constant_initializer(0.01)
uniform_init = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)
regularizer  = tf.contrib.layers.l2_regularizer(scale=0.0)
optimizer    = tf.train.AdamOptimizer
layer1 = 400
layer2 = 300
layerout = 1

class CriticNetwork(object):
   """
   An critic class with deep neural network.

      - Critic network train with predicted Q values for actor.
      - Critic network predict Q values for input states and actor action.
   """

   def __init__(self, sess, scope, state_size, action_size, lr, n_params):
      self.sess = sess
      self.scope = scope
      self.state_size = state_size
      self.action_size = action_size
      self.lr = lr
      self.n_params = n_params

      # Create actor and get network params
      self.state, self.action, self.weights, self.output = self.build_model()

      # Feed this placeholder with: targets Q predicts by actor and critic, and
      # then apply Bellman to adjust Q value predicted.
      self.q_predicted = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='q_predicted')

      # Train actor network with critic network gradients.
      self.cost = tf.reduce_mean(tf.square(self.q_predicted - self.output))
      self.backprop = optimizer(self.lr).minimize(self.cost)

      # Gradients for feed actor network: action gradients.
      self.gradients = tf.gradients(self.output, self.action)


   def get_params(self):
      """Get params from neural network."""
      return self.weights


   def get_num_params(self):
      """Get number of network parameters."""
      return len(self.weights)


   def train(self, state, action, q_predicted):
      """Feed critic neural network with state and actions and Q value predicted from actor and critic targets networks."""
      return self.sess.run([self.output, self.backprop], feed_dict={self.state: state, self.action: action, self.q_predicted: q_predicted})
   

   def predict(self, state, action):
      """Feed actor neural network with state for predict next actions."""
      return self.sess.run(self.output, feed_dict={self.state: state, self.action: action})
   

   def calculate_gradients(self, state, action):
      """Gradients for feed actor network: critic action gradients."""
      return self.sess.run(self.gradients, feed_dict={self.state: state, self.action: action})


   def build_model(self):
      """Build neural network with differents scopes: critic or target critic."""
      with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
         state  = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size], name='critic_state')
         action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name='critic_action')
         inputs = tf.concat([state, action],1)

         fc1 = tf.layers.dense(inputs=inputs, units=layer1, activation=tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='critic_layer1')
         fc2 = tf.layers.dense(inputs=fc1, units=layer2, activation=tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='critic_layer2')
         output = tf.layers.dense(inputs=fc2, units=layerout, activation=None, kernel_initializer=uniform_init, bias_initializer=bias_init, name='critic_output')
         
         return state, action, tf.trainable_variables()[self.n_params:], output