import tensorflow as tf


kernel_init  = tf.contrib.layers.xavier_initializer()
bias_init    = tf.constant_initializer(0.01)
uniform_init = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)
regularizer  = tf.contrib.layers.l2_regularizer(scale=0.0)
optimizer    = tf.train.AdamOptimizer
layer1 = 400
layer2 = 300
layerout = 1

class ActorNetwork(object):
   """
   An actor class with deep neural network.

      - Actor network train with critic network gradients.
      - Actor network predict Q actions for input states.
   """

   def __init__(self, sess, scope, state_size, action_size, batch_size, lr, n_params):
      self.sess = sess
      self.scope = scope
      self.state_size = state_size
      self.action_size = action_size
      self.batch_size = batch_size
      self.lr = lr
      self.n_params = n_params

   
      # Create actor and get network params
      self.state, self.weights, self.output = self.build_model() 

      # Feed this placeholder with critic network gradients
      self.critic_gradients = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name='critic_gradients')
      self.gradients = tf.gradients(self.output, self.weights, -self.critic_gradients)
      self.norm_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.gradients)) #¿¿¿NORMALIZED???

      # Calculate loss for train with gradients
      # self.loss_function()

      # Train actor network with critic network gradients.
      self.backprop = optimizer(self.lr).apply_gradients(zip(self.norm_gradients, self.weights))

   def get_params(self):
      """Get params from neural network."""
      return self.weights


   def get_num_params(self):
      """Get number of network parameters."""
      return len(self.weights)


   def train(self, state, critic_gradients):
      """Feed actor neural network with state and critic gradients for train it."""
      return self.sess.run(self.backprop, feed_dict={self.state: state, self.critic_gradients: critic_gradients})
   

   def predict(self, state):
      """Feed actor neural network with state for predict next actions."""
      return self.sess.run(self.output, feed_dict={self.state: state})
   

   def loss_function(self):
      """Draw the graph with loss function: Mean Squared Error."""
      self.pred1 = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name='pred1')
      self.pred2 = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name='pred2')
      self.loss = tf.reduce_mean(tf.square(self.pred1 - self.pred2))


   def calculate_loss(self, pred1, pred2):
      """Compute loss btw two predictions."""
      return self.sess.run(self.loss, feed_dict={self.pred1: pred1, self.pred2: pred2})


   def build_model(self):
      """Build neural network with differents scopes: actor or target actor."""
      with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
         state = tf.placeholder(name='actor_state', dtype=tf.float32, shape=[None, self.state_size])
         fc1 = tf.layers.dense(inputs=state, units=layer1, activation=tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='actor_layer1')
         fc2 = tf.layers.dense(inputs=fc1, units=layer2, activation=tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='actor_layer2')

         steer = tf.layers.dense(inputs=fc2, units=layerout, activation=tf.nn.tanh, kernel_initializer=uniform_init, bias_initializer=bias_init, name='steer')
         accel = tf.layers.dense(inputs=fc2, units=layerout, activation=tf.nn.sigmoid, kernel_initializer=uniform_init, bias_initializer=bias_init, name='acc')
         brake = tf.layers.dense(inputs=fc2, units=layerout, activation=tf.nn.sigmoid, kernel_initializer=uniform_init, bias_initializer=bias_init, name='brake')
         output = tf.concat([steer, accel, brake], axis=1)

         return state, tf.trainable_variables()[self.n_params:], output