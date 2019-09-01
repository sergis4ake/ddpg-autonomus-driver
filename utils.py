import tensorflow as tf
import numpy as np
from pprint import pprint
import json, os, errno, sys
from time import time

def to_json(args):
   """Save to JSON file the params of the training or testing case."""
   with open(os.path.join(args['resources'], "params") + "/" + args['file'] + '_train.json', 'w') as f:
      json.dump(args, f)

def mkdir_resources(args):
   """Create resources directory to save and load"""
   try:
      if not os.path.exists(args['resources']):
         os.mkdir(args['resources'])
      if not os.path.exists(args['resources'] + '/network'):
         os.mkdir(args['resources'] + '/network')
      if not os.path.exists(args['resources'] + '/params'):
         os.mkdir(args['resources'] + '/params')
      if not os.path.exists(args['resources'] + '/data'):
         os.mkdir(args['resources'] + '/data')
      if not os.path.exists(args['resources'] + '/plots'):
         os.mkdir(args['resources'] + '/plots')
   except OSError as exc:
      if exc.errno != errno.EEXIST:
         raise
      pass

def noise(x, epsilon_decay, dt=0.2):
   """Noise signal"""
   mu = np.array([0.0, 0.5, 0.01])
   theta = np.array([0.0, 0.0, 0.0])
   sigma = np.array([0.1, 0.1, 0.1])
   return theta * (mu - x) + sigma * np.random.randn(sigma.shape[0]) * max(epsilon_decay, 0.0)

def calculate_epsilon_decay(max_episodes, epsilon_rate):
   """Calculate discount factor to apply epsilon each iteration episode."""
   return 1/max_episodes*epsilon_rate

def transfer_network_params(sess, network, target_network, tau):
   """ Copy parameters for each N iterations from network to target network, actor or critic.
   """
   # Assign network params to target params
   transfer_params = [var_target.assign(var_net, use_locking=False) for var_net,var_target in zip(tf.trainable_variables(network.scope), tf.trainable_variables(target_network.scope))]
   sess.run(transfer_params)
