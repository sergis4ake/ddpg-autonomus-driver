import numpy as np
import tensorflow as tf
from gym_torcs import TorcsEnv
from actor import ActorNetwork
from critic import CriticNetwork
from experience_buffer import ExperienceBuffer
from utils import transfer_network_params, calculate_epsilon_decay, noise
from time import time
import os
from config import Configuration

# Configuration
config = Configuration()
args = config.args


def run():
   """Build networks, create environment and train agent."""

   # Generate a Torcs environment
   env = TorcsEnv(vision=False, throttle=True, gear_change=False)

   with tf.Session() as sess:
  
      np.random.seed(args['seed'])
      tf.set_random_seed(args['seed'])
      
      # Actor and actor target
      n_params = 0
      actor = ActorNetwork(sess=sess, scope='actor_net', state_size=args['state_size'], 
                           action_size=args['action_size'], batch_size=args['batch_size'], 
                           lr=args['actor_lr'], n_params=n_params)
      n_params += actor.get_num_params()
      actor_target = ActorNetwork(sess=sess, scope='actor_net_target', state_size=args['state_size'], 
                           action_size=args['action_size'], batch_size=args['batch_size'], 
                           lr=args['actor_lr'], n_params=n_params)
      
      # Critic and critic target
      n_params += actor_target.get_num_params()
      critic = CriticNetwork(sess=sess, scope='critic_net', state_size=args['state_size'], 
                           action_size=args['action_size'], lr=args['critic_lr'], n_params=n_params)
      n_params += critic.get_num_params()
      critic_target = CriticNetwork(sess=sess, scope='critic_net_target', state_size=args['state_size'], 
                           action_size=args['action_size'], lr=args['critic_lr'], n_params=n_params)
      

      # Train DDPG on Torcs
      train(sess, env, actor, actor_target, critic, critic_target)



def train(sess, env, actor, actor_target, critic, critic_target):

   # Arguments
   init_episode = args['init_episode']
   epsilon_rate = args['epsilon_rate']
   max_episodes = args['max_episodes']
   max_steps    = args['max_steps']
   batch_size   = args['batch_size']
   buffer_size  = args['buffer_size']
   gamma        = args['gamma']
   seed         = args['seed']
   tau          = args['tau']

   # Restore model or run init tensorflow
   saver = tf.train.Saver()
   sess.run(tf.global_variables_initializer()) if init_episode==0 else saver.restore(sess, os.path.join(args['resources'], "network") + "/" + args['file']+"_model")

   # Add tensorboard visualization
   # $ tensorboard --logdir=./resources/logdir
   # https://localghost:6006
   if args['tensorboard']:
      logdir = os.path.join(args['resources'], 'logdir')
      if not os.path.exists(logdir):
         os.mkdir(logdir)
      tf.summary.FileWriter(logdir, sess.graph)

   # Init experience buffer
   buffer = ExperienceBuffer(buffer_size, seed)

   counter = 0
   epsilon = 1.0
   transfer_network_params_every = 500
   relaunch_torcs_every = 1
   save_networks_every  = 50
   epsilon_decay = calculate_epsilon_decay(max_episodes, epsilon_rate)

   ###############################################################

   for i in range(init_episode, max_episodes):
      
      # relaunch TORCS every N episodes due to a memory leak error
      ob = env.reset(relaunch=True) if np.mod(i, relaunch_torcs_every) == 0 else env.reset()

      # Init reward counter and Q max value.
      reward_total = 0
      q_max = 0

      # Decrease noise every episode
      # epsilon -= epsilon_decay

      # Get enviroment state
      state_t0 = np.hstack((ob.angle, ob.track, ob.trackPos, 
                              ob.speedX, ob.speedY,  ob.speedZ, 
                              ob.wheelSpinVel/100.0, ob.rpm))

      ###############################################################

      for j in range(max_steps):
         
         t0 = time()
         counter += 1
       
         # Action noise: decrease noise (epsilon) for each episode to get better results
         action = actor.predict(np.reshape(state_t0, (1, actor.state_size)))
         action[0,:] += noise(x=action[0,:], epsilon_decay=epsilon)
           
         # The first 5 episodes full acceleration.
         if (i < 10):
            action[0][0] = 0.0
            action[0][1] = 1.0
            action[0][2] = 0.0

         # Run step and get data for enviroment.
         ob, reward, fin, info = env.step(action[0])

         # Update target networks
         if ((counter % transfer_network_params_every) == 0):
            transfer_network_params(sess, actor, actor_target, tau)
            transfer_network_params(sess, critic, critic_target, tau)
            print("\n***************************************************************************************")
            print("---------------------------- UPDATE TARGET NETWORK PARAMS -----------------------------")
            print("*************************************{:2.2E}**************************************************\n".format(time()-t0))
         else:
            state_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, 
                                 ob.speedX, ob.speedY, ob.speedZ, 
                                 ob.wheelSpinVel/100.0, ob.rpm))

            # Add new experience to buffer.
            buffer.append(np.reshape(state_t0, (actor.state_size,)), 
                                    np.reshape(action, (actor.action_size,)), 
                                    reward, fin, np.reshape(state_t1, (actor.state_size,)))

            

            if len(buffer) > batch_size:
               states_t0_batch, actions_batch, rewards_batch, fin_batch, states_t1_batch = buffer.get_batch(batch_size)

               #***********************************************#
               ## Calculate target_q for train critic network ##
               #***********************************************#

               # Predict target q; predict target_y actor & target_y critic and combine both.
               actor_q_target = actor_target.predict(states_t1_batch)
               _critic_q_target = critic_target.predict(states_t1_batch, actor_q_target)

               # Calculate target q with gamma and rewards: Bellman ecuation.
               critic_q_target = [x if fin_batch[i] else (x+gamma*_critic_q_target[i]) for i,x in enumerate(rewards_batch)]
                  
               # Train critic network with targets Q values.
               critic_q_predicted, _ = critic.train(states_t0_batch, actions_batch, np.reshape(critic_q_target, (batch_size, 1)))
               
               # Get max Q value predicted by critic.
               q_max += np.amax(critic_q_predicted)
            
               #*****************************************************************#
               ## Calculate Q actions and get gradients for train actor network ##
               #*****************************************************************#

               # Train actor network with critic gradients.
               actor_action_predict = actor.predict(states_t0_batch)
               gradients = critic.calculate_gradients(states_t0_batch, actor_action_predict)
               actor.train(states_t0_batch, gradients[0])

               # Compute actor loss: MSE
               # _actor_action_predict = actor.predict(states_t0_batch)
               # actor_loss = actor.calculate_loss(actor_action_predict, _actor_action_predict)
                  
               # print("a-loss: ", actor_loss, "c-loss", critic_loss)

            state_t0 = state_t1
            reward_total += reward

            print('| Buffer {:7} | Episode: {:5d} | Step: {:5d}  | Stel: {:2.2f}\t| Accel: {:2.2f}\t| Brake: {:2.2f}\t| Reward: {:6d} | Qmax: {:6d} | Time: {:2.2E}'.format(len(buffer), i, j, action[0][0], action[0][1], action[0][2], int(reward_total), int(q_max / float(j+1)), time()-t0))

            ###############################################################

            if fin:
               print('| Reward: {:d} | Episode: {:d} | Q-max: {:.4f}'.format(int(reward_total), i, (q_max / float(j))))
               with open(os.path.join(args['resources'], "data") + "/" + args['file'] + "_train.txt", "a") as f:
                  f.write(str(i) + " " + str(j) + " " + str(reward_total) + " " + str(q_max / float(j+1))  + "\n")
               break

      ###############################################################

      if ((i%save_networks_every) == 0 and i > 1):
         saver.save(sess, os.path.join(args['resources'], "network") + "/" + args['file']+"_model")
         print("--------------------------------- SAVED MODEL ------------------------------------")