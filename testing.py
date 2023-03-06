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
        actor_target = ActorNetwork(sess=sess, scope='actor_net_target',
                                    state_size=args['state_size'],
                                    action_size=args['action_size'], batch_size=args['batch_size'],
                                    lr=args['actor_lr'], n_params=n_params)

        # Critic and critic target
        n_params += actor_target.get_num_params()
        critic = CriticNetwork(sess=sess, scope='critic_net', state_size=args['state_size'],
                               action_size=args['action_size'], lr=args['critic_lr'],
                               n_params=n_params)
        n_params += critic.get_num_params()
        critic_target = CriticNetwork(sess=sess, scope='critic_net_target',
                                      state_size=args['state_size'],
                                      action_size=args['action_size'], lr=args['critic_lr'],
                                      n_params=n_params)

        # Restore network params
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(os.path.join(args['resources'], "network"),
                                         args['file'] + '_model'))

        # Train DDPG on Torcs
        test(sess, env, actor, actor_target, critic, critic_target)


def test(sess, env, actor, actor_target, critic, critic_target):
    max_episodes = args['max_episodes']
    max_steps = args['max_steps']
    epsilon_rate = args['epsilon_rate']
    epsilon_decay = calculate_epsilon_decay(max_episodes, epsilon_rate)
    epsilon = 1
    relaunch_torcs_every = 1

    for i in range(max_episodes):

        # Relaunch torcs
        ob = env.reset(relaunch=True) if np.mod(i, relaunch_torcs_every) == 0 else env.reset()

        # Init reward counter and Q max value.
        reward_total = 0
        q_max = 0

        # Decrease noise every episode
        epsilon -= epsilon_decay

        # Get environment state
        state_t0 = np.hstack((ob.angle, ob.track, ob.trackPos,
                              ob.speedX, ob.speedY, ob.speedZ,
                              ob.wheelSpinVel / 100.0, ob.rpm))

        ###############################################################

        for j in range(max_steps):

            # Action noise: decrease noise (epsilon) for each episode to get better results
            action = actor.predict(np.reshape(state_t0, (1, actor.state_size)))
            # action[0,:] += noise(x=action[0,:], epsilon_decay=epsilon)

            # Run step and get data for enviroment.
            ob, reward, fin, info = env.step(action[0])
            state_t1 = np.hstack((ob.angle, ob.track, ob.trackPos,
                                  ob.speedX, ob.speedY, ob.speedZ,
                                  ob.wheelSpinVel / 100.0, ob.rpm))

            state_t0 = state_t1
            reward_total += reward

            print(
                '| Episode: {:5d} | Step: {:5d}  | Stel: {:2.2f}\t| Accel: {:2.2f}\t| Brake: {:2.2f}\t| Reward: {:6d} | Qmax {:6d} |'.format(
                    i, j, action[0][0], action[0][1], action[0][2], int(reward_total),
                    int(q_max / float(j + 1))))

            ###############################################################

            if fin:
                print('| Reward: {:d} | Episode: {:d} | Q-max: {:.4f}'.format(int(reward_total), i,
                                                                              (q_max / float(j))))
                filepath = os.path.join(
                    os.path.join(args['resources'], "data"), f"{args['file']}_test.txt")
                with open(filepath, "a") as f:
                    f.write(str(i) + " " + str(j) + " " + str(reward_total) + " " + str(
                        q_max / float(j + 1)) + "\n")
                break
