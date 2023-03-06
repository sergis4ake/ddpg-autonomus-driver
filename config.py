from pattern import Singleton
from argparse import ArgumentParser
import utils

CONFIG_PATH = './resources/config.json'


class Configuration(metaclass=Singleton):
    """Configuration from class or config file."""

    def __init__(self):
        self.config_path = './resources/config.json'
        self.actor_lr = 0.00003
        self.critic_lr = 0.0007
        self.batch_size = 64
        self.buffer_size = 10000
        self.gamma = 0.99
        self.tau = 0.001
        self.seed = 7
        self.max_episodes = 1000
        self.max_steps = 2000
        self.state_size = 29
        self.action_size = 3
        self.epsilon_rate = 1
        self.init_episode = 0
        self.file = '1'
        self.resources = './resources'
        self.tensorboard = True
        # Load config
        if CONFIG_PATH:
            self.args = self.load_from_file(CONFIG_PATH)
        else:
            self.args = self.parse_arguments()

    def parse_arguments(self):
        """Parse arguments as configuration app."""
        parser = ArgumentParser(description='provide arguments for DDPG agent')
        parser.add_argument('-C', '--config', metavar='CONFIG', help='path to configuration file',
                            type=str, default=self.config_path)
        parser.add_argument('-a', '--actor-lr', metavar='ACTOR-LR',
                            help='learning rate for actor networks', type=float,
                            default=self.actor_lr)
        parser.add_argument('-c', '--critic-lr', metavar='CRITIC-LR',
                            help='learning rate for critic networks', type=float,
                            default=self.critic_lr)
        parser.add_argument('--batch-size', metavar='BATCH',
                            help='batch size for train neural network', type=int,
                            default=self.batch_size)
        parser.add_argument('--buffer-size', metavar='BUFFER',
                            help='max size for buffer experiences', type=int,
                            default=self.batch_size)
        parser.add_argument('--gamma', metavar='G', help='coefficient for critic networks updates',
                            type=float, default=self.gamma)
        parser.add_argument('--tau', metavar='T',
                            help='coefficient for update target networks parameters', type=float,
                            default=self.tau)
        parser.add_argument('--seed', metavar='SEED', help='random seed for repeatability',
                            type=int, default=self.seed)
        parser.add_argument('--max-episodes', metavar='EPISODES',
                            help='max num of episodes to do while training', type=int,
                            default=self.max_episodes)
        parser.add_argument('--max-steps', metavar='STEPS', help='max length of 1 episode',
                            type=int, default=self.max_steps)
        parser.add_argument('--state-size', metavar='SIZE',
                            help='number of dimension of torcs env return', type=int,
                            default=self.state_size)
        parser.add_argument('--action-size', metavar='SIZE',
                            help='number of actions to send for torcs env', type=int,
                            default=self.action_size)
        parser.add_argument('--epsilon-rate', metavar='EPSILON',
                            help='decay for noise Ornstein–Uhlenbeck process', type=float,
                            default=self.epsilon_rate)
        parser.add_argument('--init-episode', metavar='N',
                            help='0 if train new agent; > 1 if restore networks and training',
                            type=int, default=self.init_episode)
        parser.add_argument('--file', metavar='NAME', help='name of file result', type=str,
                            default=self.file)
        parser.add_argument('--resources', metavar='PATH',
                            help='relative or absolute path of resources', type=str,
                            default=self.resources)
        parser.add_argument('--tensorboard', metavar='TENSORBOARD',
                            help='save graph of tensorboard for visualize with tensorboard',
                            type=bool, default=self.tensorboard)
        return vars(parser.parse_args())

    def load_from_file(self, path=None):
        """Read configuration from file."""
        if path:
            self.config_path = path
        return utils.read_json(self.config_path)
