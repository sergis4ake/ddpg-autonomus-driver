import numpy as np
from argparse import ArgumentParser
from pprint import pprint
from utils import mkdir_resources, to_json
import training
import testing


if __name__ == '__main__':
   parser = ArgumentParser(description='provide arguments for DDPG agent')
   parser.add_argument('-a', '--actor-lr', metavar='ACTOR-LR', help='learning rate for actor networks', type=float, default=0.00003)
   parser.add_argument('-c', '--critic-lr', metavar='CRITIC-LR', help='learning rate for critic networks', type=float, default=0.0007)
   parser.add_argument('--batch-size', metavar='BATCH', help='batch size for train neural network', type=int, default=64)
   parser.add_argument('--buffer-size', metavar='BUFFER', help='max size for buffer experiences', type=int, default=10000)
   parser.add_argument('--gamma', metavar='G', help='coefficient for critic networks updates', type=float, default=0.99)
   parser.add_argument('--tau', metavar='T', help='coefficient for update target networks parameters', type=float, default=0.001)
   parser.add_argument('--seed', metavar='SEED', help='random seed for repeatability', type=int, default=777)
   parser.add_argument('--max-episodes', metavar='EPISODES', help='max num of episodes to do while training', type=int, default=1000)
   parser.add_argument('--max-steps', metavar='STEPS', help='max length of 1 episode', type=int, default=2000)
   parser.add_argument('--state-size', metavar='SIZE', help='number of dimension of torcs env return', type=int, default=29)
   parser.add_argument('--action-size', metavar='SIZE', help='number of actions to send for torcs env', type=int, default=3)
   parser.add_argument('--epsilon-rate', metavar='EPSILON', help='decay for noise Ornstein–Uhlenbeck process', type=float, default=1)
   parser.add_argument('--init-episode', metavar='N', help='0 if train new agent; > 1 if restore networks and training', type=int, default=0)
   parser.add_argument('--file', metavar='NAME', help='name of file result', type=str, default='7')
   parser.add_argument('--resources', metavar='PATH', help='relative or absolute path of resurces', type=str, default='resources')
   args = vars(parser.parse_args())

   print("\nRun torcs env with this configuration?\n")
   pprint(args)

   # Create directories
   mkdir_resources(args)

   while (True):
      question = input("\n\nPlease, press one key:\n 1. TRAIN\n 2. TEST\n 0. EXIT\n")
      question = int(question)
      if (question == 1):
         to_json(args)
         training.run(args)
      elif (question == 2):
         testing.run(args)
      elif (question == 0):
         sys.exit(0)
      else:
         continue
