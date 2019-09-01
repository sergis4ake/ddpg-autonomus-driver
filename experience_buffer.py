import numpy as np
from collections import deque
import random
import sys

class ExperienceBuffer(object):

   def __init__(self, buffer_size, seed=777):
      """
      The right side of the deque contains the most recent experiences.
      """
      self.buffer_size = buffer_size
      self.buffer = deque(maxlen=buffer_size)
      self.counter = 0
      random.seed(seed)

   def append(self, state_t0, action, reward, fin, state_t1):
      experience = (state_t0, action, reward, fin, state_t1)
      self.buffer.append(experience)
      self.counter += 1

   def clear(self):
      self.count = 0
      self.buffer.clear()

   def get_batch(self, batch_size):
      state_t0_batch = list()
      state_t1_batch = list()
      action_batch   = list()
      reward_batch   = list()
      fin_batch      = list()
      batch          = list()

      batch = random.sample(self.buffer, self.counter) if self.counter < batch_size else random.sample(self.buffer, batch_size)
      state_t0_batch = np.array([experience[0] for experience in batch])
      state_t1_batch = np.array([experience[4] for experience in batch])
      action_batch   = np.array([experience[1] for experience in batch])
      reward_batch   = np.array([experience[2] for experience in batch])
      fin_batch      = np.array([experience[3] for experience in batch])

      return state_t0_batch, action_batch, reward_batch, fin_batch, state_t1_batch
   
   def __len__(self):
      return len(self.buffer)