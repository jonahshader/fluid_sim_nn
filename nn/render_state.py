import pygame
import numpy as np
import torch


def render_state(state):
  """
  Render the state of the fluid simulation.
  State is a tensor of shape (channels, height, width).
  Assumes state has already been normalized.
  """

  # get width height from state
  batch, channels, height, width = state.shape
  surface = pygame.Surface((width, height))

  # convert to numpy array (state requires grad, so detach)
  state = state.detach().cpu().numpy().squeeze()

  # convert to 8-bit
  state = np.clip(state, 0, 1)
  state = (state * 255).astype(np.uint8)

  # convert to pygame surface
  # channel count is arbitrary.
  # if there is less than 3, duplicate last channel until we have 3
  # if there is more than 3, discard the rest
  if channels < 3:
    state = np.concatenate([state] * (3 // channels), axis=0)
  elif channels > 3:
    state = state[:3]

  # convert to 3 channel
  state = np.moveaxis(state, 0, -1)
  # state now has the shape (height, width, channels)

  # transpose height and width
  state = np.transpose(state, (1, 0, 2))

  # convert to pygame surface
  pygame.surfarray.blit_array(surface, state)

  return surface
