"""
Test fluid sim nn with initial conditions from dataset.
"""
import os
import pygame
import torch
from matplotlib import pyplot as plt
import math
from model import SimpleCNN_BPTT, NearestNeighbor
from data import load_recordings, split_recordings
from render_state import render_state

out_dir = '../models/5_cups_1_model'
device = 'cuda'
dtype = torch.float32
compile = False
dataset = '5_cups'
# overrides from command line or config file
exec(open('configurator.py').read())


class ModelInference:
  def __init__(self, model, unnormalized_state: torch.tensor, normalize: torch.nn.Module,
               walls: torch.tensor, screen_size=(1024, 1024), state_extend="zeros", desired_state_size=128):
    """State should be of shape (N, C, H, W). state_extend can be zeros or replicate"""
    pygame.init()

    assert len(unnormalized_state.shape) == 4
    self.size = (unnormalized_state.shape[-1], unnormalized_state.shape[-2])
    self.screen_size = screen_size

    self.model = model
    self.walls = walls
    self.desired_state_size = desired_state_size

    if self.walls is not None:
      walls_size = (walls.shape[-1], walls.shape[-2])
      assert walls_size == self.size
      assert len(walls.shape) == 2
    else:
      self.walls = torch.zeros(self.size)

    self.screen: pygame.Surface = pygame.display.set_mode(screen_size)
    self.clock = pygame.time.Clock()

    self.post_process = lambda s: s
    self.state_to_rgb = lambda s: s[:, 2:]

    # extend state
    if state_extend == "zeros":
      x_pad_amount = (desired_state_size - self.size[0]) // 2
      y_pad_amount = (desired_state_size - self.size[1]) // 2
      pad = (x_pad_amount, x_pad_amount, y_pad_amount, y_pad_amount)
      unnormalized_state = torch.nn.functional.pad(unnormalized_state, pad)
      self.walls = torch.nn.functional.pad(self.walls, pad)
    elif state_extend == "replicate":
      replicate_amount = desired_state_size // self.size[0]
      unnormalized_state = unnormalized_state.repeat(
          1, 1, replicate_amount, replicate_amount)
      self.walls = self.walls.repeat(replicate_amount, replicate_amount)

    # normalize & store state
    self.state = normalize(unnormalized_state)

  def run(self):
    with torch.no_grad():
      while self.step():
        pygame.display.flip()
        self.clock.tick(60)

  def step(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return False
    self.screen.fill((0, 0, 0))

    # draw walls on mouse click & drag
    x, y = pygame.mouse.get_pos()
    s = self.desired_state_size
    x = x * s // pygame.display.get_window_size()[0]
    y = y * s // pygame.display.get_window_size()[1]
    # confine to window
    x = min(x, pygame.display.get_window_size()[0] - 1)
    y = min(y, pygame.display.get_window_size()[1] - 1)
    if pygame.mouse.get_pressed()[0]:
      self.walls[y, x] = 1
    elif pygame.mouse.get_pressed()[2]:
      self.walls[y, x] = 0

    # TODO: this is a hack to get the state to the right shape
    # model needs (1, 2, channels, height, width)
    # current shape is (1, channels, height, width)
    self.state = self.state.unsqueeze(0)  # (1, 1, channels, height, width)
    # (1, 2, channels, height, width)
    self.state = self.state.repeat(1, 2, 1, 1, 1)
    self.state, _ = self.model(self.state, walls=self.walls)
    # undo the shape manipulation
    self.state = self.state[:, 1, :, :, :]

    # zero out the wall positions
    # self.state = self.state * (1 - self.walls)

    # run post_process
    self.state = self.post_process(self.state)

    # render and display state
    surface = render_state(self.state_to_rgb(self.state))
    surface = pygame.transform.scale(surface, self.screen_size)
    self.screen.blit(surface, (0, 0))

    return True

  def set_post_process(self, post_process):
    self.post_process = post_process

  def set_state_to_rgb(self, state_to_rgb):
    self.state_to_rgb = state_to_rgb


def load_model(model_class):
  # init from a model saved in a specific directory
  ckpt_path = os.path.join(out_dir, 'ckpt.pt')
  checkpoint = torch.load(ckpt_path, map_location=device)
  model = model_class(**checkpoint['model_args'])

  state_dict = checkpoint['model']
  unwanted_prefix = '_orig_mod.'
  for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
      state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  model.load_state_dict(state_dict)

  model.eval()
  model.to(device)
  if compile:
    model = torch.compile(model)
  return model


if __name__ == '__main__':
  # load the dataset
  data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', dataset)
  metadata, data, normalize_transform, wall_data = load_recordings(data_dir)
  # load the model
  # model = load_model(SimpleCNN_BPTT)
  frame_start = 600
  model = NearestNeighbor(normalize_transform(
      data[0][frame_start:frame_start+6]), wall_data * 100, 3)
  model.eval()
  model.to(device)

  # grab first recording
  data = data[0]
  # grab a frame
  state = data[frame_start-5].unsqueeze(0)
  # send to device
  state = state.to(device, dtype=dtype)
  wall_data = wall_data.to(device, dtype=dtype)

  # create ModelInference
  infer = ModelInference(model, state, normalize_transform,
                         wall_data, state_extend='replicate')
  infer.run()
