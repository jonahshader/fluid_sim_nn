"""
Test fluid sim nn with initial conditions from dataset.
"""
import os
import pygame
import torch
from matplotlib import pyplot as plt
import math
from model import SimpleCNN_BPTT
from data import load_recordings, split_recordings
from render_state import render_state

# out_dir = '../models/out_SimpleCNN_BPTT'
out_dir = '../models/5_cups_1_model'
device = 'cuda'
dtype = torch.float32
compile = False
dataset = '5_cups'
# overrides from command line or config file
exec(open('configurator.py').read())

# TODO: seed?
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model = SimpleCNN_BPTT(**checkpoint['model_args'])


state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
  if k.startswith(unwanted_prefix):
    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
print(
    f"Padding mode: {model.conv.padding_mode}, Padding: {model.conv.padding}")

model.eval()
model.to(device)
if compile:
  model = torch.compile(model)

# load the dataset
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', dataset)
metadata, data, normalize_transform, wall_data = load_recordings(data_dir)
# grab first recording
data = data[0]
# send to device
data = data.to(device, dtype=dtype)
wall_data = wall_data.to(device, dtype=dtype)


class ModelInference:
  def __init__(self, model, state: torch.tensor, walls: torch.tensor, screen_size=(1024, 1024)):
    """State should be of shape (N, C, H, W)"""
    pygame.init()

    assert len(state.shape) == 4
    self.size = (state.shape[-1], state.shape[-2])
    self.screen_size = screen_size

    self.model = model
    self.walls = walls

    if walls:
      walls_size = (walls.shape[-1], walls.shape[-2])
      assert walls_size == self.size
      assert len(walls_size.shape) == 2

    self.screen: pygame.Surface = pygame.display.set_mode(screen_size)
    self.clock = pygame.time.Clock()

    self.post_process = lambda s: s
    self.state_to_rgb = lambda s: s[:, 2:]

  def run(self):
    with torch.no_grad():
      while self.step():
        pass

  def step(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return False
    self.screen.fill((0, 0, 0))

    # draw walls on mouse click & drag
    x, y = pygame.mouse.get_pos()
    x = x * s // pygame.display.get_window_size()[0]
    y = y * s // pygame.display.get_window_size()[1]
    # confine to window
    x = min(x, pygame.display.get_window_size()[0] - 1)
    y = min(y, pygame.display.get_window_size()[1] - 1)
    if pygame.mouse.get_pressed()[0]:
      wall_data[y, x] = 1
    elif pygame.mouse.get_pressed()[2]:
      wall_data[y, x] = 0

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
    self.state = self.state * (1 - self.walls)

    # run post_process
    self.state = self.post_process(self.state)

    # render and display state
    surface = render_state(self.state_to_rgb(self.state))
    self.screen.blit(surface, (0, 0))
    pygame.transform.scale(surface, self.screen_size)
    pygame.display.flip()
    self.clock.tick(60)
    return True

  def set_post_process(self, post_process):
    self.post_process = post_process

  def set_state_to_rgb(self, state_to_rgb):
    self.state_to_rgb = state_to_rgb


if __name__ == '__main__':
  # setup pygame render loop, using render_state
  # we only want to pull from data for the initial state. the model will predict the rest iteratively.
  pygame.init()
  width = metadata['width']
  height = metadata['height']
  screen = pygame.display.set_mode((1024, 1024))
  clock = pygame.time.Clock()
  running = True
  state = data[0].unsqueeze(0)
  state.requires_grad = False
  # get the initial global properties
  total_x_y = (state[0][0] ** 2 + state[0][1] ** 2).sqrt().sum()
  total_avg_velocity = state[0][2].sum()
  total_kinetic_energy = state[0][3].sum()
  total_density = state[0][4].sum()

  # extend the width and height of the state with zeros to match the screen
  s = 128
  state = torch.nn.functional.pad(state, (0, s - width, 0, s - height))
  # repeat the wall data to match the screen
  repetitions = (s // wall_data.shape[0], s // wall_data.shape[1])
  # wall_data = wall_data.repeat(repetitions[0], repetitions[1])
  wall_data = torch.nn.functional.pad(wall_data, (0, s - width, 0, s - height))
  # wall_data *= 0
  # normalize the state
  state = normalize_transform(state)

  # plot a histogram for each channel of state
  for i in range(state.shape[1]):
    name = metadata['attributes'][i]
    plt.hist(state[0][i].flatten().cpu().numpy(), bins=100)
    plt.title(name)
    plt.show()

  with torch.no_grad():
    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
      screen.fill((0, 0, 0))

      # draw walls on mouse click & drag
      x, y = pygame.mouse.get_pos()
      x = x * s // pygame.display.get_window_size()[0]
      y = y * s // pygame.display.get_window_size()[1]
      if pygame.mouse.get_pressed()[0]:
        wall_data[y, x] = 1
      elif pygame.mouse.get_pressed()[2]:
        wall_data[y, x] = 0

      # TODO: this is a hack to get the state to the right shape
      # model needs (1, 2, channels, height, width)
      # current shape is (1, channels, height, width)
      state = state.unsqueeze(0)  # (1, 1, channels, height, width)
      state = state.repeat(1, 2, 1, 1, 1)  # (1, 2, channels, height, width)

      state, _ = model(state, walls=wall_data)
      # undo the shape manipulation
      # state = state.squeeze(0)
      state = state[:, 1, :, :, :]

      # zero out the state where the wall is
      state = state * (1 - wall_data)

      # limit x_vel to +/- 10
      state[0][0] = torch.clamp(state[0][0], -20, 20)
      # limit y_vel to +/- 10
      state[0][1] = torch.clamp(state[0][1], -20, 20)
      # limit avg_vel to [0, 8]
      state[0][2] = torch.clamp(state[0][2], -0.5, 15)
      # limit kinetic energy to 0, 15
      state[0][3] = torch.clamp(state[0][3], 0, 40)
      # limit density to -1, 2
      state[0][4] = torch.clamp(state[0][4], -0.8, 2)

      # fix the density, kinetic energy, avg vel
      current_total_density = state[0][4].sum()
      current_total_kinetic_energy = state[0][3].sum()
      current_avg_velocity = state[0][2].sum()
      # state[0][4] *= total_density / current_total_density
      # state[0][3] *= total_kinetic_energy / current_total_kinetic_energy
      # state[0][2] *= total_avg_velocity / current_avg_velocity

      # fix the x, y squared
      # current_total_x_y = (state[0][0] ** 2 + state[0][1] ** 2).sqrt().sum()
      # state[0][0] *= total_x_y / current_total_x_y
      # state[0][1] *= total_x_y / current_total_x_y

      # # limit the x y velocity to -1, 1
      # state[0][0] = torch.clamp(state[0][0], -1, 1)
      # state[0][1] = torch.clamp(state[0][1], -1, 1)

      # state[0][0] = torch.sigmoid(state[0][0])
      # state[0][1] = torch.sigmoid(state[0][1])

      # state[0][0] *= 0.9
      # state[0][1] *= 0.99
      state[0][2] *= 1.01
      state[0][3] *= 1.01
      state[0][4] *= 1.01

      # apply tanh to state
      # state = torch.tanh(state * 1.01)

      # x_vel = state[:, 0, :, :]
      # y_vel = state[:, 1, :, :]
      # vel = (x_vel ** 2 + y_vel ** 2).sqrt() + 0.0001
      # angle = torch.pi / 16
      # rotation_matrix = [[math.cos(angle), -math.sin(angle)],
      #                    [math.sin(angle), math.cos(angle)]]
      # rotation_matrix = torch.tensor(rotation_matrix).to(device)

      # x_vel = x_vel * rotation_matrix[0, 0] + y_vel * rotation_matrix[0, 1]
      # y_vel = x_vel * rotation_matrix[1, 0] + y_vel * rotation_matrix[1, 1]

      # x_vel /= vel
      # y_vel /= vel

      # state[:, 0, :, :] = x_vel
      # state[:, 1, :, :] = y_vel

      surface = render_state(state[:, 2:])
      # surface = render_state(state[:, :3])
      screen.blit(surface, (0, 0))
      # screen is larger than surface, so scale up
      pygame.transform.scale(surface, (1024, 1024), screen)

      pygame.display.flip()
      clock.tick(1200)
