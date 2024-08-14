"""
Test fluid sim nn with initial conditions from dataset.
"""
import os
import pygame
import torch
from model import SimpleCNN_BPTT
from data import load_data
from render_state import render_state

out_dir = 'out'
device = 'cuda'
dtype = torch.float32
compile = False
dataset = 'small_1'
# overrides from command line or config file
exec(open('configurator.py').read())

# TODO: seed?
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model = SimpleCNN_BPTT(**checkpoint['model_args'],
                       inference=True, top_kernel_size=3)


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
metadata, X, Y, normalize_transform = load_data(data_dir)
# normalize the data
X = normalize_transform(X)
Y = normalize_transform(Y)
# send to device
X = X.to(device, dtype=dtype)
Y = Y.to(device, dtype=dtype)


if __name__ == '__main__':
  # setup pygame render loop, using render_state
  # we only want to pull from data for the initial state. the model will predict the rest iteratively.
  pygame.init()
  width = metadata['width']
  height = metadata['height']
  screen = pygame.display.set_mode((1024, 1024))
  clock = pygame.time.Clock()
  running = True
  state = X[0].unsqueeze(0)
  state.requires_grad = False
  # get the initial global properties
  total_x_y = (state[0][0] ** 2 + state[0][1] ** 2).sqrt().sum()
  total_avg_velocity = state[0][2].sum()
  total_kinetic_energy = state[0][3].sum()
  total_density = state[0][4].sum()

  # extend the width and height of the state with zeros to match the screen
  state = torch.nn.functional.pad(state, (0, 256 - width, 0, 256 - height))
  with torch.set_grad_enabled(False):
    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
      screen.fill((0, 0, 0))
      state, _ = model(state)
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

      surface = render_state(state[:, 2:])
      screen.blit(surface, (0, 0))
      # screen is larger than surface, so scale up
      pygame.transform.scale(surface, (1024, 1024), screen)

      pygame.display.flip()
      clock.tick(15)
