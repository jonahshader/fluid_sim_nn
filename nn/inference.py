"""
Test fluid sim nn with initial conditions from dataset.
"""
import os
import pickle
import pygame
from contextlib import nullcontext
import torch
from pathlib import Path
from model import SimpleCNN
from data import load_data
from render_state import render_state
import ast

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
model = SimpleCNN(**checkpoint['model_args'], inference=True)


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
  # extend the width and height of the state with zeros to match the screen
  state = torch.nn.functional.pad(state, (0, 256 - width, 0, 256 - height))
  with torch.set_grad_enabled(False):
    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
      screen.fill((0, 0, 0))
      state, _ = model(state)
      surface = render_state(state)
      screen.blit(surface, (0, 0))
      # screen is larger than surface, so scale up
      pygame.transform.scale(surface, (1024, 1024), screen)

      pygame.display.flip()
      clock.tick(60)
