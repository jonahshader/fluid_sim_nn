import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms


def load_data(path):
  # load metadata
  with open(os.path.join(path, "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)

  # determine how many frames we have, and identify where the breaks are between recordings
  # count files that match frame_*.bin
  frames = [f for f in os.listdir(path) if f.startswith(
      "frame_") and f.endswith(".bin")]

  # sort the frames
  frames.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
  # pull out the frame number
  frame_numbers = [int(f.split("_")[1].split(".")[0]) for f in frames]

  # identify the start of each recording
  starts = [0]
  for i in range(1, len(frames)):
    if frame_numbers[i] != frame_numbers[i - 1] + 1:
      starts.append(i)

  # load the data, convert to pytorch tensors
  data = [np.memmap(os.path.join(path, f), dtype=np.float32, mode='r')
          for f in frames]
  data = [torch.tensor(d, dtype=torch.float32) for d in data]

  # reshape the data to match the metadata
  width = metadata["width"]
  height = metadata["height"]
  channels = len(metadata["attributes"])
  data = [d.reshape(channels, height, width) for d in data]
  # combine the frames into a single tensor
  data = torch.stack(data)

  # make a normalize transform
  # compute mean and std, reduce
  channel_means = data.mean(dim=[0, 2, 3])
  channel_stds = data.std(dim=[0, 2, 3])
  normalize = transforms.Normalize(mean=channel_means, std=channel_stds)

  # split the data into recordings
  recordings = []
  for i in range(len(starts) - 1):
    recordings.append(data[starts[i]:starts[i + 1]])

  # this model takes in frame n and predicts frame n+1.
  # construct the x and y tensors per recording, then combine
  x = []
  y = []
  for recording in recordings:
    x.append(recording[:-1])
    y.append(recording[1:])
  x = torch.cat(x)
  y = torch.cat(y)

  return metadata, x, y, normalize


def crop_data(x, y, crop_size=(2, 2)):
  # crop the data to the specified size
  _, _, height, width = x.shape
  crop_height, crop_width = crop_size
  x_start = crop_width // 2
  y_start = crop_height // 2
  x_end = width - crop_width // 2
  y_end = height - crop_height // 2
  # we only want to crop the output, not the input
  y = y[:, :, y_start:y_end, x_start:x_end]
  return x, y


def split_data(x, y, test_fraction=0.1):
  n = len(x)
  indices = np.arange(n)
  np.random.shuffle(indices)
  split = int(n * test_fraction)
  test_indices = indices[:split]
  train_indices = indices[split:]
  return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


if __name__ == "__main__":
  # test crop_data
  x = torch.randn(2, 3, 4, 4)
  y = torch.randn(2, 3, 4, 4)
  x, y = crop_data(x, y, (2, 2))
  assert x.shape == (2, 3, 4, 4)
  assert y.shape == (2, 3, 2, 2)

  print("All tests pass")
