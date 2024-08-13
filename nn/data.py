import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms


def load_recordings(path):
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

  return metadata, recordings, normalize


def split_recordings(recordings, test_fraction=0.1, batch_depth=4):
  """Splits recordings into training and test sets s.t. each recording is >= batch_depth"""

  # drop recordings that are too short
  recordings = [r for r in recordings if len(r) >= batch_depth]

  batch_start_indices = []
  current_start = 0
  for recording in recordings:
    batch_start_indices.append(
        current_start + np.arange(0, len(recording) - batch_depth + 1))
    current_start += len(recording)
  batch_start_indices = np.concatenate(batch_start_indices)

  # split the data
  test_mask = np.random.rand(len(batch_start_indices)) < test_fraction
  train_indices = batch_start_indices[~test_mask]
  test_indices = batch_start_indices[test_mask]

  # combine the recordings
  combined_recordings = torch.cat(recordings)

  return train_indices, test_indices, combined_recordings
