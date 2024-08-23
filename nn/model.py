import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


class SimpleCNN_BPTT(nn.Module):
  def __init__(self, channels, top_kernel_size, skip_con=False):
    super(SimpleCNN_BPTT, self).__init__()
    self.top_kernel_size = top_kernel_size
    padding = top_kernel_size // 2
    self.conv = nn.Conv2d(channels + 1, 256, kernel_size=top_kernel_size,
                          padding_mode='circular', padding=padding)

    self.act1 = nn.GELU()
    # fc1 is pixel-wise, so the input is 32
    self.conv2 = nn.Conv2d(256, 256, kernel_size=1)
    self.act2 = nn.Tanh()
    self.conv3 = nn.Conv2d(256, channels, kernel_size=1)
    self.skip_con = skip_con

  def forward_single(self, x, y=None, walls=None):
    x_orig = x
    # result should be (batch, channels, height, width)

    # add walls as a channel. walls is of shape (height, width)
    if walls is not None:
      walls = walls.unsqueeze(0).unsqueeze(0)
      # repeat walls to match the batch size
      walls = walls.repeat(x.shape[0], 1, 1, 1)
      x = torch.cat((x, walls), dim=1)
    else:
      x = torch.cat((x, torch.zeros_like(x[:, 0:1, :, :])), dim=1)
    x = self.conv(x)
    x = self.act1(x)
    x = self.conv2(x)
    x = self.act2(x)
    x = self.conv3(x)

    if self.skip_con:
      x = x + x_orig

    if y is not None:
      # compute the loss
      loss = F.mse_loss(x, y)
    else:
      loss = None

    return x, loss

  def forward(self, x, walls=None):
    # x is of shape (batch_size, batch_depth, channels, height, width)
    batch_size, batch_depth, channels, height, width = x.shape

    steps = []
    losses = []

    # Initial step
    current_step = x[:, 0, :, :, :]
    steps.append(current_step.unsqueeze(1))

    # Iterate through the sequence
    for i in range(1, batch_depth):
      target = x[:, i, :, :, :]
      current_step, loss = self.forward_single(current_step, target, walls)
      steps.append(current_step.unsqueeze(1))
      losses.append(loss)

    # Stack all steps and losses
    all_steps = torch.cat(steps, dim=1)
    all_losses = torch.stack(losses) if losses else None

    # Take mean of losses
    if all_losses is not None:
      all_losses = all_losses.mean()

    return all_steps, all_losses


class NearestNeighbor(nn.Module):
  """Performs nearest neighbor on 3x3 kernels of a subset of the data to "lookup" the next value per kernel.
  Operates on normalized data."""

  def __init__(self, recording, walls, top_kernel_size):
    super(NearestNeighbor, self).__init__()
    assert len(recording.shape) == 4
    padding = top_kernel_size // 2
    self.top_kernel_size = top_kernel_size

    # add walls as a channel. walls is of shape (height, width)
    self.walls = walls.unsqueeze(0).unsqueeze(0)
    # repeat walls to match the batch size
    walls_repeated = self.walls.repeat(recording.shape[0], 1, 1, 1)
    recording = torch.cat((recording, walls_repeated), dim=1)

    batch_size, channels, height, width = recording.shape

    # assuming data is of shape (batch_size, channels, height, width)

    # manually apply circular padding (F.unfold only supports zero padding)
    padded = F.pad(recording, (padding, padding,
                   padding, padding), mode='circular')

    # unfold the data to get width*height kernels
    unfolded = F.unfold(padded, kernel_size=top_kernel_size, padding=0)
    # unfold produces a tensor of shape (batch_size, channels*top_kernel_size^2, height*width)
    # we want (batch_size * height * width, channels*top_kernel_size^2)
    unfolded = unfolded.transpose(1, 2).reshape(
        batch_size * height * width, channels * top_kernel_size**2)
    self.unfolded = nn.Parameter(unfolded, requires_grad=False)
    self.walls = nn.Parameter(self.walls, requires_grad=False)

  def forward_single(self, x, y=None):
    # x is of shape (batch_size, channels, height, width)
    # y is of shape (batch_size, channels, height, width)

    batch_size, channels, height, width = x.shape
    padding = self.top_kernel_size // 2

    # manually apply circular padding (F.unfold only supports zero padding)
    x = F.pad(x, (padding, padding, padding, padding), mode='circular')

    # unfold the input to get width*height kernels
    x_unfolded = F.unfold(x, kernel_size=self.top_kernel_size, padding=0)
    # unfold produces a tensor of shape (batch_size, channels*top_kernel_size^2, height*width)
    # we want (batch_size, height * width, channels*top_kernel_size^2)
    x_unfolded = x_unfolded.transpose(1, 2).reshape(
        batch_size, height * width, channels * self.top_kernel_size**2)

    # don't include last batch of unfolded
    unfolded = self.unfolded[:-height*width]
    unfolded_target = self.unfolded[height*width:]

    # we want to compute the distance between each kernel in x and each kernel in self.unfolded,
    # so we need to repeat self.unfolded to match the batch size
    unfolded = unfolded.repeat(batch_size, 1, 1)

    # compute the distance between each kernel in x and each kernel in self.unfolded
    # distance is of shape (batch_size, height*width, height*width)
    distance = torch.cdist(x_unfolded, unfolded, p=2)

    # add a bit of noise to break ties
    # distance += torch.rand_like(distance)

    # get the index of the minimum distance for each kernel
    # index is of shape (batch_size, height*width)
    index = distance.argmin(dim=2)

    # get the value of the minimum distance for each kernel
    value = unfolded_target[index]
    # value is of shape (batch_size, height*width, channels*top_kernel_size^2)
    # we want (batch_size, height*width, channels), grabbing the middle of the kernel
    middle = self.top_kernel_size**2 // 2
    # value = value[:, :, middle * channels:(middle + 1) * channels]
    # value = value.reshape(batch_size, height * width,
    #                       self.top_kernel_size**2, channels)
    # value = value[:, :, middle, :].squeeze(2)

    value = value.reshape(batch_size, height * width,
                          channels, self.top_kernel_size**2)
    value = value[:, :, :, middle].squeeze(2)

    # reshape value to match the input shape
    # value is of shape (batch_size, height*width, channels)
    # need to transpose first to get (batch_size, channels, height*width)
    value = value.transpose(1, 2)
    # then reshape to get (batch_size, channels, height, width)
    value = value.reshape(batch_size, channels, height, width)

    if y is not None:
      # compute the loss
      loss = F.mse_loss(value, y)
    else:
      loss = None

    return value, loss

  def forward(self, x, walls=None):
    # x is of shape (batch_size, batch_depth, channels, height, width)
    batch_size, batch_depth, channels, height, width = x.shape

    # add walls as a channel. walls is of shape (height, width)
    if walls is not None:
      walls_repeated = walls.unsqueeze(0).unsqueeze(0).unsqueeze(0)
      # repeat walls to match the batch size and batch depth
      walls_repeated = walls_repeated.repeat(batch_size, batch_depth, 1, 1, 1)
      x = torch.cat((x, walls_repeated), dim=2)
    elif self.walls is not None:
      walls_repeated = self.walls.unsqueeze(0)
      # repeat walls to match the batch size and batch depth
      walls_repeated = walls_repeated.repeat(batch_size, batch_depth, 1, 1, 1)
      x = torch.cat((x, walls_repeated), dim=2)

    steps = []
    losses = []

    # Initial step
    current_step = x[:, 0, :, :, :]
    steps.append(current_step.unsqueeze(1))

    # Iterate through the sequence
    for i in range(1, batch_depth):
      target = x[:, i, :, :, :]
      current_step, loss = self.forward_single(current_step, target)
      steps.append(current_step.unsqueeze(1))
      losses.append(loss)

    # Stack all steps and losses
    all_steps = torch.cat(steps, dim=1)
    all_losses = torch.stack(losses) if losses else None

    # Take mean of losses
    if all_losses is not None:
      all_losses = all_losses.mean()

    # pull out wall channel
    all_steps = all_steps[:, :, :-1, :, :]

    return all_steps, all_losses


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
  # start with all of the candidate parameters
  param_dict = {pn: p for pn, p in model.named_parameters()}
  # filter out those that do not require grad
  param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
  # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
  # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
  decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
  nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
  optim_groups = [
      {'params': decay_params, 'weight_decay': weight_decay},
      {'params': nodecay_params, 'weight_decay': 0.0}
  ]
  num_decay_params = sum(p.numel() for p in decay_params)
  num_nodecay_params = sum(p.numel() for p in nodecay_params)
  print(
      f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
  print(
      f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
  # Create AdamW optimizer and use the fused version if it is available
  fused_available = 'fused' in inspect.signature(
      torch.optim.AdamW).parameters
  use_fused = fused_available and device_type == 'cuda'
  extra_args = dict(fused=True) if use_fused else dict()
  optimizer = torch.optim.AdamW(
      optim_groups, lr=learning_rate, betas=betas, **extra_args)
  print(f"using fused AdamW: {use_fused}")

  return optimizer
