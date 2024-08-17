import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


class SimpleCNN_BPTT(nn.Module):
  def __init__(self, channels, top_kernel_size, skip_con=False):
    super(SimpleCNN_BPTT, self).__init__()
    self.top_kernel_size = top_kernel_size
    padding = top_kernel_size // 2
    self.conv = nn.Conv2d(channels, 256, kernel_size=top_kernel_size,
                          padding_mode='circular', padding=padding)

    self.act1 = nn.GELU()
    # fc1 is pixel-wise, so the input is 32
    self.conv2 = nn.Conv2d(256, 256, kernel_size=1)
    self.act2 = nn.Tanh()
    self.conv3 = nn.Conv2d(256, channels, kernel_size=1)
    self.skip_con = skip_con

  def forward_single(self, x, y=None):
    x_orig = x
    # result should be (batch, channels, height-2, width-2)
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

  def forward(self, x):
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
      current_step, loss = self.forward_single(current_step, target)
      steps.append(current_step.unsqueeze(1))
      losses.append(loss)

    # Stack all steps and losses
    all_steps = torch.cat(steps, dim=1)
    all_losses = torch.stack(losses) if losses else None

    # Take mean of losses
    if all_losses is not None:
      all_losses = all_losses.mean()

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
