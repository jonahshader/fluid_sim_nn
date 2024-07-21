import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


class SimpleCNN(nn.Module):
  def __init__(self, channels):
    super(SimpleCNN, self).__init__()
    self.conv = nn.Conv2d(channels, 32, kernel_size=3)  # no padding
    self.act1 = nn.GELU()
    # fc1 is pixel-wise, so the input is 32
    self.conv2 = nn.Conv2d(32, 32, kernel_size=1)
    self.act2 = nn.GELU()
    self.conv3 = nn.Conv2d(32, channels, kernel_size=1)

  def forward(self, x, y=None):
    # result should be (batch, channels, height-2, width-2)
    x = self.conv(x)
    x = self.act1(x)
    x = self.conv2(x)
    x = self.act2(x)
    x = self.conv3(x)

    # print(x.shape)
    # print(y.shape)

    if y is not None:
      # compute the loss
      loss = F.mse_loss(x, y)
    else:
      loss = None

    return x, loss

  def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
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

  # def get_num_params(self):
  #   """
  #   Return the number of parameters in the model.
  #   """
  #   n_params = sum(p.numel() for p in self.parameters())
  #   return n_params

  # def estimate_mfu(self, batch_size, dt):
  #   """ estimate model flops utilization (MFU) in units of RTX 2070 SUPER float32 peak FLOPS."""
  #   # first estimate the number of flops we do per iteration.
  #   N = self.get_num_params()
