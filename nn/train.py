"""
Only runs on a single GPU.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import torch

from model import SimpleCNN_BPTT, configure_optimizers
from data import load_recordings, split_recordings


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
eval_interval = 200
log_interval = 10
eval_iters = 20
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume'
# wandb logging
wandb_log = True  # disabled by default
wandb_project = 'fluid_sim_nn'
wandb_run_name = 'run' + str(time.time())
model_type = SimpleCNN_BPTT
out_dir = 'out_' + model_type.__name__
# data
dataset = '4_walls'
batch_size = 256
batch_depth = 4
# model
top_kernel_size = 3
skip_con = True

# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 6000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 20  # how many steps to warm up for
lr_decay_iters = 6000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# dtype = 'bfloat16' if torch.cuda.is_available(
# ) and torch.cuda.is_bf16_supported() else 'float16'
dtype = 'float32'
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith(
    '_') and isinstance(v, (int, float, bool, str))]
# overrides from command line or config file
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# we are running on a single gpu, and one process
master_process = True
seed_offset = 0

# output at ../models/out_dir
full_out_dir = os.path.join(os.path.dirname(__file__), '..', 'models', out_dir)
if master_process:
  os.makedirs(full_out_dir, exist_ok=True)


torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
# for later use in torch.autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype)

# poor man's data loader
# current dir is fluid_sim_nn/nn, data is in fluid_sim_nn/data/{dataset}
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', dataset)
metadata, data, normalize_transform, wall_data = load_recordings(data_dir)

train_indices, test_indices, data = split_recordings(
    data, batch_depth=batch_depth)

# normalize the data
data = normalize_transform(data)

# send to device
data = data.to(device=device, dtype=ptdtype)
wall_data = wall_data.to(device=device, dtype=ptdtype)
train_indices = train_indices.to(device=device, dtype=torch.int32)
test_indices = test_indices.to(device=device, dtype=torch.int32)


def get_batch(split):
  indices = train_indices if split == 'train' else test_indices
  indices = indices.to(torch.int32)  # Ensure int32
  idx = torch.randperm(len(indices), dtype=torch.int32)[:batch_size]
  start_indices = indices[idx]

  offsets = torch.arange(
      batch_depth, device=start_indices.device, dtype=torch.int32)

  batch_indices = start_indices.unsqueeze(1) + offsets

  X_batch = data[batch_indices]

  # print(f"batch_indices: {batch_indices.shape}")
  # print(f"X_batch: {X_batch.shape}")
  # print(f"X: {X.shape}")

  return X_batch


# for i in range(10):
#   X = get_batch('train')
#   print(f"X: {X.shape}")


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
  with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
  meta_vocab_size = meta['vocab_size']
  print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
# start with model_args from command line
model_args = dict(channels=len(
    metadata['attributes']), top_kernel_size=top_kernel_size, skip_con=skip_con)
if init_from == 'scratch':
  # init a new model from scratch
  print("Initializing a new model from scratch")
  # determine the vocab size we'll use for from-scratch training
  model = model_type(**model_args)
elif init_from == 'resume':
  print(f"Resuming training from {full_out_dir}")
  # resume training from a checkpoint.
  ckpt_path = os.path.join(full_out_dir, 'ckpt.pt')
  checkpoint = torch.load(ckpt_path, map_location=device)
  checkpoint_model_args = checkpoint['model_args']
  # force these config attributes to be equal otherwise we can't even resume training
  # the rest of the attributes (e.g. dropout) can stay as desired from command line
  for k in ['top_kernel_size', 'skip_con']:
    model_args[k] = checkpoint_model_args[k]
  # create the model
  model = model_type(**model_args)
  state_dict = checkpoint['model']
  # fix the keys of the state dictionary :(
  # honestly no idea how checkpoints sometimes get this prefix, have to debug more
  unwanted_prefix = '_orig_mod.'
  for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
      state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  model.load_state_dict(state_dict)
  iter_num = checkpoint['iter_num']
  best_val_loss = checkpoint['best_val_loss']
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = configure_optimizers(
    model, weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
  optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model
if compile:
  print("compiling the model... (takes a ~minute)")
  unoptimized_model = model
  model = torch.compile(model)  # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X = get_batch(split)
      with ctx:
        logits, loss = model(X, walls=wall_data)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

# learning rate decay scheduler (cosine with warmup)


def get_lr(it):
  # 1) linear warmup for warmup_iters steps
  if it < warmup_iters:
    return learning_rate * it / warmup_iters
  # 2) if it > lr_decay_iters, return min learning rate
  if it > lr_decay_iters:
    return min_lr
  # 3) in between, use cosine decay down to min learning rate
  decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
  return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
  import wandb
  wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process

while True:
  # determine and set the learning rate for this iteration
  lr = get_lr(iter_num) if decay_lr else learning_rate
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  # evaluate the loss on train/val sets and write checkpoints
  if iter_num % eval_interval == 0 and master_process:
    losses = estimate_loss()
    print(
        f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if wandb_log:
      wandb.log({
          "iter": iter_num,
          "train/loss": losses['train'],
          "val/loss": losses['val'],
          "lr": lr,
      })
    if losses['val'] < best_val_loss or always_save_checkpoint:
      best_val_loss = losses['val']
      if iter_num > 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        print(f"saving checkpoint to {full_out_dir}")
        torch.save(checkpoint, os.path.join(full_out_dir, 'ckpt.pt'))
  if iter_num == 0 and eval_only:
    break

  # forward backward update, using the GradScaler if data type is float16
  with ctx:
    logits, loss = model(X, walls=wall_data)

  # immediately async prefetch next batch while model is doing the forward pass on the GPU
  X = get_batch('train')
  # backward pass, with gradient scaling if training in fp16
  scaler.scale(loss).backward()
  # clip the gradient
  if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
  # step the optimizer and scaler if training in fp16
  scaler.step(optimizer)
  scaler.update()
  # flush the gradients as soon as we can, no need for this memory anymore
  optimizer.zero_grad(set_to_none=True)

  # timing and logging
  t1 = time.time()
  dt = t1 - t0
  t0 = t1
  if iter_num % log_interval == 0 and master_process:
    # get loss as float. note: this is a CPU-GPU sync point
    # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
    lossf = loss.item()
    print(
        f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
  iter_num += 1
  local_iter_num += 1

  # termination conditions
  if iter_num > max_iters:
    break
