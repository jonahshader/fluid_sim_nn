import os
import json
# import torch
import numpy as np

with open("../data/metadata.json", encoding="utf-8") as f:
  metadata = json.load(f)

print(metadata)

batch_0_dir = "../data/batch_0.bin"
batch_0 = np.memmap(batch_0_dir, dtype=np.float32, mode='r')
print(batch_0)
