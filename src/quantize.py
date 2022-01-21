import os
import torch

from train import Model

model = Model.load_from_checkpoint("checkpoints/balmy-donkey-71.ckpt").model

# import pdb; pdb.set_trace()
# Raspberry Pi 4 workaround for the following error:
# RuntimeError: Didn't find engine for operation quantized::conv_prepack NoQEngine
import platform
if platform.system() == 'Linux' and not torch.cuda.is_available():
    torch.backends.quantized.engine = 'qnnpack'


def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
    os.remove("tmp.pt")


# backend = "fbgemm"
# model.qconfig = torch.quantization.get_default_qconfig(backend)
# torch.backends.quantized.engine = backend
# model_static_quantized = torch.quantization.prepare(model, inplace=False)
# model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

# print_model_size(model)
# print_model_size(model_static_quantized)

# print(model_static_quantized)

# x = torch.randn(32, 1, 64, 44)
# import pdb; pdb.set_trace()


# Quantization aware training
model.eval()
model_int8 = torch.quantization.convert(model)
print_model_size(model)
print_model_size(model_int8)
x = torch.randn(1, 1, 64, 44)

import time

all_times = []
for _ in range(500):
    start = time.time()
    model(x)
    elapsed = time.time() - start
    all_times.append(elapsed)
average = sum(all_times) / len(all_times)
print(f"Regular: {average * 1000:.3f}ms")


all_times = []
for _ in range(500):
    start = time.time()
    model_int8(x)
    elapsed = time.time() - start
    all_times.append(elapsed)
average = sum(all_times) / len(all_times)
print(f"Quantized: {average * 1000:.3f}ms")
