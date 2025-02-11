from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
import time

# -----------------------------------------------------------------------------
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw. Typically set to more than 1, but we set to 1 to mitigate its effects for pedgagogical reasons.
max_new_tokens = 250 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Instatiate new model.
cnfg = GPTConfig()
model = GPT(cnfg)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

## First, we warmup the model. ##
for _ in range(5):
    model(torch.randint(low=0, high=cnfg.vocab_size-1, size=(16, 200), dtype=torch.long).to(device))

# Next, run generation and timing.
with torch.no_grad():
    with ctx:
        ## We generate fake data first.
        for s in range(200, 801, 200):
            x = (torch.tensor(torch.randint(low=0, high=cnfg.vocab_size-1, size=(16, 200), dtype=torch.long).to(device)))
            cum_time = 0
            for _ in range(3): ## We take the average of 3 tries to reduce variance.
                ## Start the timer.
                torch.cuda.synchronize()
                start_time = time.time()
                for k in range(num_samples):
                    y = model.generate(x, s, temperature=temperature, top_k=top_k)
                ## End the timer.
                torch.cuda.synchronize()
                end_time = time.time()
                cum_time += (end_time-start_time)
            print(f'prefill length: {200}, generation length: {s}, time taken: {cum_time/3:0.4f}')
