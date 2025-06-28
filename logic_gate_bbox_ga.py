"""
Convolutional Binary Logic-Gate Network + Genetic-Algorithm RL
--------------------------------------------------------------
* Custom reward: score = 2·IoU – 1   ∈ [-1, 1]
* This v5 fixes:
  • added ScanConvLogicGateNet.copy_empty()          # FIX
  • saved kernel_sizes / channels as attributes      # FIX
  • run_pipeline_once now uses ScanConvLogicGateNet  # FIX
  • crossover builds children with same net class    # FIX
"""

import argparse, functools, random
from pathlib import Path
import numpy as np, tensorflow as tf
from PIL import Image
from ultralytics import YOLO

# ───────────────────────── 1. IMAGE → BITS ──────────────────────────
BLOCK_SIZE   = 8
IMAGE_SIZE   = 512
PERSON_ID    = 0
BIT_ALLOC_58 = [8, 7, 7, 6, 6, 6, 4, 5, 5, 4] + [0]*54

def quantize_scalar_uniform(x, n_bits, *, lo=-1.0, hi=1.0):
    levels = 2**n_bits
    step   = (hi - lo) / levels
    idx    = int(np.clip((x - lo) / step, 0, levels-1))
    return f'{idx:0{n_bits}b}'

def get_lookup_table(_, dimension):           # stub
    return np.eye(dimension)

def apply_dct(block, lut):
    return lut @ block @ lut.T

def image_to_bits(path, *, lut_path="dct_lut.json"):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img, IMAGE_SIZE, IMAGE_SIZE)
    arr = img.numpy()

    lut  = get_lookup_table(lut_path, dimension=BLOCK_SIZE)
    bits = []
    for c in range(3):
        for i in range(0, IMAGE_SIZE, BLOCK_SIZE):
            for j in range(0, IMAGE_SIZE, BLOCK_SIZE):
                coeff = apply_dct(arr[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, c], lut).flatten()
                for k, n in enumerate(BIT_ALLOC_58):
                    if n:
                        bits.append(quantize_scalar_uniform(coeff[k], n))
    return np.frombuffer("".join(bits).encode(), dtype=np.uint8) - ord("0")

# ───────────────────────── 2. TEACHER (YOLO) ─────────────────────────
_YOLO = YOLO("yolov8n.pt")

@functools.lru_cache(maxsize=None)
def teacher_bbox(path):
    img  = Image.open(path).convert("RGB")
    res  = _YOLO(img, verbose=False)[0]
    boxes = res.boxes.data.cpu().numpy()
    persons = [b for b in boxes if int(b[5]) == PERSON_ID]
    if not persons: raise RuntimeError("No person detected.")
    x1,y1,x2,y2,*_ = max(persons, key=lambda b: b[4])
    H,W = img.size[1], img.size[0]
    return np.array([(x1+x2)/(2*W), (y1+y2)/(2*H), (x2-x1)/W, (y2-y1)/H], np.float32)

# ───────────────────── 3. 1-D SCANNING CONV NET ─────────────────────
def binary_conv1d_same(x, W):
    C_out, C_in, k = W.shape
    _, L           = x.shape
    pad            = k//2
    xp = np.zeros((C_in, L+2*pad), dtype=x.dtype) if pad else x
    if pad: xp[:, pad:-pad] = x
    out = np.empty((C_out, L), dtype=np.uint8)
    for pos in range(L):
        win  = xp[:, pos:pos+k]                 # (C_in,k)
        match = ~(win ^ W) & 1
        out[:, pos] = match.sum((1,2)) & 1      # parity
    return out

class ScanConvLogicGateNet:
    """Keeps sequence length; final dense binary layer outputs decimals."""
    def __init__(self, seq_len, kernel_sizes, channels, out_dim=4):
        assert len(kernel_sizes) == len(channels)
        self.seq_len       = seq_len
        self.kernel_sizes  = kernel_sizes[:]     # FIX store for copy_empty
        self.channels      = channels[:]         # FIX
        self.out_dim       = out_dim

        C_in = 1
        self.conv_layers = []
        for k, C_out in zip(kernel_sizes, channels):
            self.conv_layers.append(np.random.randint(0,2,(C_out, C_in, k), dtype=np.uint8))
            C_in = C_out
        self.fc = np.random.randint(0,2,(out_dim, C_in*seq_len), dtype=np.uint8)

    def copy_empty(self):                        # FIX helper for GA
        return ScanConvLogicGateNet(self.seq_len,
                                    self.kernel_sizes,
                                    self.channels,
                                    self.out_dim)

    def iter_weights(self):
        yield from self.conv_layers
        yield self.fc

    def forward(self, bits):
        x = bits.reshape(1, self.seq_len)
        for W in self.conv_layers:
            x = binary_conv1d_same(x, W)
        flat   = x.reshape(-1)
        counts = self.fc @ flat
        return counts.astype(np.float32) / self.fc.shape[1]

# ─────────────────────── 4.  custom reward ──────────────────────────
def _xyxy(b):
    cx,cy,w,h = b; return cx-w/2, cy-h/2, cx+w/2, cy+h/2
def overlap_minus_nonoverlap(a,b):
    ax1,ay1,ax2,ay2 = _xyxy(a)
    bx1,by1,bx2,by2 = _xyxy(b)
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
    union   = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-9
    return (2*inter - union) / union

# ───────────────────── 5.  GA INDIVIDUAL ────────────────────────────
class Individual:
    def __init__(self, net=None):
        self.net = net or ScanConvLogicGateNet(
            seq_len=712_704,
            kernel_sizes=[9,9,9],
            channels=[8,16,32],
            out_dim=4
        )
        self.fitness = -np.inf
        self.pred    = None

    def evaluate(self, bits, target):
        self.pred    = self.net.forward(bits)
        self.fitness = overlap_minus_nonoverlap(self.pred, target)

    def mutate(self, rate=0.1):
        for W in self.net.iter_weights():
            m = np.random.rand(*W.shape) < rate
            W[m] ^= 1

    def crossover(self, other):
        child_net = self.net.copy_empty()        # FIX same arch
        for dst, Wa, Wb in zip(child_net.iter_weights(),
                               self.net.iter_weights(),
                               other.net.iter_weights()):
            choose = np.random.rand(*Wa.shape) < 0.5
            np.copyto(dst, np.where(choose, Wa, Wb))
        return Individual(child_net)

# ───────────────────── 6A. single-pass debug ────────────────────────
def run_pipeline_once(img_path):
    print("🖼 ", Path(img_path).name)
    bits = image_to_bits(img_path)
    print("🔢 bits:", bits.size)

    tgt  = teacher_bbox(img_path)
    print("🎯 teacher:", np.round(tgt,4))

    net  = ScanConvLogicGateNet(seq_len=bits.size,
                                kernel_sizes=[9,9,9],
                                channels=[8,16,32])
    print("⚙️  layer shapes:", [w.shape for w in net.iter_weights()])

    pred = net.forward(bits)
    print("🤖 pred   :", np.round(pred,4))
    print("🏆 score  :", overlap_minus_nonoverlap(pred, tgt))

# ───────────────────── 6B. evolutionary run ─────────────────────────
def evolve(path, *, pop_size=50, generations=100, elite=10):
    bits   = image_to_bits(path)
    target = teacher_bbox(path)
    pop    = [Individual() for _ in range(pop_size)]

    for g in range(generations):
        for ind in pop: ind.evaluate(bits, target)
        pop.sort(key=lambda x: x.fitness, reverse=True)
        best = pop[0]
        print(f"Gen {g:03d} | best={best.fitness:+.4f} | pred={np.round(best.pred,3)}")

        survivors = pop[:elite]
        next_gen  = survivors.copy()
        while len(next_gen) < pop_size:
            c = random.choice(survivors).crossover(random.choice(survivors))
            c.mutate(); next_gen.append(c)
        pop = next_gen

# ───────────────────── 7.  CLI ──────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--once", action="store_true")
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--gen", type=int, default=100)
    a = p.parse_args()

    if a.once: run_pipeline_once(a.image)
    else:      evolve(a.image, pop_size=a.pop, generations=a.gen)
