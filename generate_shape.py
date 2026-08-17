"""Shape-only MPS generation for Hunyuan3D-2.1 (mac fork) — parameterized version of smoke_test.py.

Usage (must be run from the repo root; imports rely on ./hy3dshape):
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python generate_shape.py IMAGE OUT.glb \
        [--steps 30] [--octree 256] [--seed 2026] [--guidance 5.0] [--chunks 8000]

Texture generation is intentionally untouched (CUDA-only on this fork).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, './hy3dshape')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from torchvision_fix import apply_fix
apply_fix()  # must run before any torchvision.transforms.functional_tensor import

import torch
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

parser = argparse.ArgumentParser()
parser.add_argument('image')
parser.add_argument('output')
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--octree', type=int, default=256)
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--guidance', type=float, default=5.0)
parser.add_argument('--chunks', type=int, default=8000)
args = parser.parse_args()

assert torch.backends.mps.is_available(), 'MPS not available; aborting'
print('torch', torch.__version__, '| device: mps', flush=True)
print(f'in={args.image} out={args.output} steps={args.steps} octree={args.octree} seed={args.seed}', flush=True)

image = Image.open(args.image)
if image.mode != 'RGBA':
    t0 = time.time()
    image = BackgroundRemover()(image.convert('RGB'))
    print(f'background removed in {time.time() - t0:.1f}s', flush=True)


def generate(dtype):
    t0 = time.time()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2.1',
        subfolder='hunyuan3d-dit-v2-1',
        use_safetensors=False,   # fork default: loads model.fp16.ckpt
        device='mps',
        dtype=dtype,
    )
    print(f'pipeline loaded ({dtype}) in {time.time() - t0:.1f}s', flush=True)

    torch.manual_seed(args.seed)
    t0 = time.time()
    mesh = pipeline(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        octree_resolution=args.octree,
        num_chunks=args.chunks,
        mc_algo='mc',            # 'dmc' is CUDA-only (diso)
        output_type='trimesh',
    )[0]
    print(f'generation took {time.time() - t0:.1f}s', flush=True)
    return mesh


try:
    mesh = generate(torch.float16)
except Exception as e:
    print(f'fp16 attempt failed ({type(e).__name__}: {e}); retrying in float32', flush=True)
    mesh = generate(torch.float32)

mesh.export(args.output)
print('faces:', len(mesh.faces), 'vertices:', len(mesh.vertices))
print(f'OK: {args.output} written')
