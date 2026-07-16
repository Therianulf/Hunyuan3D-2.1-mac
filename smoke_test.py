"""Shape-only MPS smoke test for Hunyuan3D-2.1 (mac fork).

Generates smoke_test.glb from assets/demo.png on Apple Silicon.
Texture pipeline is intentionally untouched (CUDA-only on this fork).
"""
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

assert torch.backends.mps.is_available(), 'MPS not available; aborting'
print('torch', torch.__version__, '| device: mps')


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

    image = Image.open('assets/demo.png')
    if image.mode != 'RGBA':
        image = BackgroundRemover()(image.convert('RGB'))

    torch.manual_seed(2026)
    t0 = time.time()
    mesh = pipeline(
        image=image,
        num_inference_steps=30,
        guidance_scale=5.0,
        octree_resolution=256,   # start low for speed; 384 for more detail
        num_chunks=8000,
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

mesh.export('smoke_test.glb')
print('faces:', len(mesh.faces), 'vertices:', len(mesh.vertices))
print('OK: smoke_test.glb written')
