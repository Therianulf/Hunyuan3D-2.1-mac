"""PBR texture generation for an existing Hunyuan3D-2.1 mesh (Windows / CUDA).

Companion to generate_shape.py: that script produces the untextured GLB on the
shape pipeline, this one paints it. Shape does NOT need re-running -- the paint
pipeline takes a mesh path directly.

Requires both native extensions to be built first (they are CUDA/MSVC only, and
are what blocked this step on the Mac):
    hy3dpaint/custom_rasterizer            -> custom_rasterizer_kernel
    hy3dpaint/DifferentiableRenderer       -> mesh_inpaint_processor
See build_extensions_windows.ps1.

Usage (must be run from the repo root; imports rely on ./hy3dpaint):
    .venv\\Scripts\\python.exe texture_mesh.py MESH.glb IMAGE.png OUT.obj \\
        [--views 6] [--resolution 512] [--render-size 2048] [--texture-size 4096]

Writes OUT.obj + OUT.jpg (albedo) + OUT_metallic/_roughness maps + OUT.mtl, and
OUT.glb if Blender is reachable for the conversion.

VRAM notes (this box is an RTX 2070, 8GB; upstream quotes 21GB for texturing):
  --views        drives the joint multiview attention batch. Biggest single
                 lever. 6 is the demo.py default, 8 the gradio default; drop to
                 4 if the denoise step OOMs.
  --resolution   per-view diffusion resolution. 512 or 768.
  --render-size  rasterisation/baking resolution. Independent of diffusion.
  --texture-size output texture resolution -- this is the "master" size and is
                 cheap (a 4096^2 float32 buffer is ~200MB), so keep it high.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")  # must be first on sys.path: `utils` and
                                   # `DifferentiableRenderer` resolve from here

from torchvision_fix import apply_fix
apply_fix()  # must run before basicsr/realesrgan import torchvision.transforms.functional_tensor

import torch
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

parser = argparse.ArgumentParser()
parser.add_argument("mesh")
parser.add_argument("image")
parser.add_argument("output", help="output .obj path; .glb is written alongside")
parser.add_argument("--views", type=int, default=6)
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--render-size", type=int, default=2048, dest="render_size")
parser.add_argument("--texture-size", type=int, default=4096, dest="texture_size")
parser.add_argument("--no-remesh", action="store_true",
                    help="skip the 40k-face decimation the pipeline does by default")
parser.add_argument("--offload-dino", action="store_true", dest="offload_dino",
                    help="keep DINOv2-giant on the CPU except while building the "
                         "conditioning; frees ~2.2GB during the denoise loop. "
                         "Required on 8GB cards -- without it the driver spills "
                         "to system RAM and the denoise crawls.")
parser.add_argument("--cpu-offload", action="store_true", dest="cpu_offload",
                    help="stream pipeline components on/off the GPU so only the "
                         "running module is resident (~2GB during denoise instead "
                         "of ~4.5GB). Slower per step, but keeps the loop inside "
                         "real VRAM. No effect on output.")
parser.add_argument("--progress", action="store_true",
                    help="show the per-step denoise bar (upstream disables it)")
args = parser.parse_args()

if args.offload_dino:
    os.environ["HY3D_OFFLOAD_DINO"] = "1"
if args.cpu_offload:
    os.environ["HY3D_CPU_OFFLOAD"] = "1"
if args.progress:
    os.environ["HY3D_PROGRESS"] = "1"

assert torch.cuda.is_available(), "CUDA not available; aborting (this pipeline has no usable CPU path)"
gpu = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | {gpu.name} | {gpu.total_memory / 1024**3:.1f}GB "
      f"| sm_{gpu.major}{gpu.minor}", flush=True)
print(f"mesh={args.mesh} image={args.image} out={args.output}", flush=True)
print(f"views={args.views} resolution={args.resolution} "
      f"render_size={args.render_size} texture_size={args.texture_size}", flush=True)

conf = Hunyuan3DPaintConfig(args.views, args.resolution)
# Paths are relative to the repo root rather than hy3dpaint/, matching demo.py.
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
conf.render_size = args.render_size
conf.texture_size = args.texture_size

t0 = time.time()
paint_pipeline = Hunyuan3DPaintPipeline(conf)
print(f"pipeline loaded in {time.time() - t0:.1f}s "
      f"| VRAM {torch.cuda.memory_allocated() / 1024**3:.2f}GB", flush=True)

t0 = time.time()
output_mesh_path = paint_pipeline(
    mesh_path=args.mesh,
    image_path=args.image,
    output_mesh_path=args.output,
    use_remesh=not args.no_remesh,
)
print(f"texturing took {time.time() - t0:.1f}s", flush=True)
print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB "
      f"of {gpu.total_memory / 1024**3:.1f}GB", flush=True)
print(f"OK: {output_mesh_path} written")
