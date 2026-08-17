# Hunyuan3D pipeline: concept image → textured game character

Reference for agents working on this repo. Covers the full path from a concept
image to a textured, game-ready character, across machines.

Companion docs: `HUNYUAN3D_MARVIN.md` (original Mac/MPS setup, licensing),
`README_macOS.md` (upstream fork docs).

Last verified 2026-08-17.

---

## 1. Constraints

- **Do not run `git commit` or `git push`.** Ben handles all git history.
- Stay inside this working directory. Do not touch other repos on the machine.
- **Texturing requires CUDA.** There is no usable CPU or MPS path — the CPU
  rasterizer fallback (`custom_rasterizer/render_fallback.py`) is a pure-Python
  per-triangle, per-pixel loop and will not finish on a real mesh.
- If a step fails twice with different fixes attempted, stop and report.
- License: outputs are yours, but the Territory excludes the EU, UK, and South
  Korea. Fine for prototyping; not compliant for worldwide commercial release.
  See `HUNYUAN3D_MARVIN.md` §0 and its License findings section.

---

## 2. Pipeline

```
 [1] concept image, letterboxed to 1:1
 [2] generate_shape.py --octree 384          →  character.glb   (~140k+ faces)
 [3] decimate once to 40k faces                                 (shared boundary)
 [4] split by height threshold               →  head.glb + body.glb
 [5] crop the letterboxed image              →  head.png + body.png
 [6] texture_mesh.py head.glb head.png --no-remesh --resolution 512 --texture-size 4096
 [7] texture_mesh.py body.glb body.png --no-remesh --resolution 512 --texture-size 4096
 [8] merge                                   →  one GLB, two meshes, two material sets
 [9] blender_ps1_pass.py                     →  ~1200 tris, flat shaded, FBX
```

**Built:** 1 (manual), 2, 6, 7, 9, plus `obj_to_glb.py`.
**Not built:** 3, 4, 5, 8 — the decimate / split / crop / merge tooling.

Steps 3–8 exist to solve face quality. A single whole-body texture pass spends
~90% of its resolution on the body; see §6.

---

## 3. Source image

The paint pipeline treats the image as an appearance reference and downsamples it
aggressively:

```
input PNG → resize((512,512))          textureGenPipeline.py:153   NON-UNIFORM
          → resize((resolution,)*2)     multiview_utils.py
          → DINOv2: shortest_edge 256, center-crop 224×224
```

Rules that follow:

- **Letterbox to 1:1 before use.** `resize((512,512))` is a squash, not a fit. A
  1152×1728 source is compressed 33% vertically, so the model conditions on a
  shorter, wider character. Pad to square with white.
- **Do not supply more than 512×512 of useful detail.** Everything above that is
  discarded. Higher-resolution sources buy nothing.
- **Keep the subject away from the frame edge.** DINOv2 center-crops 256→224,
  removing ~6% per edge, which can clip the top of the head.
- Single centred subject, plain background, RGBA after background removal.

---

## 4. Shape generation

`generate_shape.py IMAGE OUT.glb [--octree N] [--steps N] [--seed N]`

Runs on MPS (Marvin) or CUDA. `--octree` sets the marching-cubes grid:

| `--octree` | Result |
|---|---|
| 256 | Faster. ~141k faces on a full-body character. |
| 384 | **Preferred.** Noticeably better surface detail. |

`--octree 384` costs nothing downstream: texturing decimates to 40k faces either
way (§5), so the denser mesh only means the surviving 40k faces sit on a more
accurate surface.

Reference timing, Marvin (M4 Max, MPS), octree 256: ~5.4 min total.

`mc_algo='mc'` is mandatory — `dmc` needs the CUDA-only `diso` package.

---

## 5. Texture generation

`texture_mesh.py MESH IMAGE OUT.obj [flags]`

Takes an existing mesh; **never re-runs shape generation**. Writes OUT.obj plus
albedo / metallic / roughness maps and an MTL.

### What it does to geometry first

1. **Decimate** — `utils/simplify_mesh_utils.py:23`, `target_count=40000`.
   Disable with `--no-remesh`.
2. **UV unwrap** — `utils/uvwrap_utils.py` via `xatlas`. **Any existing UVs are
   discarded and replaced.**
3. **Rasterize** the UV atlas at `--texture-size` (4096²) via the CUDA rasterizer.

### Then

4. **View selection** — best N cameras by surface coverage, from ~30 candidates.
5. **Multiview diffusion** — N albedo + N metallic-roughness views generated
   jointly, 15 UniPC steps. Dominant cost in both time and VRAM.
6. **ESRGAN 4× upscale** of every generated view.
7. **Back-project and bake** into the UV atlas.
8. **Inpaint** UV regions no camera saw (`mesh_inpaint_processor`).
9. **Save** OBJ + maps, then attempt GLB conversion.

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--views` | 6 | Joint-attention batch size. Primary VRAM lever. |
| `--resolution` | 512 | **Keep at 512.** See below. |
| `--render-size` | 2048 | Rasterisation/baking resolution. Affects bake stage only. |
| `--texture-size` | 4096 | Output texture. Cheap — a 4096² f32 buffer is ~200MB. |
| `--no-remesh` | off | Skip the 40k decimation. Required for pre-split meshes. |
| `--offload-dino` | off | Moves DINOv2-giant to CPU between calls. Frees ~2.1GB. Required on 8GB cards. |
| `--progress` | off | Show the per-step denoise bar. |
| `--cpu-offload` | off | **Broken** — see §9. |

**`--resolution` must be 512.** `hy3dpaint/cfgs/hunyuan-paint-pbr.yaml:6` sets
`view_size: 512`, the resolution the paint model was trained at. Lower values are
off-distribution and degrade output disproportionately.

**Do not confuse `--resolution` with `--octree`.** They are separate knobs on
separate models: `--octree` controls mesh density in shape generation,
`--resolution` controls per-view detail in texture diffusion.

---

## 6. Head/body split

### Why

The renderer auto-centers and scales whatever mesh it is given to fill the frame,
and `xatlas` allocates UV space across whatever mesh it is given.

- Whole body: the head fills ~10% of a 512² view and gets ~10% of one UV atlas.
- Head alone: the head fills ~90% of the same 512² view and gets ~90% of its own
  atlas.

Roughly **10× the effective face detail on identical hardware**. This is why more
VRAM does not fix faces — raising 384→768 is 4× face pixels; the split is ~10×.
The two compose.

Separate head and body materials is standard game-character practice.

### Ordering is load-bearing

```
decimate the FULL mesh once  →  split  →  texture each half with --no-remesh
```

Splitting first and letting the pipeline decimate each half independently
simplifies the two halves differently along the cut, producing a visible crack at
the neck. One decimation up front gives both halves identical boundary vertices.

### Assembly

Export as **two meshes with two material sets inside one GLB**. Do **not** weld
them into a single mesh — that collapses them back to one UV atlas and removes
the entire benefit.

### Known risks

- The two textures are generated independently and will not tone-match perfectly
  at the seam. A hooded character hides most of that boundary.
- A head-only mesh is somewhat off-distribution for the paint model. Unproven.

---

## 7. Hardware and VRAM

Upstream requirement (`README.md:76`): 10GB shape, **21GB texture**, 29GB both.

**Target platform: AWS `g5.xlarge`** — A10G, 24GB, ~$1/hr. Clears 21GB outright:
run 9 views at 768 with no offload flags and no compromises. Avoid `g4dn` (T4,
16GB, Turing — same generation as the RTX 2070, not meaningfully faster).

Linux is materially easier than Windows here; every fix in §10 is Windows-only.

### Measured on RTX 2070 (8GB, sm_75)

| Configuration | Result |
|---|---|
| Pipeline resident, stock | 6.62GB — leaves ~1.4GB |
| Pipeline resident, `--offload-dino` | 4.51GB |
| 6 views / 512 / 2048 / 4096 | Never completed; 20+ min pinned at 7,981 MiB |
| 4 views / 384 / 1024 / 2048 + `--offload-dino` | **85.9s**, peak 8.16GB, completed |

Weights, fp16: UNet ~2.0GB, DINOv2-giant ~2.2GB, CLIP text encoder ~0.7GB, image
encoder ~0.6GB, VAE ~0.2GB. Mesh complexity is negligible against these — the
cost is the paint model, not the geometry.

**Failure mode on Windows:** the NVIDIA driver oversubscribes VRAM and spills to
system RAM instead of raising `OutOfMemoryError`. Exceeding VRAM presents as a
~20× slowdown, not a crash. Do not wait on a run that has gone quiet with memory
pinned at the card limit — kill it and reduce `--views` or enable
`--offload-dino`.

Useful env vars: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`HY3D_ESRGAN_TILE` (default 512), `PYTHONUTF8=1`.

---

## 8. Diagnosing a slow or stuck run

The denoise loop has progress bars disabled upstream; use `--progress` to enable.
To locate a running stage without restarting:

```
.venv/Scripts/py-spy.exe dump --pid <pid>
```

The Python stack names the exact stage (`multiview_utils.forward_one`,
`MeshRender.extract_textiles`, `image_super_utils`, etc.).

---

## 9. Known issues

- **`--cpu-offload` is broken.** `enable_model_cpu_offload()` fails because the
  custom pipeline calls `vae.encode()` inside `encode_images` outside the order
  accelerate's hooks expect, giving a cpu/cuda device mismatch. Recovering it
  needs per-module hand-wiring. Irrelevant on a 24GB card.
- **Blender is not installed on the Windows box.** `download.blender.org` returns
  403 from this network via both winget and direct fetch. `convert_obj_to_glb`
  therefore fails and the pipeline writes no GLB. Use `obj_to_glb.py` instead —
  it produces a *better* GLB than the Blender path, packing metallic and
  roughness into one glTF-correct texture (roughness in G, metallic in B) rather
  than two separate greyscale JPEGs. The PS1 pass still needs real Blender.
- **`demo_macos.py` is broken** — wrong `from_pretrained` signature. Use
  `generate_shape.py`.
- **`run_gradio_macos.sh`** hardcodes a personal miniconda path.
- Steps 3–5 and 8 of §2 are unbuilt.
- The head/body split is designed but unproven.

---

## 10. Windows platform notes

Irrelevant on Linux. Three classes of LP64 bug were fixed in `custom_rasterizer`,
all because `long` is 64-bit on Linux and 32-bit on Windows:

1. **`grid_neighbor.cpp`** — `size_t`→`int64_t` narrowing inside
   `torch::zeros({...})` braced-init-lists. MSVC treats this as a hard error
   (C2398); GCC only warns. ~13 sites.
2. **`data_ptr<long>()`** on int64 tensors in `grid_neighbor.cpp`,
   `rasterizer.cpp`, `rasterizer_gpu.cu` → `data_ptr<int64_t>()`.
3. **`rasterizer_gpu.cu:111` and `rasterizer.cpp:103`** — z-buffer seeded with
   `(long)maxint`, truncating the 64-bit sentinel `0x3FFFFFFF7FFFFFFF` to
   `0x7FFFFFFF`. Every pixel then began at depth 0 with face index 2147483647,
   used directly as an array index → *"CUDA error: an illegal memory access was
   encountered"*.

Build requirements:

- Pin the MSVC toolset to **14.38** (`vcvarsall.bat x64 -vcvars_ver=14.38`). CUDA
  12.4's nvcc rejects MSVC 19.40+, and the Build Tools default is 14.44.
- Set `DISTUTILS_USE_SDK=1` and `TORCH_CUDA_ARCH_LIST=7.5`.
- **`import torch` must precede importing any native extension.** Python 3.8+
  ignores `PATH` for extension DLLs; importing torch registers `torch/lib`
  (carrying `cudart64_12.dll`) as a DLL search directory. Otherwise:
  `ImportError: DLL load failed while importing custom_rasterizer_kernel`.

Cross-platform fixes also applied:

- `image_super_utils.py` — RealESRGAN ran with `tile=0`, upscaling each view in
  one allocation. Now tiled via `HY3D_ESRGAN_TILE` (default 512).
- `mesh_utils.py` — Blender discovery was macOS-only; now honours `BLENDER_PATH`,
  `PATH`, and Windows install locations. Subprocess timeout 30s → 600s.

---

## 11. Files

| File | Purpose |
|---|---|
| `generate_shape.py` | image → mesh (MPS; needs a CUDA variant) |
| `texture_mesh.py` | mesh + image → textured OBJ + PBR maps |
| `obj_to_glb.py` | textured OBJ → PBR GLB, no Blender needed |
| `blender_ps1_pass.py` | GLB → decimated flat-shaded FBX |
| `blender_render_check.py` | headless 3-view render (needs Blender) |
| `build_extensions_windows.ps1` | builds both native extensions |
| `hy3dpaint/DifferentiableRenderer/setup_windows.py` | MSVC build for the inpaint extension |
| `requirements-windows.txt` | CUDA deps, Windows / Python 3.11 |
| `requirements-macos.txt` | MPS deps, shape only |

Environment: `.venv/` (Python 3.11, torch 2.5.1+cu124). Weights cache to
`~/.cache/huggingface`; shape weights to `~/.cache/hy3dgen`.
