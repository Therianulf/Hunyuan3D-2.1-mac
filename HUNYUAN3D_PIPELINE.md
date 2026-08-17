# Hunyuan3D end-to-end pipeline: concept image → textured game character

Companion to `HUNYUAN3D_MARVIN.md` (which covers the original Mac/MPS setup).
This document describes the **full pipeline**, across machines, including the
head/body split that exists to solve the face-quality problem.

Status 2026-08-17. Numbers in this doc are measured on real runs, not estimates,
unless explicitly labelled as an estimate.

---

## 0. Your restatement, corrected

You wrote:

> making a model from an image: generate source image, should be 1:1, model the
> image. i made a new model at 384 on marvin and it looks way better. so generate
> the model on marvin from our source image. then we take it to our
> meshing/texturing machine ... it then makes a mesh from the model in whole,
> decimating it and making a shared seam, we then take and submesh the mesh into
> the head and the body. we then divide our source picture by the head, and the
> body, and then draw the new textures with this source image

This is substantially right. Five corrections:

| # | Your statement | Correction |
|---|---|---|
| 1 | "a new model at 384" | **There are two different 384s and they are unrelated.** See §1 below — this is the single most important thing to keep straight. |
| 2 | "source image should be 1:1" | **Correct, and for a concrete reason** — the pipeline does a non-uniform `resize((512,512))`, so a 2:3 source gets squashed 33%. But also: anything above 512×512 is discarded. 1:1 matters; extra resolution does not. |
| 3 | "it then makes a mesh from the model" | The model **is** the mesh. Shape generation outputs a mesh (GLB); the texture stage never creates geometry. What the texture stage does to geometry is *decimate* and *UV-unwrap* it. |
| 4 | "decimating it and making a shared seam" | "Making a shared seam" is not a step. The shared seam is a **consequence** of decimating *before* splitting. Do it in the other order and you get a crack at the neck. |
| 5 | "then we attach the head and body mesh" | Do **not** weld them into one mesh — that would collapse them back to one UV atlas and destroy the entire benefit. They ship as **two meshes with two materials inside one GLB**. |

One thing missing from your version entirely: **UV unwrapping**. It is the step
that makes the head/body split work at all. See §5.

---

## 1. The two different "384"s

This trips everyone up. They are separate knobs on separate models.

| | `--octree` (shape) | `--resolution` (texture) |
|---|---|---|
| Script | `generate_shape.py` | `texture_mesh.py` |
| What it controls | Marching-cubes grid for the isosurface | Per-view resolution of the multiview diffusion |
| Effect of raising it | More faces, finer surface detail | More texture detail per view |
| Your "384 looks way better" | **This one.** octree 256 → 384 | not this one |
| Sane values | 256 (fast), 384 (better) | 512 only — see below |

**`--octree 384` is a real quality win and costs you nothing downstream**, because
texturing decimates to 40k faces regardless (§4). A denser source mesh means the
40k faces you keep sit on a more accurate surface. Use 384.

**`--resolution` must be 512.** `hy3dpaint/cfgs/hunyuan-paint-pbr.yaml:6` sets
`view_size: 512` — that is the resolution the paint model was *trained* at.
Running it at 384 is off-distribution and degrades output disproportionately,
beyond the raw pixel loss. The only reason we ran 384 was to fit an 8GB card, and
it was a validation compromise, not a quality setting.

---

## 2. Source image preparation

The paint pipeline consumes your concept image as an *appearance reference*, and
it is brutal about it:

```
your PNG  →  resize((512, 512))      textureGenPipeline.py:153   ← NON-UNIFORM
          →  resize((resolution,)*2)  multiview_utils.py
          →  DINOv2: shortest_edge 256, then center-crop 224×224
```

Consequences:

- **Anything above 512×512 is thrown away.** `sneaker_tpose.png` is 1152×1728 —
  roughly 9× more pixels than are ever used. A higher-resolution source does
  nothing.
- **Non-square input is distorted.** 1152×1728 forced to 512×512 scales X by
  0.444 and Y by 0.296 — the character is **compressed vertically by 33%**. The
  model's appearance reference is a shorter, wider character than yours.
- **DINOv2 center-crops.** 256 → 224 shaves ~6% off each edge, which on a
  full-body figure can clip the top of the head.

**So: letterbox to square before feeding it.** Pad 1152×1728 out to 1728×1728
with white. This is a pure win and costs nothing.

---

## 3. Machine split

| Stage | Where | Why |
|---|---|---|
| Concept image | wherever | — |
| Shape generation | Marvin (M4 Max, MPS) **or** AWS | Works on MPS. ~5.4 min at octree 256; longer at 384. |
| Texturing | **AWS (CUDA, ≥24GB)** | Impossible on MPS — needs the CUDA rasterizer. |
| PS1 pass | any machine with Blender | Blender-only |

Shape generation is the only stage Marvin can do, and it is the stage that
matters least for the current problem. Once the AWS box exists it should do
both — one machine, one environment, no file shuttling.

**Recommended AWS instance: `g5.xlarge`** — A10G, 24GB, ~$1/hr on-demand. 24GB
clears the documented 21GB texture requirement outright, so **no offload hacks,
no reduced view counts, no compromises**. Avoid `g4dn` (T4): 16GB and it is
Turing, same generation as the RTX 2070, so it is not meaningfully faster.

**Linux is materially easier than Windows here.** Every bug fixed on 2026-08-16
(§8) exists only because `long` is 32-bit on Windows and 64-bit on Linux. On
Linux the native extensions build clean from the stock `setup.py`.

---

## 4. What the texture pipeline does to your mesh

Before any texturing happens, `Hunyuan3DPaintPipeline.__call__` rewrites the
geometry. This is the part your restatement compressed into "makes a mesh":

1. **Decimate** — `utils/simplify_mesh_utils.py:23`, `target_count=40000`.
   Measured: `sneaker_tpose_256.glb` 141,212 faces → 40,000 faces / 26,532 verts.
   Controlled by `use_remesh` (our `--no-remesh` flag turns it off).
2. **UV unwrap** — `utils/uvwrap_utils.py`, via `xatlas`. Generates a fresh UV
   atlas. **Any UVs on the input mesh are discarded and replaced.**
3. **Rasterize** the UV atlas at `texture_size` (4096²) using the CUDA rasterizer.

Then the actual texturing:

4. **View selection** — picks the best N cameras by surface coverage from ~30
   candidates.
5. **Multiview diffusion** — generates N albedo + N metallic-roughness views
   jointly, 15 UniPC steps. This is the expensive stage.
6. **ESRGAN 4× upscale** of every generated view.
7. **Back-project and bake** the views into the UV atlas.
8. **Inpaint** UV regions no camera saw (`mesh_inpaint_processor`).
9. **Save** OBJ + albedo/metallic/roughness maps, then GLB.

---

## 5. Why the head/body split works (the part your version was missing)

Two facts combine:

- **The renderer auto-centers and scales whatever mesh you give it to fill the
  frame.**
- **`xatlas` allocates UV space across whatever mesh you give it.**

Hand it the whole body: the head fills ~10% of a 512² view, and gets ~10% of one
UV atlas. Hand it just the head: the head fills ~90% of that *same* 512² view,
and gets ~90% of its *own* UV atlas.

Same GPU, same settings — roughly **10× the effective face detail**. This is why
a bigger GPU does not fix faces: you don't need more pixels, you need to stop
spending 90% of them on boots and rope. A 24GB card takes 384→768, which is 4×
face pixels; the split is ~10×, and the two compose.

Separate head and body materials is standard game-character practice, for exactly
this reason.

### Ordering is load-bearing

```
decimate the FULL mesh once  →  split  →  texture each half with --no-remesh
```

If you split first and let the pipeline decimate each half independently, the two
halves are simplified *differently along the cut* and you get a visible crack at
the neck. Decimating once, up front, means both halves share identical boundary
vertices.

### Known risks

- **Tone mismatch at the seam.** The two textures are generated independently and
  will not match perfectly. This character wears a hood, so the boundary is
  mostly hidden fabric — unusually favourable.
- **A head-only mesh is somewhat off-distribution** for the paint model. Heads
  are common objects, so this is expected to be fine, but it is unproven.

---

## 6. The full pipeline

```
  [1] concept image
        └─ letterbox to 1:1                         (§2)
  [2] generate_shape.py --octree 384                (Marvin or AWS)
        └─ character.glb          ~140k+ faces
  [3] decimate once to 40k                          (shared boundary)
  [4] split by height threshold  →  head.glb + body.glb
  [5] crop letterboxed image     →  head.png + body.png
  [6] texture_mesh.py head.glb head.png --no-remesh --resolution 512 --texture-size 4096
  [7] texture_mesh.py body.glb body.png --no-remesh --resolution 512 --texture-size 4096
  [8] merge → one GLB, two meshes, two PBR material sets
  [9] blender_ps1_pass.py  →  decimate to ~1200 tris, flat shade, FBX
```

Steps 3–5 and 8 are not yet built. Steps 1, 2, 6, 7, 9 exist.

---

## 7. Measured VRAM and timings (RTX 2070, 8GB, sm_75)

Upstream states 10GB shape / **21GB texture** / 29GB total (`README.md:76`).

| Configuration | Result |
|---|---|
| Paint pipeline resident, stock | **6.62GB** — leaves ~1.4GB, thrashes |
| Paint pipeline resident, `--offload-dino` | **4.51GB** — DINOv2-giant moved to CPU between calls |
| 6 views / 512 / 2048 / 4096 | Never completed. 20+ min pinned at 7,981 MiB |
| 4 views / 384 / 1024 / 2048, `--offload-dino` | **85.9s**, peak 8.16GB — completed end-to-end |
| `--cpu-offload` | **Fails.** The custom pipeline calls `vae.encode()` in `encode_images` outside the order accelerate's hooks expect → device mismatch |

Weight breakdown (fp16): UNet ~2.0GB, DINOv2-giant ~2.2GB, CLIP text encoder
~0.7GB, image encoder ~0.6GB, VAE ~0.2GB.

**Note on the failure mode:** on Windows the NVIDIA driver oversubscribes VRAM and
spills to system RAM rather than raising `OutOfMemoryError`. So exceeding VRAM
presents as a ~20× slowdown, not a crash. Do not wait on a run that has gone
quiet with memory pinned at the card limit.

On a 24GB card none of this applies — run 9 views / 768 and drop `--offload-dino`.

---

## 8. Windows-specific fixes (irrelevant on Linux)

Three classes of LP64 bug in `custom_rasterizer`, all because `long` is 64-bit on
Linux and 32-bit on Windows:

1. **`grid_neighbor.cpp`** — `size_t`→`int64_t` narrowing inside
   `torch::zeros({...})` braced-init-lists. MSVC makes this a hard error (C2398);
   GCC only warns. ~13 sites.
2. **`data_ptr<long>()`** on int64 tensors, in `grid_neighbor.cpp`,
   `rasterizer.cpp`, `rasterizer_gpu.cu` → `data_ptr<int64_t>()`.
3. **`rasterizer_gpu.cu:111` / `rasterizer.cpp:103`** — z-buffer seeded with
   `(long)maxint`, truncating the 64-bit sentinel `0x3FFFFFFF7FFFFFFF` to
   `0x7FFFFFFF`. Every pixel then started at depth 0 with face index 2147483647,
   which was used as an array index → *"CUDA error: an illegal memory access was
   encountered"*. This was the subtle one.

Also Windows-only: MSVC toolset must be pinned to **14.38** (`-vcvars_ver=14.38`)
because CUDA 12.4's nvcc rejects MSVC 19.40+, and the Build Tools default is
14.44. And `import torch` must precede importing the native extension — Python
3.8+ ignores `PATH` for extension DLLs, and importing torch is what registers
`torch/lib` (which carries `cudart64_12.dll`) as a DLL search directory.

Non-portability fixes that apply everywhere:

- `image_super_utils.py` — RealESRGAN ran with `tile=0` (whole image in one
  allocation). Now tiled via `HY3D_ESRGAN_TILE`, default 512.
- `mesh_utils.py` — Blender discovery was macOS-paths-only; now honours
  `BLENDER_PATH`, `PATH`, and Windows install locations. Subprocess timeout
  raised 30s → 600s.

---

## 9. Open items

- **Blender is not installed on the Windows box.** `download.blender.org` returns
  403 from this network, via both winget and direct fetch. `obj_to_glb.py` was
  written to produce the GLB via trimesh instead — and it actually produces a
  *better* GLB than the Blender path, because it packs metallic and roughness
  into a single glTF-correct texture (roughness in G, metallic in B) rather than
  leaving two separate greyscale JPEGs. The PS1 pass still needs real Blender.
- **Steps 3–5 and 8 of §6 are not built yet** — the decimate/split/crop/merge
  tooling.
- **`--cpu-offload` is broken** and would need per-module hand-wiring to recover.
  Irrelevant on a 24GB card.
- The head/body split is **designed but unproven**.

---

## 10. Files

| File | Purpose |
|---|---|
| `generate_shape.py` | image → mesh (MPS; needs a CUDA variant for AWS) |
| `texture_mesh.py` | mesh + image → textured OBJ. Flags: `--views --resolution --render-size --texture-size --no-remesh --offload-dino --progress` |
| `obj_to_glb.py` | textured OBJ → PBR GLB, no Blender required |
| `build_extensions_windows.ps1` | builds `custom_rasterizer` + `mesh_inpaint_processor` |
| `hy3dpaint/DifferentiableRenderer/setup_windows.py` | MSVC build for the inpaint extension |
| `requirements-windows.txt` | CUDA deps for Windows/Python 3.11 |
| `blender_ps1_pass.py` | GLB → decimated flat-shaded FBX |
| `blender_render_check.py` | headless 3-view render (needs Blender) |
