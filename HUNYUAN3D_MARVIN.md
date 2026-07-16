# Hunyuan3D on Marvin (Mac Studio M4 Max, 64GB)

Goal: get Hunyuan3D image-to-3D shape generation running locally on Apple Silicon, producing GLB/OBJ meshes that feed a Blender low-poly pass for PS1-style game assets.

This doc is written for a human (Ben) plus a Claude Code agent working together. Agent: read the whole file before running anything.

> Status 2026-06-10: DONE. All three definition-of-done items verified on this machine via Path B (this repo). See Section 10 Notes for findings, deviations, and working commands.

---

## 0. Read this first: license reality check

Verified against the actual LICENSE file in Tencent-Hunyuan/Hunyuan3D-2 (Jan 2025 text):

- Section 6.d: Tencent claims no rights in Outputs. The meshes are yours.
- Section 5.c: you may not use, reproduce, modify, distribute, or display the Works **or Output** outside the "Territory." Territory = worldwide EXCLUDING the European Union, United Kingdom, and South Korea.
- Section 4: separate license required if you had >1M MAU at model release date. Not applicable here.
- Exhibit A item 12: machine-generated content placed in public contexts must be conspicuously identified as machine generated. Steam's AI disclosure checkbox likely covers this, but note it.

Practical meaning for this project:

- Private use, prototyping, and the summer build with your son in California: fully fine.
- Shipping Hunyuan-generated assets in a Steam game sold worldwide: NOT fine as-is, because Steam distributes to the EU/UK/SK by default. Options are region-restricting the store page or swapping production assets to rights-clean tools (SPAR3D/SF3D, TripoSR, TRELLIS) before launch.
- Recommended role for Hunyuan3D: shape-quality benchmark and prototyping engine. Treat shipped-asset generation as a separate decision.

Agent task: after cloning, re-read LICENSE in the repo (and the 2.1 repo's LICENSE if using Path B) and append any differences to the Notes section at the bottom of this file.

---

## 1. Agent brief

Environment:
- Machine: "Marvin," Mac Studio, M4 Max, 64GB unified memory, Apple Silicon (arm64), recent macOS.
- No NVIDIA GPU. No CUDA. GPU acceleration is PyTorch MPS (Metal) only.
- Ben is an experienced Python dev. Assume Homebrew exists; verify conda/miniforge before installing.

Hard constraints:
- Do not run `git commit` or `git push`. Ben handles all git history himself.
- Stay inside the working directory created for this project. Do not touch other repos on the machine.
- Do not attempt to build the texture pipeline native extensions (Section 3). They are CUDA-only and the failure mode wastes time.
- If a step fails twice with different fixes attempted, stop and report rather than thrashing.

Definition of done (minimum):
1. `smoke_test.glb` generated locally from a demo image via MPS.
2. Mesh opens in Blender and looks like the input subject.
3. Gradio app serves on localhost and generates from a dragged-in image.

Stretch: one full asset end-to-end (concept image -> mesh -> Blender decimate + flat shade -> FBX export).

---

## 2. What works on Apple Silicon and what does not

| Component | Mac status | Notes |
|---|---|---|
| Shape generation (`hy3dgen.shapegen`, DiT flow-matching pipeline) | WORKS on MPS | This is the part we want |
| Background removal (`hy3dgen.rembg`) | Works | CPU/onnxruntime |
| Texture generation (`hy3dgen.texgen`) | BROKEN on Mac | `custom_rasterizer` requires CUDA_HOME; `differentiable_renderer` fails with a clang `-bundle` vs `-dynamiclib` flag conflict. Confirmed upstream (GitHub issue #320), no official fix |
| FlashVDM turbo decoding | Works with `mc_algo='mc'` | The `dmc` algo needs the CUDA-only `diso` package; do not use it |

Consequence: generate geometry locally, do texturing in Blender. For the PS1 aesthetic this is no loss at all, since textures will be repainted at 64 to 256px with nearest-neighbor filtering anyway.

---

## 3. Model menu

| Model | Params | Why | HF repo / subfolder |
|---|---|---|---|
| Hunyuan3D-2mini Turbo (RECOMMENDED FIRST) | 0.6B | Smallest, step-distilled (5 steps), fastest on MPS | `tencent/Hunyuan3D-2mini`, subfolder `hunyuan3d-dit-v2-mini-turbo` |
| Hunyuan3D-2 base | 1.1B | Higher fidelity, slower (30 to 50 steps) | `tencent/Hunyuan3D-2`, subfolder `hunyuan3d-dit-v2-0` |
| Hunyuan3D-2.1 | larger | Best quality, use the Mac fork (Path B) | `tencent/Hunyuan3D-2.1` |

Disk budget: roughly 10 to 15GB for 2mini plus deps; 30GB+ if also pulling 2.0/2.1 weights. Set `HF_HOME` if the cache should live somewhere specific.

Memory hygiene: shape generation needs roughly 6 to 10GB. 64GB unified memory is plenty, but shut down any local LLM servers (Qwen/CoinWatchLLM, LM Studio, Ollama) during generation runs to avoid pressure.

---

## 4. Path A: official repo, shape-only (start here)

*(Kept for reference. In practice this machine went straight to Path B because the working directory was already a clone of the 2.1 Mac fork — see Notes.)*

### Step 1: prerequisites

```bash
xcode-select --install            # no-op if already present
python3 --version                 # any system python is fine, we use conda below
conda --version || brew install miniforge
```

Expected result: Xcode CLT present, conda available.

### Step 2: environment

```bash
conda create -n hunyuan3d python=3.11 -y
conda activate hunyuan3d
pip install --upgrade pip
pip install torch torchvision           # arm64 wheels ship MPS support
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

Expected result: `MPS: True`. If False, you are on x86 python (check `python -c "import platform; print(platform.machine())"` returns `arm64`) or torch is too old.

Pin python to 3.10, 3.11, or 3.12. Do not use 3.13; several deps lack wheels.

### Step 3: clone and install deps

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
cd Hunyuan3D-2
pip install -r requirements.txt
pip install -e .
```

Expected result: clean install of the shape pipeline.

If it fails: requirements.txt is written for Linux/CUDA. Known substitutions on Mac:
- `onnxruntime-gpu` -> replace with `onnxruntime`
- `xformers` -> remove entirely, not needed without CUDA
- `flash-attn` or `triton` -> remove, CUDA-only
- `pymeshlab` or `open3d` wheel errors -> `conda install -c conda-forge pymeshlab open3d`
Comment out the offending line, install the substitute, note it at the bottom of this file.

DO NOT run these (texture extras, CUDA-only, will fail):
```bash
# cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install     <- skip
# cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install  <- skip
```

### Step 4: environment variables

Add to the shell session (and Ben can add to ~/.zshrc later):

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # required: unimplemented MPS ops fall back to CPU instead of crashing
# export HF_HOME=/path/to/big/disk/hf  # optional cache relocation
```

### Step 5: smoke test

Create `smoke_test.py` in the repo root:

```python
import torch
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('device:', device)

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-dit-v2-mini-turbo',
    use_safetensors=True,
    device=device,
)
pipeline.enable_flashvdm(mc_algo='mc')   # 'mc' only; 'dmc' is CUDA-only

image = Image.open('assets/demo.png')
if image.mode == 'RGB':
    image = BackgroundRemover()(image)

mesh = pipeline(
    image=image,
    num_inference_steps=5,        # turbo is step-distilled, 5 is correct
    octree_resolution=256,        # start low for speed; 380 for more detail
    num_chunks=8000,
    generator=torch.manual_seed(2026),
    output_type='trimesh',
)[0]

mesh.export('smoke_test.glb')
print('faces:', len(mesh.faces), 'vertices:', len(mesh.vertices))
```

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python smoke_test.py
```

Expected result: first run downloads weights (minutes) and compiles MPS shaders, then writes `smoke_test.glb` with a nonzero face count. Generation itself should land in roughly 1 to 5 minutes on the M4 Max. The sub-second marketing numbers are NVIDIA data-center benchmarks; ignore them.

IMPORTANT for the agent: this script matches the documented API as of early 2026, but these repos move. If imports or kwargs fail, diff against the minimal example in the repo's current README and `examples/` before debugging anything else. Trust the README over this file.

### Step 6: visual verification

Open `smoke_test.glb` in Blender (File -> Import -> glTF 2.0) or quick-look it. Confirm the mesh resembles the input image, has sane scale, and is not inside-out (Alt+N -> Recalculate Outside if normals look wrong).

---

## 5. Path B: Hunyuan3D-2.1 Mac fork (if A succeeds and more quality is wanted)

The `Brainkeys/Hunyuan3D-2.1-mac` fork ships MPS acceleration and CUDA-free fallbacks for 2.1, with its own `README_macOS.md`. Same rules apply: shape only, skip texture extras, `mc` algo, MPS fallback env var.

```bash
git clone https://github.com/Brainkeys/Hunyuan3D-2.1-mac.git
cd Hunyuan3D-2.1-mac
# follow README_macOS.md in that repo
```

Agent: check the fork's last-commit date first. If it has gone stale relative to upstream 2.1, say so before investing time. Also re-check the 2.1 LICENSE; expect the same territory terms as 2.0 but verify.

## 6. Path C: Pinokio escape hatch

If dependency hell wins on both paths, Pinokio (pinokio.computer) has a one-click Hunyuan3D installer for Mac that manages its own isolated env. Less controllable, but it works. Use it to confirm the model is worth the effort, then return to Path A.

---

## 7. Gradio app (the sit-down-with-your-son workflow)

Once the smoke test passes:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python gradio_app.py \
  --model_path tencent/Hunyuan3D-2mini \
  --subfolder hunyuan3d-dit-v2-mini-turbo \
  --device mps --enable_flashvdm
```

Flags may differ by repo version; `python gradio_app.py --help` is authoritative. Expected result: a local web UI where you drag in an image and download a GLB. There is also `api_server.py --device mps` if you would rather hit it programmatically from another tool.

Note: the UI will have texture options. They will fail on Mac. Shape only.

*(See Notes for the exact command verified working on this machine for the 2.1 fork.)*

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Install error mentioning `CUDA_HOME` | You are building texture extras | Skip them entirely (Section 4, Step 3) |
| clang error: `-bundle` not allowed with `-dynamiclib` | `differentiable_renderer` build | Same: skip, texture in Blender |
| `NotImplementedError: aten::... not implemented for MPS` | Missing MPS op | Ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in the same shell before launching python |
| Import error for `diso` or dmc complaints | Turbo decoder set to `dmc` | Use `enable_flashvdm(mc_algo='mc')` |
| `Cannot convert a MPS Tensor to float64` | Some op upcasting | Patch the offending tensor to `.float()` / float32; report location |
| Severe slowdown, fans, swap | Memory pressure | Close LLM servers and browsers; lower `octree_resolution` to 256 and `num_chunks` |
| HF download stalls | Network/cache | `pip install hf_transfer`, set `HF_HUB_ENABLE_HF_TRANSFER=1`, or pre-pull with `huggingface-cli download tencent/Hunyuan3D-2mini` |
| Output mesh is noise/blob | Bad input image | Use a single centered object, plain background, RGBA after BackgroundRemover |

Rollback: everything lives in the conda env and the cloned repo dir. `conda env remove -n hunyuan3d` and delete the directory restores the machine to clean state. The HF cache can be cleared with `huggingface-cli delete-cache`.

*(Rollback on this machine: delete `.venv/` in the repo and `~/.cache/hy3dgen/` — no conda was installed. See Notes.)*

---

## 9. Hand-off to Blender (PS1 pass, summary)

Hunyuan output is dense smooth triangle soup. The retro look is made in post:

1. Import GLB.
2. Decimate modifier (Collapse), ratio tuned to land roughly 200 to 1,500 tris for props.
3. Shade Flat plus Weighted Normals for faceted PS1 surfaces.
4. Hand-paint or bake a 64 to 256px texture, Image Texture node interpolation set to Closest.
5. Export FBX/GLB for the engine. Export STL only if a print is wanted; run 3D Print Toolbox checks first since AI meshes are often non-manifold.

Full pipeline details, prompt templates, and the concept-image-first workflow live in the research doc from this conversation.

---

## 10. Notes (agent appends findings here)

All notes below recorded 2026-06-10 by the Claude Code agent after completing setup on Marvin.

### What was actually done (deviations from the plan above)

- **Went straight to Path B.** The working directory was already a fresh clone of `Brainkeys/Hunyuan3D-2.1-mac` (via the `zettai-seigi` intermediate fork), so Path A was never run. All definition-of-done items were completed against Hunyuan3D-2.1.
- **venv instead of miniforge.** conda was not installed; instead of `brew install miniforge`, a venv was created at `.venv/` inside the repo from pyenv's Python 3.11.14 (already on the machine). Lighter footprint, stays inside the working directory, and no conda-forge substitute turned out to be needed. Rollback = `rm -rf .venv ~/.cache/hy3dgen`.
- **`install-macos.sh` was NOT run.** It compiles the texture-pipeline natives (`custom_rasterizer`, `DifferentiableRenderer`) — exactly what Section 1 forbids — and downloads a texture-only RealESRGAN checkpoint. Install was done manually: `pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1`, then `pip install -r requirements-macos.txt`, then `pip install pymeshlab`.
- **No dependency substitutions were needed.** `requirements-macos.txt` is already scrubbed for Mac. Every package resolved to an arm64 wheel on Python 3.11, including `open3d` 0.19 and `pymeshlab` 2025.7.post1 (the old "ARM64 issues" warnings in the fork's comments are outdated). `tb_nightly==2.18.0a20240726` (a pinned nightly) still resolves but could be yanked someday — substitute stable `tensorboard` if it 404s.

### Verified results (definition of done)

1. **`smoke_test.glb` generated via MPS** — `PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python smoke_test.py` (script in repo root, written against this fork's actual API). fp16 worked directly on MPS, no fp32 fallback needed. Timings on the M4 Max: pipeline load 64s, 30 diffusion steps at ~5.1s/step (2:37), volume decoding 2:42, **total generation 323s (~5.4 min)** at `octree_resolution=256`, `num_chunks=8000`, `mc_algo='mc'`. Output: 321,170 faces / 160,579 vertices, 5.5MB GLB.
2. **Opens in Blender and matches the subject** — verified headlessly with Blender 5.0.1 (`blender_render_check.py` in repo root renders any GLB from 3 angles). The mesh is unmistakably the demo penguin, including embossed "HY3D" lettering on its sign. Note: mesh is NOT watertight (euler number −2) — normal for AI meshes, irrelevant for game assets, but run 3D Print Toolbox first if printing.
3. **Gradio serves on localhost** — working command for this fork (README's `--low_vram_mode` skipped; 64GB doesn't need it):
   ```bash
   PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python gradio_app.py \
     --model_path tencent/Hunyuan3D-2.1 --subfolder hunyuan3d-dit-v2-1 \
     --device mps --disable_tex --host 127.0.0.1 --port 8081
   ```
   `--disable_tex` is mandatory: on a high-end Mac the fork's hardware detector would otherwise *enable* texture gen, which then attempts the CPU-fallback texture path. The shape path force-uses `mc` on mps (gradio_app.py:895) and FlashVDM is force-disabled in the gradio path (line 784) regardless of flags. Note the app serves plain uvicorn — there is no "Running on local URL" gradio banner; readiness line is `Uvicorn running on http://127.0.0.1:8081`.
   End-to-end generation through the app was verified via its `/shape_generation` API (same code path as the UI's Gen Shape button): 312.9s, 215,810 faces from `assets/demo.png` at the same settings as the smoke test.

4. **Stretch goal complete** — `blender_ps1_pass.py` (repo root) runs the PS1 pass headlessly: decimate 321,166 → 1,200 tris, flat shade, FBX export. Output `penguin_ps1.fbx` + `penguin_ps1_preview.png` look correctly faceted. Usage:
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender -b -P blender_ps1_pass.py -- input.glb /abs/out.fbx 1200
   ```

### License findings for 2.1 (the Section 0 agent task)

Tencent Hunyuan 3D 2.1 Community License, release date June 13, 2025. Verified against both the fork's LICENSE and upstream:

- **All four Section 0 facts carry over unchanged**: Outputs are yours (6(d)); Territory excludes EU/UK/South Korea (1(l)) with use/distribution of Works *or Output* outside it prohibited (5(c)); >1M MAU clause (Section 4); machine-generated-content disclosure (Exhibit A item 12). 2.1's 5(c) adds an explicit sentence: "Any such use outside the Territory is unlicensed and unauthorized under this Agreement."
- **NEW in 2.1: Section 3(e)** — if you deploy the Works in any service/product for third parties, you must prominently disclose the actual provider's legal name and state that Tencent is not affiliated or endorsing. Not relevant to local personal use; relevant if the Gradio app were ever exposed to others as a service.
- **The fork's LICENSE file omits Section 3(e) entirely** (its Section 3 runs a–d only; upstream has a–e). The omission has no legal effect — upstream's text controls — but treat the upstream LICENSE as authoritative, not the fork's copy.
- Other notable: Outputs may not be used to train other AI models (5(b)); suing Tencent over the Materials terminates the license (6(c)); governing law Hong Kong SAR.
- **Practical posture unchanged from Section 0**: prototyping in California fine; worldwide Steam shipping of Hunyuan meshes not compliant as-is. Keep Hunyuan as the prototyping/benchmark engine.

### Fork staleness (the Section 5 agent task)

Checked 2026-06-10. Fork's last commit: 2025-08-10. Upstream `Tencent-Hunyuan/Hunyuan3D-2.1` tip: 2025-10-17 (~13 commits ahead). **Verdict: safe — nothing missing affects shape generation.** The only shape-touching upstream changes are a training-only config fix (PR #102) and opt-in FlashVDM speed flags (PR #104); shape weights on HF unchanged since June 2025. Worth watching: upstream **PR #206 (open, April 2026)** adds official MPS support to the main repo and may eventually make this fork unnecessary.

### Fork gotchas discovered

- `demo_macos.py` is broken: it calls `from_pretrained(model_path, device_map="auto", torch_dtype=device.type)` but this fork's signature takes `device=` / `dtype=` (a torch dtype, not a string). Use `smoke_test.py` instead.
- `run_gradio_macos.sh` hardcodes the fork author's personal miniconda path (`/opt/homebrew/Caskroom/miniconda/...`) — useless here; launch `gradio_app.py` directly as shown above.
- Weights cache to `~/.cache/hy3dgen/` (NOT `~/.cache/huggingface`), and only the requested subfolder is downloaded (`hunyuan3d-dit-v2-1` ≈ 4.9GB fp16 ckpt + config).
- The torchvision functional_tensor shim (`torchvision_fix.apply_fix()`) must run before pipeline imports; `smoke_test.py` does this.
- gradio_app.py's CUDA texture-compile block is safely gated behind `ENV == 'Huggingface'` (hardcoded `'Local'` at line 71) — launching locally never triggers it.
- Optional speedup for future downloads: `pip install hf_xet` (HF log suggested it; downloads fell back to plain HTTP).

### Files added to the repo by the agent (untracked; Ben decides their git fate)

- `smoke_test.py` — working MPS smoke test for this fork's API
- `blender_render_check.py` — headless 3-view GLB render for visual verification (`Blender -b -P blender_render_check.py -- file.glb /abs/out_prefix`)
- `blender_ps1_pass.py` — headless decimate + flat shade + FBX export (the PS1 pass)
- `smoke_test.glb`, `render_check_view{0,1,2}.png` — smoke test artifacts
- `gradio_api_test.glb` — mesh generated end-to-end through the running Gradio app's API
- `penguin_ps1.fbx`, `penguin_ps1_preview.png` — stretch-goal output (1,200-tri PS1 penguin)
- `HUNYUAN3D_MARVIN.md` — this file (the planning doc was not previously on disk)
