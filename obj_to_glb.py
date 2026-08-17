"""Convert a textured OBJ from the paint pipeline into a PBR GLB, without Blender.

textureGenPipeline calls DifferentiableRenderer.mesh_utils.convert_obj_to_glb at
the end, which needs either bpy or an external Blender binary. Neither is
available here (download.blender.org is blocked from this network), so the
pipeline writes the .obj + maps and silently skips the .glb. This fills that gap.

It also produces a *better* GLB than the Blender path for engine use: the paint
pipeline writes metallic and roughness as two separate greyscale JPEGs, which is
not what glTF wants. glTF expects ONE metallicRoughness texture with roughness in
the green channel and metallic in the blue. This packs them correctly.

Usage:
    .venv\\Scripts\\python.exe obj_to_glb.py sneaker_tpose_textured.obj [out.glb]
"""
import argparse
import os
import sys

import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial

parser = argparse.ArgumentParser()
parser.add_argument("obj")
parser.add_argument("glb", nargs="?", help="defaults to <obj basename>.glb")
args = parser.parse_args()

base = os.path.splitext(args.obj)[0]
out = args.glb or base + ".glb"

mesh = trimesh.load(args.obj, force="mesh", process=False)
uv = getattr(mesh.visual, "uv", None)
if uv is None:
    sys.exit(f"ERROR: {args.obj} has no UV coordinates; cannot build a textured GLB")
print(f"loaded {len(mesh.faces)} faces, {len(mesh.vertices)} vertices, uv {uv.shape}")


def load(suffix):
    for ext in (".jpg", ".png"):
        p = base + suffix + ext
        if os.path.exists(p):
            return Image.open(p)
    return None


albedo = load("")
metallic = load("_metallic")
roughness = load("_roughness")
normal = load("_normal")
print(f"maps: albedo={albedo is not None} metallic={metallic is not None} "
      f"roughness={roughness is not None} normal={normal is not None}")

mr = None
if metallic is not None or roughness is not None:
    # glTF metallicRoughness packing: R unused, G = roughness, B = metallic.
    ref = (metallic or roughness).size
    g = (roughness or Image.new("L", ref, 255)).convert("L").resize(ref)
    b = (metallic or Image.new("L", ref, 0)).convert("L").resize(ref)
    mr = Image.merge("RGB", (Image.new("L", ref, 0), g, b))

material = PBRMaterial(
    baseColorTexture=albedo.convert("RGB") if albedo else None,
    metallicRoughnessTexture=mr,
    normalTexture=normal.convert("RGB") if normal else None,
    # Factors must be 1.0 so the sampled texture values are used as-is rather
    # than being scaled down by a default factor.
    metallicFactor=1.0 if metallic is not None else 0.0,
    roughnessFactor=1.0,
)
mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

mesh.export(out)
print(f"OK: {out} written ({os.path.getsize(out) / 1024**2:.1f}MB)")
