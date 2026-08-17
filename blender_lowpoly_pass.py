"""Headless low-poly pass: decimate a GLB to a target tri count, set shading, export GLB.

Successor to blender_ps1_pass.py — exports GLB instead of FBX, and makes shading a choice
rather than forcing the faceted PS1 look. Dark-Engine-era models were Gouraud shaded, so
'auto' (smooth, with hard edges preserved above the angle threshold) is the default.

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender -b -P blender_lowpoly_pass.py -- \
        input.glb /abs/out.glb 1200 [auto|smooth|flat] [angle_degrees]

Output path must be absolute. Also renders /abs/out_preview.png for a quick look.
"""
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index('--') + 1:]
glb_path, out_glb, target_tris = argv[0], argv[1], int(argv[2])
shading = argv[3] if len(argv) > 3 else 'auto'
smooth_angle = float(argv[4]) if len(argv) > 4 else 30.0

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
total = sum(len(o.data.polygons) for o in meshes)
for o in meshes:
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    mod = o.modifiers.new('dec', 'DECIMATE')
    mod.ratio = min(1.0, target_tris / total)
    bpy.ops.object.modifier_apply(modifier='dec')

    if shading == 'flat':
        bpy.ops.object.shade_flat()
    elif shading == 'smooth':
        bpy.ops.object.shade_smooth()
    else:  # auto: smooth curves, keep hard edges above the angle threshold
        try:
            bpy.ops.object.shade_auto_smooth(angle=smooth_angle * 3.14159265 / 180.0)
        except AttributeError:  # older Blender without the operator
            bpy.ops.object.shade_smooth()

after = sum(len(o.data.polygons) for o in meshes)
print(f'DECIMATED: {total} -> {after} polys (target {target_tris}, shading={shading})')

bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB')
print('GLB EXPORTED:', out_glb)

mins = mathutils.Vector((1e9, 1e9, 1e9))
maxs = mathutils.Vector((-1e9, -1e9, -1e9))
for o in meshes:
    for corner in o.bound_box:
        wc = o.matrix_world @ mathutils.Vector(corner)
        mins = mathutils.Vector((min(mins[i], wc[i]) for i in range(3)))
        maxs = mathutils.Vector((max(maxs[i], wc[i]) for i in range(3)))
center = (mins + maxs) / 2
size = max(maxs - mins)

cam_data = bpy.data.cameras.new('cam')
cam = bpy.data.objects.new('cam', cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam
direction = mathutils.Vector((0.0, -1.0, 0.45)).normalized()
cam.location = center + direction * size * 2.4
cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'STUDIO'
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.filepath = out_glb.rsplit('.', 1)[0] + '_preview.png'
bpy.ops.render.render(write_still=True)
print('RENDER OK')
