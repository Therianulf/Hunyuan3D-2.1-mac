"""Headless PS1-style pass: decimate a GLB to a target tri count, flat shade, export FBX.

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender -b -P blender_ps1_pass.py -- input.glb /abs/out.fbx 1200
Also renders /abs/out_preview.png for a quick look.
"""
import math
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index('--') + 1:]
glb_path, out_fbx, target_tris = argv[0], argv[1], int(argv[2])

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
    bpy.ops.object.shade_flat()

after = sum(len(o.data.polygons) for o in meshes)
print(f'DECIMATED: {total} -> {after} polys (target {target_tris})')

bpy.ops.export_scene.fbx(filepath=out_fbx)
print('FBX EXPORTED:', out_fbx)

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
scene.render.filepath = out_fbx.rsplit('.', 1)[0] + '_preview.png'
bpy.ops.render.render(write_still=True)
print('RENDER OK')
