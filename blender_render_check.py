"""Headless Blender check: import a GLB and render 3 views for visual verification.

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender -b -P blender_render_check.py -- input.glb out_prefix
Writes out_prefix_view{0,1,2}.png and prints mesh stats.
"""
import math
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index('--') + 1:]
glb_path, out_prefix = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not meshes:
    print('ERROR: no mesh objects imported')
    sys.exit(1)

mins = mathutils.Vector((1e9, 1e9, 1e9))
maxs = mathutils.Vector((-1e9, -1e9, -1e9))
for o in meshes:
    for corner in o.bound_box:
        wc = o.matrix_world @ mathutils.Vector(corner)
        mins = mathutils.Vector((min(mins[i], wc[i]) for i in range(3)))
        maxs = mathutils.Vector((max(maxs[i], wc[i]) for i in range(3)))
center = (mins + maxs) / 2
size = max((maxs - mins))

cam_data = bpy.data.cameras.new('cam')
cam = bpy.data.objects.new('cam', cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'STUDIO'
scene.render.resolution_x = 640
scene.render.resolution_y = 640

for i, azimuth_deg in enumerate((0, 120, 240)):
    az = math.radians(azimuth_deg)
    direction = mathutils.Vector((math.sin(az), -math.cos(az), 0.45)).normalized()
    cam.location = center + direction * size * 2.4
    look = center - cam.location
    cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = f'{out_prefix}_view{i}.png'
    bpy.ops.render.render(write_still=True)

total_polys = sum(len(o.data.polygons) for o in meshes)
total_verts = sum(len(o.data.vertices) for o in meshes)
print(f'MESH STATS: objects={len(meshes)} polys={total_polys} verts={total_verts}')
print(f'BOUNDS: min={tuple(round(v, 3) for v in mins)} max={tuple(round(v, 3) for v in maxs)}')
print('RENDER OK')
