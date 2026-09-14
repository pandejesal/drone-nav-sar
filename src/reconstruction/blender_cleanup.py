#!/usr/bin/env python3
"""
BlenderProc Cleanup Script
Post-processes reconstructed meshes: decimate, UV unwrap, PBR bake, collision hulls, export .glb/.usd
Run with: blender --background --python blender_cleanup.py -- <input.glb> <output.glb> [--target-tris 50000]
"""

import sys
import os
import argparse
import bpy
import math
import bmesh
from mathutils import Vector, Matrix


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Clear orphaned data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def import_mesh(filepath: str):
    """Import mesh (supports .glb, .obj, .ply, .fbx)."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.glb' or ext == '.gltf':
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == '.ply':
        bpy.ops.import_mesh.ply(filepath=filepath)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Get imported mesh objects
    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not mesh_objects:
        raise RuntimeError("No mesh objects found after import")

    # Join if multiple
    if len(mesh_objects) > 1:
        bpy.context.view_layer.objects.active = mesh_objects[0]
        bpy.ops.object.join()

    return bpy.context.active_object


def remove_loose_geometry(obj):
    """Remove loose vertices, edges, faces."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode='OBJECT')
    log("Removed loose geometry")


def remove_non_manifold(obj):
    """Remove non-manifold geometry."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')
    log("Removed non-manifold geometry")


def smart_uv_unwrap(obj, margin: float = 0.001):
    """Smart UV project for lightmap/PBR baking."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(66),
        island_margin=margin,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    bpy.ops.object.mode_set(mode='OBJECT')
    log("Smart UV unwrap complete")


def decimate_mesh(obj, target_tris: int):
    """Decimate mesh to target triangle count."""
    current_tris = len(obj.data.polygons)
    if current_tris <= target_tris:
        log(f"Mesh already under target ({current_tris} <= {target_tris}), skipping decimation")
        return

    ratio = target_tris / current_tris
    log(f"Decimating: {current_tris} → {target_tris} tris (ratio: {ratio:.3f})")

    mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    mod.ratio = ratio
    mod.use_collapse_triangulate = True

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Decimate")

    new_tris = len(obj.data.polygons)
    log(f"Decimated to {new_tris} triangles")


def create_pbr_materials(obj, texture_size: int = 2048):
    """Create PBR materials and bake textures."""
    # Ensure UVs exist
    if not obj.data.uv_layers:
        smart_uv_unwrap(obj)

    # Create material
    mat = bpy.data.materials.new(name="PBR_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)

    # Texture nodes
    tex_albedo = nodes.new('ShaderNodeTexImage')
    tex_albedo.location = (-400, 200)
    tex_albedo.name = "Albedo"
    tex_albedo.label = "Albedo"

    tex_normal = nodes.new('ShaderNodeTexImage')
    tex_normal.location = (-400, 0)
    tex_normal.name = "Normal"
    tex_normal.label = "Normal"
    tex_normal.image = bpy.data.images.new(
        name="Normal", width=texture_size, height=texture_size,
        alpha=False, float_buffer=True
    )
    tex_normal.image.colorspace_settings.name = 'Non-Color'

    tex_roughness = nodes.new('ShaderNodeTexImage')
    tex_roughness.location = (-400, -200)
    tex_roughness.name = "Roughness"
    tex_roughness.label = "Roughness"
    tex_roughness.image = bpy.data.images.new(
        name="Roughness", width=texture_size, height=texture_size,
        alpha=False, float_buffer=True
    )
    tex_roughness.image.colorspace_settings.name = 'Non-Color'

    tex_metallic = nodes.new('ShaderNodeTexImage')
    tex_metallic.location = (-400, -400)
    tex_metallic.name = "Metallic"
    tex_metallic.label = "Metallic"
    tex_metallic.image = bpy.data.images.new(
        name="Metallic", width=texture_size, height=texture_size,
        alpha=False, float_buffer=True
    )
    tex_metallic.image.colorspace_settings.name = 'Non-Color'

    # Normal map node
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.location = (-100, 0)

    # Connect
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    links.new(tex_albedo.outputs['Color'], principled.inputs['Base Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    links.new(tex_roughness.outputs['Color'], principled.inputs['Roughness'])
    links.new(tex_metallic.outputs['Color'], principled.inputs['Metallic'])
    links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])

    # Set defaults
    principled.inputs['Roughness'].default_value = 0.5
    principled.inputs['Metallic'].default_value = 0.0

    # Assign material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    log(f"Created PBR material with {texture_size}px textures")
    return mat, {
        'albedo': tex_albedo,
        'normal': tex_normal,
        'roughness': tex_roughness,
        'metallic': tex_metallic,
    }


def bake_textures(obj, texture_nodes: dict, output_dir: str, texture_size: int = 2048):
    """Bake PBR textures from existing materials or geometry."""
    log("Baking PBR textures...")

    # Set up render engine for baking
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU' if hasattr(bpy.context.preferences.addons['cycles'], 'preferences') else 'CPU'
    bpy.context.scene.cycles.samples = 16  # Fast for baking
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True
    bpy.context.scene.render.bake.use_selected_to_active = False
    bpy.context.scene.render.bake.margin = 16

    # Bake each texture type
    bake_configs = [
        ('albedo', 'DIFFUSE', {'use_pass_color': True}),
        ('normal', 'NORMAL', {'use_pass_normal': True}),
        ('roughness', 'ROUGHNESS', {'use_pass_roughness': True}),
        ('metallic', 'METALLIC', {'use_pass_metallic': True}),
    ]

    for tex_name, bake_type, settings in bake_configs:
        tex_node = texture_nodes[tex_name]
        if not tex_node.image:
            continue

        # Set active texture node
        bpy.context.view_layer.objects.active = obj
        obj.active_material_index = 0
        bpy.ops.object.mode_set(mode='OBJECT')

        # Ensure node is selected for baking
        nodes = obj.active_material.node_tree.nodes
        for node in nodes:
            node.select = False
        tex_node.select = True
        nodes.active = tex_node

        # Configure bake
        for k, v in settings.items():
            setattr(bpy.context.scene.render.bake, k, v)
        bpy.context.scene.render.bake_type = bake_type

        # Bake
        bpy.ops.object.bake(type=bake_type)

        # Save image
        output_path = os.path.join(output_dir, f"{os.path.basename(output_dir)}_{tex_name}.png")
        tex_node.image.save_render(output_path)
        log(f"  Baked {tex_name}: {output_path}")

    log("Texture baking complete")


def generate_collision_hulls(obj, max_hulls: int = 10):
    """Generate convex collision hulls using VHACD approximation."""
    log("Generating collision hulls...")

    # Create a copy for collision
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    collision_obj = bpy.context.active_object
    collision_obj.name = f"{obj.name}_collision"

    # Simple convex hull approximation (Blender doesn't have VHACD built-in)
    # Use decimate + convex hull as approximation
    bpy.context.view_layer.objects.active = collision_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.convex_hull()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Move to separate collection
    coll_collection = bpy.data.collections.get("Collision")
    if not coll_collection:
        coll_collection = bpy.data.collections.new("Collision")
        bpy.context.scene.collection.children.link(coll_collection)

    bpy.context.collection.objects.unlink(collision_obj)
    coll_collection.objects.link(collision_obj)

    log(f"Created collision hull: {collision_obj.name}")
    return collision_obj


def normalize_scale(obj):
    """Center pivot, apply scale, normalize to unit bounding box."""
    bpy.context.view_layer.objects.active = obj

    # Apply transforms
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Center origin to geometry center
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # Normalize to unit scale (max dimension = 1.0)
    max_dim = max(obj.dimensions)
    if max_dim > 0:
        scale_factor = 1.0 / max_dim
        obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    log(f"Normalized scale: max dimension = 1.0")


def export_glb(obj, output_path: str):
    """Export as GLB with embedded textures."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=True,
        export_apply=True,
        export_colors=True,
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_materials='EXPORT',
        export_image_format='AUTO',
    )
    log(f"Exported GLB: {output_path}")


def export_usd(obj, output_path: str):
    """Export as USD (requires Blender with USD support)."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    try:
        bpy.ops.wm.usd_export(
            filepath=output_path,
            selected_objects_only=True,
            export_materials=True,
            export_textures=True,
        )
        log(f"Exported USD: {output_path}")
    except AttributeError:
        log("USD export not available in this Blender build", "WARNING")
        # Fallback: export OBJ
        obj_path = output_path.replace('.usd', '.obj')
        bpy.ops.export_scene.obj(
            filepath=obj_path,
            use_selection=True,
            use_materials=True,
        )
        log(f"Exported OBJ (fallback): {obj_path}")


def main():
    parser = argparse.ArgumentParser(description="BlenderProc Mesh Cleanup")
    parser.add_argument("input_file", help="Input mesh file (.glb, .obj, .ply, .fbx)")
    parser.add_argument("output_file", help="Output mesh file (.glb)")
    parser.add_argument("--target-tris", type=int, default=50000, help="Target triangle count")
    parser.add_argument("--texture-size", type=int, default=2048, help="PBR texture resolution")
    parser.add_argument("--skip-bake", action="store_true", help="Skip texture baking")
    parser.add_argument("--skip-collision", action="store_true", help="Skip collision hull generation")
    parser.add_argument("--export-usd", action="store_true", help="Also export USD")

    args = parser.parse_args()

    log("=" * 60)
    log("BlenderProc Mesh Cleanup")
    log(f"  Input: {args.input_file}")
    log(f"  Output: {args.output_file}")
    log(f"  Target tris: {args.target_tris}")
    log("=" * 60)

    # Clear scene
    clear_scene()

    # Import
    try:
        obj = import_mesh(args.input_file)
        log(f"Imported: {obj.name}")
    except Exception as e:
        log(f"Import failed: {e}", "ERROR")
        sys.exit(1)

    # Cleanup pipeline
    remove_loose_geometry(obj)
    remove_non_manifold(obj)

    # Decimate if needed
    decimate_mesh(obj, args.target_tris)

    # UV unwrap
    smart_uv_unwrap(obj)

    # Create PBR materials
    mat, texture_nodes = create_pbr_materials(obj, args.texture_size)

    # Bake textures
    if not args.skip_bake:
        output_dir = os.path.dirname(args.output_file)
        bake_textures(obj, texture_nodes, output_dir, args.texture_size)

    # Generate collision hulls
    if not args.skip_collision:
        generate_collision_hulls(obj)

    # Normalize
    normalize_scale(obj)

    # Export GLB
    export_glb(obj, args.output_file)

    # Export USD if requested
    if args.export_usd:
        usd_path = args.output_file.replace('.glb', '.usd')
        export_usd(obj, usd_path)

    log("=" * 60)
    log("BlenderProc cleanup completed!")
    log(f"  Output: {args.output_file}")
    log("=" * 60)


if __name__ == "__main__":
    main()