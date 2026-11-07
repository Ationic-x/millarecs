//! wgpu pipeline setup: one WGSL module with two instanced-quad pipelines
//! (world sprites with SDF shapes, and atlas-sampled HUD glyphs).

use wgpu::{
    BindGroupLayout, BlendState, ColorTargetState, ColorWrites, Device, FragmentState,
    MultisampleState, PrimitiveState, RenderPipeline, TextureFormat, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};

/// Shared WGSL for both pipelines. The camera uniform converts pixel-space
/// positions to clip space (Y flipped to match screen coordinates).
pub const SHADER_WGSL: &str = r#"
struct Camera {
    screen: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_samp: sampler;

fn to_clip(p: vec2<f32>) -> vec4<f32> {
    let ndc = p / camera.screen * 2.0 - vec2<f32>(1.0, 1.0);
    return vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
}

struct SpriteOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) shape: f32,
};

@vertex
fn vs_sprite(
    @location(0) corner: vec2<f32>,
    @location(1) center: vec2<f32>,
    @location(2) extent: vec2<f32>,
    @location(3) rot: f32,
    @location(4) color: vec4<f32>,
    @location(5) shape: u32,
) -> SpriteOut {
    let s = sin(rot);
    let c = cos(rot);
    let local = vec2<f32>(corner.x * extent.x, corner.y * extent.y);
    let world = center + vec2<f32>(local.x * c - local.y * s, local.x * s + local.y * c);
    var result: SpriteOut;
    result.clip = to_clip(world);
    result.local = corner;
    result.color = color;
    result.shape = f32(shape);
    return result;
}

fn circle_alpha(p: vec2<f32>) -> f32 {
    let d = dot(p, p);
    let aa = fwidth(d) * 1.5 + 1e-6;
    return 1.0 - smoothstep(1.0 - aa, 1.0 + aa, d);
}

fn tri_alpha(p: vec2<f32>) -> f32 {
    let a = vec2<f32>(1.0, 0.0);
    let b = vec2<f32>(-0.72, 0.7);
    let c = vec2<f32>(-0.72, -0.7);
    let w0 = (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
    let w1 = (p.x - c.x) * (b.y - c.y) - (b.x - c.x) * (p.y - c.y);
    let w2 = (p.x - a.x) * (c.y - a.y) - (c.x - a.x) * (p.y - a.y);
    let aa = max(max(fwidth(w0), fwidth(w1)), fwidth(w2)) + 1e-6;
    return min(
        min(smoothstep(-aa, aa, w0), smoothstep(-aa, aa, w1)),
        smoothstep(-aa, aa, w2),
    );
}

@fragment
fn fs_sprite(sprite: SpriteOut) -> @location(0) vec4<f32> {
    var a = 1.0;
    if (sprite.shape > 0.5 && sprite.shape < 1.5) {
        a = circle_alpha(sprite.local);
    } else if (sprite.shape > 1.5) {
        a = tri_alpha(sprite.local);
    }
    let alpha = sprite.color.a * a;
    if (alpha <= 0.003) {
        discard;
    }
    return vec4<f32>(sprite.color.rgb, alpha);
}

struct GlyphOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_glyph(
    @location(0) corner: vec2<f32>,
    @location(1) pos: vec2<f32>,
    @location(2) size: vec2<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) color: vec4<f32>,
) -> GlyphOut {
    let t = corner * 0.5 + vec2<f32>(0.5, 0.5);
    var result: GlyphOut;
    result.clip = to_clip(pos + t * size);
    result.uv = mix(uv0, uv1, vec2<f32>(1.0 - t.x, t.y));
    result.color = color;
    return result;
}

@fragment
fn fs_glyph(glyph_quad: GlyphOut) -> @location(0) vec4<f32> {
    let cover = textureSample(atlas_tex, atlas_samp, glyph_quad.uv).r;
    let alpha = glyph_quad.color.a * cover;
    if (alpha <= 0.003) {
        discard;
    }
    return vec4<f32>(glyph_quad.color.rgb, alpha);
}
"#;

/// The unit-quad corner data: two triangles covering `[-1, 1]²`.
pub const QUAD_CORNERS: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [1.0, 1.0],
];

/// Per-vertex corner buffer layout (stride 8).
fn quad_layout() -> VertexBufferLayout<'static> {
    VertexBufferLayout {
        array_stride: 8,
        step_mode: VertexStepMode::Vertex,
        attributes: &[VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        }],
    }
}

/// Sprite instance buffer layout (44-byte stride; must match
/// [`crate::draw::SpriteInstance`]).
fn sprite_layout() -> VertexBufferLayout<'static> {
    VertexBufferLayout {
        array_stride: 44,
        step_mode: VertexStepMode::Instance,
        attributes: &[
            VertexAttribute { format: VertexFormat::Float32x2, offset: 0, shader_location: 1 },
            VertexAttribute { format: VertexFormat::Float32x2, offset: 8, shader_location: 2 },
            VertexAttribute { format: VertexFormat::Float32, offset: 16, shader_location: 3 },
            VertexAttribute { format: VertexFormat::Float32x4, offset: 20, shader_location: 4 },
            VertexAttribute { format: VertexFormat::Uint32, offset: 36, shader_location: 5 },
        ],
    }
}

/// Glyph instance buffer layout (48-byte stride; must match
/// [`crate::draw::GlyphInstance`]).
fn glyph_layout() -> VertexBufferLayout<'static> {
    VertexBufferLayout {
        array_stride: 48,
        step_mode: VertexStepMode::Instance,
        attributes: &[
            VertexAttribute { format: VertexFormat::Float32x2, offset: 0, shader_location: 1 },
            VertexAttribute { format: VertexFormat::Float32x2, offset: 8, shader_location: 2 },
            VertexAttribute { format: VertexFormat::Float32x2, offset: 16, shader_location: 3 },
            VertexAttribute { format: VertexFormat::Float32x2, offset: 24, shader_location: 4 },
            VertexAttribute { format: VertexFormat::Float32x4, offset: 32, shader_location: 5 },
        ],
    }
}

/// Builds both render pipelines for `format`.
pub fn create_pipelines(
    device: &Device,
    format: TextureFormat,
    camera_bgl: &BindGroupLayout,
    atlas_bgl: &BindGroupLayout,
) -> (RenderPipeline, RenderPipeline) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("millarecs_playground_shaders"),
        source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
    });

    let camera_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sprite_pipeline_layout"),
        bind_group_layouts: &[Some(camera_bgl)],
        immediate_size: 0,
    });
    let glyph_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("glyph_pipeline_layout"),
        bind_group_layouts: &[Some(camera_bgl), Some(atlas_bgl)],
        immediate_size: 0,
    });

    let sprite_targets = [Some(ColorTargetState {
        format,
        blend: Some(BlendState::ALPHA_BLENDING),
        write_mask: ColorWrites::ALL,
    })];

    let sprite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sprite_pipeline"),
        layout: Some(&camera_pipeline_layout),
        vertex: VertexState {
            module: &module,
            entry_point: Some("vs_sprite"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(quad_layout()), Some(sprite_layout())],
        },
        fragment: Some(FragmentState {
            module: &module,
            entry_point: Some("fs_sprite"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &sprite_targets,
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let glyph_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("glyph_pipeline"),
        layout: Some(&glyph_pipeline_layout),
        vertex: VertexState {
            module: &module,
            entry_point: Some("vs_glyph"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(quad_layout()), Some(glyph_layout())],
        },
        fragment: Some(FragmentState {
            module: &module,
            entry_point: Some("fs_glyph"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &sprite_targets,
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    (sprite_pipeline, glyph_pipeline)
}
