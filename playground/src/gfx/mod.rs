//! wgpu renderer for the playground: owns the device, the two instanced-quad
//! pipelines and either a window surface or an offscreen target.

use std::error::Error;
use std::sync::Arc;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    Color, CommandEncoderDescriptor, CurrentSurfaceTexture, Device, Extent3d, Features,
    Instance, InstanceDescriptor, Limits, LoadOp, MapMode, MemoryHints, Operations, Origin3d,
    PowerPreference, PresentMode, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RequestAdapterOptions, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
    StoreOp, Surface, SurfaceConfiguration, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
    TextureViewDimension, Trace,
};

use crate::atlas::FontAtlas;
use crate::color::Rgba;
use crate::draw::DrawList;

mod pipeline;

use pipeline::{create_pipelines, QUAD_CORNERS};

/// Initial capacity (instances) of the per-frame instance buffers; they grow
/// on demand.
const INITIAL_INSTANCE_CAPACITY: u64 = 4096;

/// Where the renderer draws: a window surface (presented uncapped, so the
/// FPS meter shows real throughput) or an offscreen texture used by
/// `--frames N` smoke runs and tests.
enum Target {
    Surface {
        surface: Surface<'static>,
        config: SurfaceConfiguration,
    },
    Offscreen {
        texture: Texture,
        view: TextureView,
        size: (u32, u32),
    },
}

/// Owns all GPU state. Reused across frames; one instance per app.
pub struct Gfx {
    device: Device,
    queue: Queue,
    camera_buf: Buffer,
    camera_bg: BindGroup,
    atlas_bg: BindGroup,
    sprite_pipeline: wgpu::RenderPipeline,
    glyph_pipeline: wgpu::RenderPipeline,
    quad_buf: Buffer,
    sprite_buf: Option<InstanceBuffer>,
    glyph_buf: Option<InstanceBuffer>,
    target: Target,
}

struct InstanceBuffer {
    buf: Buffer,
    cap_items: u64,
}

impl Gfx {
    /// Creates a renderer presenting to `window`'s surface at `size` physical
    /// pixels. `display` is the event loop's owned display handle, which
    /// wgpu 30 expects at instance creation when targeting windows.
    pub async fn new_windowed(
        window: &Arc<winit::window::Window>,
        display: winit::event_loop::OwnedDisplayHandle,
        size: (u32, u32),
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let instance =
            Instance::new(InstanceDescriptor::new_with_display_handle(Box::new(display)));
        let surface = instance.create_surface(window.clone())?;
        let (adapter, device, queue) = create_core(&instance, Some(&surface)).await?;
        let mut config = surface
            .get_default_config(&adapter, size.0.max(1), size.1.max(1))
            .ok_or("the window surface is not supported by the selected adapter")?;
        // Uncapped presentation: the FPS counter must reflect real frame
        // throughput, not the monitor refresh rate.
        config.present_mode = PresentMode::AutoNoVsync;
        surface.configure(&device, &config);
        Self::build(device, queue, Target::Surface { surface, config })
    }

    /// Creates a renderer drawing into an offscreen RGBA8 texture: no window,
    /// no surface. Used by `--frames N` smoke runs and headless tests.
    pub async fn new_headless(size: (u32, u32)) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let (_adapter, device, queue) = create_core(&instance, None).await?;
        let size = (size.0.max(1), size.1.max(1));
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("millarecs_playground_offscreen"),
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self::build(device, queue, Target::Offscreen { texture, view, size })
    }

    fn build(
        device: Device,
        queue: Queue,
        target: Target,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let format = match &target {
            Target::Surface { config, .. } => config.format,
            Target::Offscreen { .. } => TextureFormat::Rgba8Unorm,
        };

        let camera_buf = device.create_buffer(&BufferDescriptor {
            label: Some("camera_uniform"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bgl: BindGroupLayout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("camera_bgl"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let camera_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &camera_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let atlas = FontAtlas::with_font_chars();
        let (aw, ah) = atlas.size();
        let atlas_tex = device.create_texture(&TextureDescriptor {
            label: Some("font_atlas"),
            size: Extent3d {
                width: aw,
                height: ah,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &atlas_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            atlas.pixels(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aw),
                rows_per_image: None,
            },
            Extent3d {
                width: aw,
                height: ah,
                depth_or_array_layers: 1,
            },
        );
        let atlas_view = atlas_tex.create_view(&TextureViewDescriptor::default());
        let atlas_sampler: Sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("font_atlas_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let atlas_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("atlas_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let atlas_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("atlas_bg"),
            layout: &atlas_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        let quad_buf = create_quad_buffer(&device);

        let (sprite_pipeline, glyph_pipeline) =
            create_pipelines(&device, format, &camera_bgl, &atlas_bgl);

        Ok(Self {
            device,
            queue,
            camera_buf,
            camera_bg,
            atlas_bg,
            sprite_pipeline,
            glyph_pipeline,
            quad_buf,
            sprite_buf: None,
            glyph_buf: None,
            target,
        })
    }

    /// Current render target size in pixels.
    pub fn size(&self) -> (u32, u32) {
        match &self.target {
            Target::Surface { config, .. } => (config.width, config.height),
            Target::Offscreen { size, .. } => *size,
        }
    }

    /// Reconfigures the surface after a window resize; zero sizes are ignored
    /// (minimized windows).
    pub fn resize(&mut self, size: (u32, u32)) {
        if let Target::Surface { surface, config } = &mut self.target {
            if size.0 == 0 || size.1 == 0 {
                return;
            }
            config.width = size.0;
            config.height = size.1;
            surface.configure(&self.device, config);
        }
    }

    /// Records and submits one frame: clears to `clear`, draws the sprites,
    /// then the HUD glyphs. Transient surface-acquire failures skip the frame
    /// instead of panicking.
    pub fn render(&mut self, out: &DrawList, clear: Rgba) {
        let (w, h) = self.size();
        self.queue.write_buffer(
            &self.camera_buf,
            0,
            bytemuck::cast_slice(&[w as f32, h as f32, 0.0, 0.0]),
        );

        let sprite_ref = ensure_instance_buffer(
            &mut self.sprite_buf,
            &self.device,
            out.sprites.len(),
            std::mem::size_of::<crate::draw::SpriteInstance>() as u64,
            "sprite_instances",
        );
        if !out.sprites.is_empty() {
            self.queue
                .write_buffer(sprite_ref, 0, bytemuck::cast_slice(&out.sprites));
        }
        let glyph_ref = ensure_instance_buffer(
            &mut self.glyph_buf,
            &self.device,
            out.glyphs.len(),
            std::mem::size_of::<crate::draw::GlyphInstance>() as u64,
            "glyph_instances",
        );
        if !out.glyphs.is_empty() {
            self.queue
                .write_buffer(glyph_ref, 0, bytemuck::cast_slice(&out.glyphs));
        }

        // Acquire the frame (surfaces only); transient failures skip it.
        let mut surface_texture: Option<wgpu::SurfaceTexture> = None;
        match &mut self.target {
            Target::Surface { surface, config } => match surface.get_current_texture() {
                CurrentSurfaceTexture::Success(t) => surface_texture = Some(t),
                CurrentSurfaceTexture::Suboptimal(t) => {
                    surface.configure(&self.device, config);
                    surface_texture = Some(t);
                }
                CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return,
                _ => {
                    // Outdated/Lost/Validation: reconfigure, retry next frame.
                    surface.configure(&self.device, config);
                    return;
                }
            },
            Target::Offscreen { .. } => {}
        }
        let owned_view: Option<TextureView> = surface_texture
            .as_ref()
            .map(|t| t.texture.create_view(&TextureViewDescriptor::default()));

        let pass_view: &TextureView = match (&self.target, &owned_view) {
            (Target::Surface { .. }, Some(view)) => view,
            (Target::Offscreen { view, .. }, None) => view,
            _ => return,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("frame_encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("frame_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: pass_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            r: clear[0] as f64,
                            g: clear[1] as f64,
                            b: clear[2] as f64,
                            a: clear[3] as f64,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            if !out.sprites.is_empty() {
                pass.set_pipeline(&self.sprite_pipeline);
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_vertex_buffer(0, self.quad_buf.slice(..));
                pass.set_vertex_buffer(1, sprite_ref.slice(..));
                pass.draw(0..6, 0..out.sprites.len() as u32);
            }
            if !out.glyphs.is_empty() {
                pass.set_pipeline(&self.glyph_pipeline);
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_bind_group(1, &self.atlas_bg, &[]);
                pass.set_vertex_buffer(0, self.quad_buf.slice(..));
                pass.set_vertex_buffer(1, glyph_ref.slice(..));
                pass.draw(0..6, 0..out.glyphs.len() as u32);
            }
        }
        self.queue.submit([encoder.finish()]);

        drop(owned_view);
        if let Some(texture) = surface_texture {
            self.queue.present(texture);
        }
    }

    /// Reads back one pixel of the offscreen target as RGBA8 (windowed
    /// targets return `None`). Used by tests to assert frames are not empty.
    pub fn read_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        let Target::Offscreen { texture, size, .. } = &self.target else {
            return None;
        };
        let (w, h) = *size;
        if x >= w || y >= h {
            return None;
        }
        let bytes_per_row = ((w * 4) + 255) / 256 * 256;
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("readback_staging"),
            size: (bytes_per_row * h) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &staging,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        for _ in 0..128 {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            if rx.try_recv().is_ok() {
                break;
            }
        }

        let data = staging.slice(..).get_mapped_range().expect("map staging buffer for readback");
        let offset = (y * bytes_per_row + x * 4) as usize;
        let pixel = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
        drop(data);
        staging.unmap();
        Some(pixel)
    }
}

/// Requests adapter + device; the adapter doubles as the surface-capability
/// source for `get_default_config`.
async fn create_core(
    instance: &Instance,
    compatible_surface: Option<&Surface<'static>>,
) -> Result<(wgpu::Adapter, Device, Queue), Box<dyn Error + Send + Sync>> {
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface,
            apply_limit_buckets: false,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("no suitable GPU adapter: {e}"))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("millarecs_playground_device"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
        })
        .await
        .map_err(|e| format!("GPU device request failed: {e}"))?;
    Ok((adapter, device, queue))
}

/// Grows (or creates) an instance buffer so it fits `needed_items`, returning
/// the buffer to bind at instance step.
fn ensure_instance_buffer<'a>(
    slot: &'a mut Option<InstanceBuffer>,
    device: &Device,
    needed_items: usize,
    stride: u64,
    label: &'static str,
) -> &'a Buffer {
    let needed = needed_items as u64;
    let must_replace = match slot {
        Some(existing) => needed > existing.cap_items,
        None => true,
    };
    if must_replace {
        let cap = (needed.max(1) * 2).max(INITIAL_INSTANCE_CAPACITY);
        let buf = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: cap * stride,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *slot = Some(InstanceBuffer { buf, cap_items: cap });
    }
    &slot.as_ref().expect("instance buffer just created").buf
}

/// Uploads the static unit-quad corner data (two triangles over `[-1, 1]²`).
fn create_quad_buffer(device: &Device) -> Buffer {
    let mut data = Vec::with_capacity(QUAD_CORNERS.len() * 8);
    for corner in QUAD_CORNERS {
        data.extend_from_slice(bytemuck::bytes_of(&corner));
    }
    let buf = device.create_buffer(&BufferDescriptor {
        label: Some("unit_quad"),
        size: data.len() as u64,
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    let mut quad_view = buf.slice(..).get_mapped_range_mut().expect("map quad buffer");
    quad_view.copy_from_slice(&data);
    drop(quad_view);
    buf.unmap();
    buf
}

#[cfg(test)]
mod tests {
    use super::Gfx;
    use crate::color::rgb;
    use crate::draw::DrawList;
    use crate::math::Vec2;

    #[test]
    fn headless_render_reaches_the_target() {
        let mut gfx = pollster::block_on(Gfx::new_headless((64, 64))).expect("headless gfx");
        let mut out = DrawList::new();
        out.rect(Vec2::new(32.0, 32.0), Vec2::splat(20.0), rgb(1.0, 0.0, 0.0));
        gfx.render(&out, [0.0, 0.0, 0.0, 1.0]);

        let background = gfx.read_pixel(1, 1).expect("offscreen readback");
        assert!(
            background[0] < 8 && background[1] < 8 && background[2] < 8,
            "corner must stay at the clear color, got {background:?}"
        );
        let center = gfx.read_pixel(32, 32).expect("offscreen readback");
        assert!(
            center[0] > 200 && center[1] < 40 && center[2] < 40,
            "center must be the red rect, got {center:?}"
        );
    }
}
