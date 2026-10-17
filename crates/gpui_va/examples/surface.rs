//! One window that shows every way to composite pixels GPUI did not draw.
//!
//! ```sh
//! cargo run -p gpui_va --example surface
//! cargo run -p gpui_va --example surface -- --tab 2   # open on a given format tab
//! ```
//!
//! The window is a labelled grid of solid olive-green tiles. Each tile is one combination of an
//! **authoring layer** and a **payload**, and the grid is the point: the same colour reaches the
//! window whether it arrives through the `surface()` element or through a `gpu_canvas()` callback,
//! and whichever payload the host renderer can composite.
//!
//! # The two layers
//!
//! - `surface(source)` is an ordinary element. It lays out and stacks like any other child, and the
//!   renderer samples `source` while it composites the frame.
//! - `gpu_canvas(..)` is a box whose content a paint-time callback supplies: it receives a
//!   `GpuCanvasContext` and either paints a surface source into the canvas, or paints a texture the
//!   producer rendered on the window's own device.
//!
//! # The payloads, by renderer
//!
//! Only the combinations the host renderer supports are built and shown; each is labelled in place.
//!
//! - **wgpu** (Linux): `surface()` carries a dma-buf handle in each of `Bgra8`, `Rgba8` and `Nv12`;
//!   `gpu_canvas(..)` paints a `wgpu` texture view.
//! - **Direct3D 11** (Windows): `surface()` carries a `B8G8R8A8_UNORM` texture, and separately a
//!   shader-resource view the producer made; `gpu_canvas(..)` paints a surface source.
//! - **Metal** (macOS): `surface()` carries a CoreVideo pixel buffer; `gpu_canvas(..)` paints the same
//!   buffer and an `id<MTLTexture>`.
//!
//! # One colour, in each payload's own format
//!
//! Every tile is the same olive green. A single-plane payload stores it directly, in that format's
//! byte order; the two-plane `Nv12` payload stores the luma and chroma bytes the renderer's BT.601
//! conversion maps back to that green, and that conversion is worked out and commented where the
//! bytes are made.
//!
//! The producers are the platform's own: on Linux the renderer's own device — borrowed through
//! `gpui_interop` — exports the dma-bufs, and the window's device backs the textures. This is a
//! working application, not a sketch.

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
fn main() {
    eprintln!(
        "surface is a desktop example: it produces a GPU texture or dma-buf and composites it \
         through the window's renderer. Run it on Linux, Windows or macOS."
    );
}

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
fn main() {
    install_logger_if_requested();
    demo::run();
}

/// Prints the `log` records the libraries emit, which are otherwise dropped — most importantly the
/// renderer's reason for *dropping a dma-buf surface*, which it logs at `warn` and which would
/// otherwise leave a tile simply blank. Off unless `RUST_LOG` is set, so the example's own output is
/// unchanged; a value of `debug` or `trace` widens what is shown.
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
fn install_logger_if_requested() {
    struct Stderr;

    impl log::Log for Stderr {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::max_level()
        }

        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                eprintln!("[{}] {}", record.level(), record.args());
            }
        }

        fn flush(&self) {}
    }

    let level = match std::env::var("RUST_LOG").ok().as_deref() {
        None => return,
        Some(value) if value.contains("trace") => log::LevelFilter::Trace,
        Some(value) if value.contains("debug") => log::LevelFilter::Debug,
        Some(value) if value.contains("info") => log::LevelFilter::Info,
        Some(value) if value.contains("error") => log::LevelFilter::Error,
        Some(_) => log::LevelFilter::Warn,
    };
    let _ = log::set_boxed_logger(Box::new(Stderr));
    log::set_max_level(level);
}

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
mod demo {
    use gpui::{
        AnyElement, App, Bounds, Context, IntoElement, Render, Window, WindowBounds, WindowOptions,
        div, prelude::*, px, rgb, size,
    };

    /// The produced surface size, in pixels; each element scales it into its tile.
    const TILE: u32 = 128;

    /// The colour the `VA-API` tab falls back to: olive green, the demo's fixture, as RGBA. Where a
    /// real decode is available the tab shows the fixture's bars instead.
    const DECODED_COLOUR: [u8; 4] = [128, 128, 0, 255];

    /// The demo's payload: SMPTE 75% colour bars, in the byte layout each format carries.
    ///
    /// A card rather than a flat colour, because each bar is a known triple: a wrong channel order,
    /// a transfer function applied the wrong way, or a mis-set row stride shows up as the wrong bar,
    /// where a solid fill would look plausible. The card is the seven vertical bars of SMPTE RP 219
    /// at 75% amplitude — grey, yellow, cyan, green, magenta, red, blue.
    mod test_card {
        /// The bars, left to right, as sRGB-encoded `u8` RGB.
        const BARS: [[u8; 3]; 7] = [
            [191, 191, 191],
            [191, 191, 0],
            [0, 191, 191],
            [0, 191, 0],
            [191, 0, 191],
            [191, 0, 0],
            [0, 0, 191],
        ];

        /// The bar at column `x` of `width`, as an sRGB RGB triple.
        fn bar(x: u32, width: u32) -> [u8; 3] {
            BARS[(x * BARS.len() as u32 / width).min(BARS.len() as u32 - 1) as usize]
        }

        /// The card's pixel at `(x, y)` of a `width`-square image, as an sRGB RGB triple.
        ///
        /// The top three quarters are the colour bars. The bottom quarter is fine detail in two
        /// halves: left, one-pixel red/blue columns — colour detail a 4:2:0 chroma plane cannot
        /// carry, so it smears toward purple; right, one-pixel black/white columns — luma detail,
        /// which NV12 keeps. The pair shows what subsampling costs and what it does not.
        pub fn pixel(x: u32, y: u32, width: u32) -> [u8; 3] {
            if y < width * 3 / 4 {
                return bar(x, width);
            }
            if x < width / 2 {
                if x.is_multiple_of(2) {
                    [255, 0, 0]
                } else {
                    [0, 0, 255]
                }
            } else if x.is_multiple_of(2) {
                [0, 0, 0]
            } else {
                [255, 255, 255]
            }
        }

        /// The card as 4-byte `Bgra8` pixels, row-major.
        pub fn bgra8(width: u32) -> Vec<u8> {
            let mut bytes = Vec::with_capacity((width * width * 4) as usize);
            for y in 0..width {
                for x in 0..width {
                    let [r, g, b] = pixel(x, y, width);
                    bytes.extend_from_slice(&[b, g, r, 255]);
                }
            }
            bytes
        }

        /// The card as 4-byte `Rgba8` pixels, row-major.
        #[cfg(target_os = "linux")]
        pub fn rgba8(width: u32) -> Vec<u8> {
            let mut bytes = Vec::with_capacity((width * width * 4) as usize);
            for y in 0..width {
                for x in 0..width {
                    let [r, g, b] = pixel(x, y, width);
                    bytes.extend_from_slice(&[r, g, b, 255]);
                }
            }
            bytes
        }

        /// The card's full-resolution luma plane, `width`-square, full-range BT.601.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        pub fn luma(width: u32) -> Vec<u8> {
            let mut bytes = Vec::with_capacity((width * width) as usize);
            for y in 0..width {
                for x in 0..width {
                    bytes.push(super::nv12_from_rgb(to_rgba(pixel(x, y, width))).0);
                }
            }
            bytes
        }

        /// The card's half-resolution interleaved chroma plane, Cb then Cr per sample.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        pub fn chroma(width: u32) -> Vec<u8> {
            let mut bytes = Vec::with_capacity(((width / 2) * (width / 2) * 2) as usize);
            for row in 0..width / 2 {
                for x in 0..width / 2 {
                    // A chroma sample is the average of its 2x2 block, as an encoder's box filter
                    // makes it. Over the fine red/blue columns that average is the murky purple the
                    // subsampling bleeds — the point of the patch.
                    let (_, cb, cr) = super::nv12_from_rgb(block_average(x * 2, row * 2, width));
                    bytes.push(cb);
                    bytes.push(cr);
                }
            }
            bytes
        }

        /// The average of the 2x2 block at `(x, y)`, as the RGBA the conversion takes.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        fn block_average(x: u32, y: u32, width: u32) -> [u8; 4] {
            let mut sums = [0u32; 3];
            for dy in 0..2 {
                for dx in 0..2 {
                    let [r, g, b] = pixel((x + dx).min(width - 1), (y + dy).min(width - 1), width);
                    sums[0] += u32::from(r);
                    sums[1] += u32::from(g);
                    sums[2] += u32::from(b);
                }
            }
            [
                (sums[0] / 4) as u8,
                (sums[1] / 4) as u8,
                (sums[2] / 4) as u8,
                255,
            ]
        }

        /// Write the card's full-resolution luma plane into `dst`, starting at `offset`, `pitch`
        /// bytes per row.
        #[cfg(target_os = "linux")]
        pub fn write_luma(dst: &mut [u8], width: u32, offset: usize, pitch: usize) {
            let row = width as usize;
            for (index, line) in luma(width).chunks(row).enumerate() {
                dst[offset + index * pitch..][..row].copy_from_slice(line);
            }
        }

        /// Write the card's half-resolution interleaved chroma plane into `dst`, at `offset`, `pitch`
        /// bytes per row.
        #[cfg(target_os = "linux")]
        pub fn write_chroma(dst: &mut [u8], width: u32, offset: usize, pitch: usize) {
            let row = width as usize;
            for (index, line) in chroma(width).chunks(row).enumerate() {
                dst[offset + index * pitch..][..row].copy_from_slice(&line[..row]);
            }
        }

        /// An sRGB triple widened to the RGBA the conversion takes.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        fn to_rgba(rgb: [u8; 3]) -> [u8; 4] {
            [rgb[0], rgb[1], rgb[2], 255]
        }
    }

    #[cfg(target_os = "linux")]
    const BACKEND: &str =
        "Linux · wgpu — dma-buf Bgra8, Rgba8 and Nv12, and an imported texture view";
    #[cfg(target_os = "windows")]
    const BACKEND: &str =
        "Windows · Direct3D 11 — a texture payload, a view payload, and a canvas surface";
    #[cfg(target_os = "macos")]
    const BACKEND: &str = "macOS · Metal — a CoreVideo buffer and an imported Metal texture";

    pub fn run() {
        gpui::application().run(|cx: &mut App| {
            let bounds = Bounds::centered(None, size(px(780.), px(680.)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |_, cx| cx.new(|_| Showcase::new()),
            )
            .expect("a window to composite into");
            cx.activate(true);
        });
    }

    /// The window's view. The producers live here so their resources outlive every frame, and the
    /// selected tab names the format dimension the grid is showing.
    struct Showcase {
        tab: usize,
        #[cfg(target_os = "linux")]
        producer: Option<wgpu_backend::Producer>,
        #[cfg(target_os = "windows")]
        producer: Option<directx_backend::Producer>,
        #[cfg(target_os = "macos")]
        producer: Option<metal_backend::Producer>,
    }

    impl Showcase {
        fn new() -> Self {
            // `--tab N` opens on a given format tab, so one payload can be checked without a click.
            let mut tab = 0;
            let mut args = std::env::args().skip(1);
            while let Some(arg) = args.next() {
                if arg == "--tab" {
                    tab = args.next().and_then(|n| n.parse().ok()).unwrap_or(tab);
                }
            }
            log::info!("surface: opening on tab {tab}");
            Self {
                tab,
                producer: None,
            }
        }
    }

    impl Render for Showcase {
        fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            // A producer's dma-buf is importable only on the GPU the window draws through, and that
            // GPU is reachable only from a `gpu_canvas`. So paint one reporting canvas first and
            // build no producer until it has: a producer that picked its own device would export on
            // a different GPU and every tile would be dropped at import. One frame of empty tiles.
            #[cfg(target_os = "linux")]
            if !renderer_gpu_reported() {
                cx.notify();
                window.request_animation_frame();
                return gpui::gpu_canvas(report_renderer_gpu).size_full().into_any_element();
            }

            #[cfg(target_os = "linux")]
            let (formats, tiles) = (
                wgpu_backend::FORMATS,
                wgpu_backend::tiles(&mut self.producer, self.tab, window),
            );
            #[cfg(target_os = "windows")]
            let (formats, tiles) = (
                directx_backend::FORMATS,
                directx_backend::tiles(&mut self.producer, self.tab, window),
            );
            #[cfg(target_os = "macos")]
            let (formats, tiles) = (
                metal_backend::FORMATS,
                metal_backend::tiles(&mut self.producer, self.tab, window),
            );

            let selected = self.tab;
            let tabs = div().flex().flex_wrap().justify_center().gap_2().children(
                formats.iter().enumerate().map(|(index, name)| {
                    let active = index == selected;
                    div()
                        .id(("format-tab", index))
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .cursor_pointer()
                        .text_sm()
                        .bg(if active { rgb(0x3a3a44) } else { rgb(0x1c1c22) })
                        .text_color(if active { rgb(0xffffff) } else { rgb(0x9a9aa2) })
                        .child(*name)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.tab = index;
                            cx.notify();
                        }))
                }),
            );

            div()
                .size_full()
                .bg(rgb(0x101014))
                .flex()
                .flex_col()
                .items_center()
                .gap_6()
                .p_8()
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap_1()
                        .child(
                            div()
                                .text_2xl()
                                .text_color(rgb(0xffffff))
                                .child("External surfaces"),
                        )
                        .child(div().text_sm().text_color(rgb(0x9a9aa2)).child(BACKEND)),
                )
                .child(tabs)
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .justify_center()
                        .gap_6()
                        .children(tiles),
                )
                .into_any_element()
        }
    }

    /// Whether the window's renderer has reported the GPU a producer must export on. `None` until a
    /// `gpu_canvas` has painted, which is why the window paints one before it builds any producer.
    #[cfg(target_os = "linux")]
    fn renderer_gpu_reported() -> bool {
        gpui_interop::gpu_id().is_some()
    }

    std::thread_local! {
        /// The Vulkan device borrowed from the window's renderer, stashed by the first reporting
        /// canvas so a producer built off the paint exports its dma-bufs on the GPU the window draws
        /// through. A dma-buf is importable only on the GPU it was exported from, and the renderer's
        /// device is reachable only from a canvas, so this is where the two frames meet.
        static RENDERER_DEVICE: std::cell::RefCell<Option<gpui_interop::GpuDevice>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Report the renderer's GPU and stash its device, so the producer exports on the GPU the window
    /// draws through. A dma-buf is importable only on the GPU it was exported from, and that device is
    /// reachable only from the canvas, so this is the one place it can be learned.
    #[cfg(target_os = "linux")]
    fn report_renderer_gpu(gpu: &mut gpui::GpuCanvasContext) {
        let Some(interop) = gpui_interop::attach(gpu.try_device::<gpui_wgpu::WgpuRenderer>()).ok()
        else {
            return;
        };
        if let Some(id) = interop.adapter().gpu_id() {
            gpui_interop::remember(id);
        }
        if let Some(device) = interop.adapter().gpu_device() {
            RENDERER_DEVICE.with(|slot| *slot.borrow_mut() = Some(device));
        }
    }

    /// A square tile with a caption under it, so each combination is labelled the way it was made.
    fn panel(content: impl IntoElement, caption: impl IntoElement) -> AnyElement {
        div()
            .flex()
            .flex_col()
            .items_center()
            .gap_2()
            .w(px(220.))
            .child(div().w(px(170.)).h(px(170.)).child(content))
            .child(div().text_color(rgb(0xd8d8d8)).text_center().child(caption))
            .into_any_element()
    }

    /// A tile that stands in for one the host could not produce, so the grid still explains itself.
    fn error_panel(message: String) -> AnyElement {
        panel(
            div()
                .size_full()
                .bg(rgb(0x2a1518))
                .flex()
                .items_center()
                .justify_center()
                .p_3()
                .child(div().text_color(rgb(0xffaaaa)).text_center().child(message)),
            "unavailable on this host",
        )
    }

    /// The Y, Cb and Cr bytes the renderer's BT.601 surface conversion maps back to `rgb`.
    ///
    /// The surface fragment of a two-plane payload converts with this matrix:
    ///
    /// ```text
    /// R = y + 1.4020*cr - 0.7010
    /// G = y - 0.3441*cb - 0.7141*cr + 0.5291
    /// B = y + 1.7720*cb - 0.8860
    /// ```
    ///
    /// which is BT.601, full range — the same matrix the wgpu and Metal shaders carry. Inverting it
    /// (move the offsets to the source side, then apply the luma/chroma rows of the inverse) gives the
    /// bytes to store for a target colour.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_from_rgb(rgb: [u8; 4]) -> (u8, u8, u8) {
        let r = f32::from(rgb[0]) / 255.0;
        let g = f32::from(rgb[1]) / 255.0;
        let b = f32::from(rgb[2]) / 255.0;
        let (r, g, b) = (r + 0.7010, g - 0.5291, b + 0.8860);
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = -0.168736 * r - 0.331264 * g + 0.5 * b;
        let cr = 0.5 * r - 0.418688 * g - 0.081312 * b;
        (
            (y.clamp(0.0, 1.0) * 255.0).round() as u8,
            (cb.clamp(0.0, 1.0) * 255.0).round() as u8,
            (cr.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
    }

    /// Linux's producer: a bare Vulkan device exports the dma-bufs, and the window's own wgpu device
    /// backs the imported texture.
    #[cfg(target_os = "linux")]
    mod wgpu_backend {
        use std::cell::RefCell;
        use std::rc::Rc;
        use std::sync::Arc;

        use super::{DECODED_COLOUR, TILE, error_panel, panel, test_card};
        use anyhow::{Context as _, Result};
        use gpui::{
            AnyElement, ChromaReconstruction, Corners, DmaBufHandle, ImportedTextureExt as _,
            ImportedTextureHandle, SurfaceFormatKind, Window, gpu_canvas, prelude::*, surface,
        };
        use gpui_interop::{GpuDevice, PlaneLayout, SharedBuffer};

        /// The format dimension: one tab per payload the platform can carry, plus the same NV12
        /// buffer sampled both ways so the chroma reconstruction can be compared, and the one a
        /// hardware decoder hands over.
        pub const FORMATS: &[&str] = &[
            "Bgra8",
            "Rgba8",
            "Nv12 · bilinear",
            "Nv12 · luma-guided",
            "wgpu texture",
            "Nv12 · decoded (VA-API)",
        ];

        pub struct Producer {
            /// The shareable buffers, held so their memory outlives the handles made from them.
            _buffers: Vec<SharedBuffer>,
            /// `surface()` with a dma-buf in each of the three formats.
            bgra: DmaBufHandle,
            rgba: DmaBufHandle,
            nv12: DmaBufHandle,
            /// The same NV12 buffer, asking the renderer for luma-guided chroma.
            nv12_sharp: DmaBufHandle,
            /// The imported-texture tile. Built at paint time, when the window's device exists; the
            /// `Rc` is shared with the callback so the tile survives across frames instead of being
            /// rebuilt every paint.
            imported: Rc<RefCell<Option<ImportedTile>>>,
            /// The VA-API surface the decoded tab shows, and the dma-buf it exported. `None` where
            /// `libva`, its driver, or the GPU is missing; the surface is held so its memory
            /// outlives the handle.
            decoded: Option<DecodedSurface>,
            decoded_handle: Option<DmaBufHandle>,
        }

        /// The decoded tab's surface: a real hardware decode of the fixture's bitstream, or — where
        /// libavcodec of the declared ABI is not installed — the GPU-filled stand-in. The payload is
        /// never read: either type is held only so its memory outlives the exported dma-buf.
        #[allow(
            dead_code,
            reason = "held so the surface outlives the dma-buf it exported"
        )]
        enum DecodedSurface {
            Decoded(Box<gpui_va::Frame>),
            StandIn(gpui_va::Surface),
        }

        impl Producer {
            fn new() -> Result<Self> {
                super::RENDERER_DEVICE.with(|slot| {
                    let device = slot.borrow();
                    let device = device
                        .as_ref()
                        .context("the window's renderer has not reported its GPU yet")?;
                    Self::on(device)
                })
            }

            /// Build every payload on `device`, the renderer's own device.
            fn on(device: &GpuDevice) -> Result<Self> {
                let mut buffers = Vec::new();

                // A single plane per format, each holding the test card in its own byte order.
                let bgra_buffer = SharedBuffer::export_at(
                    device,
                    TILE,
                    TILE,
                    SurfaceFormatKind::bgra8(),
                    &[PlaneLayout {
                        offset: 0,
                        stride: TILE * 4,
                    }],
                    &test_card::bgra8(TILE),
                )
                .context("export the bgra8 card")?;
                let bgra = bgra_buffer.handle();
                buffers.push(bgra_buffer);

                let rgba_buffer = SharedBuffer::export_at(
                    device,
                    TILE,
                    TILE,
                    SurfaceFormatKind::rgba8(),
                    &[PlaneLayout {
                        offset: 0,
                        stride: TILE * 4,
                    }],
                    &test_card::rgba8(TILE),
                )
                .context("export the rgba8 card")?;
                let rgba = rgba_buffer.handle();
                buffers.push(rgba_buffer);

                // Two planes in one allocation: full-resolution luma, then half-resolution interleaved
                // chroma — at the offset **the producer declares**, the canonical DRM `NV12` layout
                // with chroma directly after the luma. The renderer honours the handle's layout rather
                // than its own, so this renders where a driver whose linear layout pads luma to 64 KiB
                // (NVIDIA) would otherwise read the padding as chroma — zero chroma, which converts to
                // green.
                let luma = (TILE * TILE) as usize;
                let chroma = (TILE / 2) as usize * (TILE / 2) as usize * 2;
                let mut content = vec![0u8; luma + chroma];
                test_card::write_luma(&mut content, TILE, 0, TILE as usize);
                test_card::write_chroma(&mut content, TILE, luma, TILE as usize);
                let planes = [
                    PlaneLayout {
                        offset: 0,
                        stride: TILE,
                    },
                    PlaneLayout {
                        offset: luma as u64,
                        stride: TILE,
                    },
                ];

                let nv12_buffer = SharedBuffer::export_at(
                    device,
                    TILE,
                    TILE,
                    SurfaceFormatKind::nv12(),
                    &planes,
                    &content,
                )
                .context("export the nv12 card")?;
                let nv12 = nv12_buffer.handle();
                buffers.push(nv12_buffer);

                // A second NV12 buffer over the same card, asking for luma-guided chroma: the same
                // bytes the renderer would otherwise sample bilinearly, reconstructed against the
                // sharp luma plane instead.
                let sharp_buffer = SharedBuffer::export_at(
                    device,
                    TILE,
                    TILE,
                    SurfaceFormatKind::nv12(),
                    &planes,
                    &content,
                )
                .context("export the luma-guided nv12 card")?;
                let nv12_sharp = sharp_buffer
                    .handle()
                    .with_chroma(ChromaReconstruction::LumaGuided);
                buffers.push(sharp_buffer);

                // The decoded tab: `gpui_va` decodes the fixture's H.264 keyframe on the GPU with
                // libavcodec, into the same `Y_TILED` surface a video player would, and exports it.
                // Where libavcodec is missing it falls back to a GPU-filled stand-in.
                let (decoded, decoded_handle) = match gpui_va::decode(gpui_va::FIXTURE) {
                    Some(frame) => {
                        let handle = frame.handle().clone();
                        (Some(DecodedSurface::Decoded(Box::new(frame))), Some(handle))
                    }
                    None => match gpui_va::nv12(TILE, TILE, DECODED_COLOUR) {
                        Some((surface, handle)) => {
                            (Some(DecodedSurface::StandIn(surface)), Some(handle))
                        }
                        None => (None, None),
                    },
                };

                Ok(Self {
                    _buffers: buffers,
                    bgra,
                    rgba,
                    nv12,
                    nv12_sharp,
                    imported: Rc::new(RefCell::new(None)),
                    decoded,
                    decoded_handle,
                })
            }
        }

        /// The imported-texture tile: a wgpu texture the window's own device draws through.
        struct ImportedTile {
            /// Held so the device outlives the view taken from the texture.
            _device: Arc<wgpu::Device>,
            _texture: wgpu::Texture,
            handle: ImportedTextureHandle,
        }

        impl ImportedTile {
            fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<Self> {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("surface_showcase"),
                    size: wgpu::Extent3d {
                        width: TILE,
                        height: TILE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // sRGB, matching the renderer's imported-texture path: the sampler decodes to
                    // linear and the fragment re-encodes, so these sRGB-encoded bytes come back
                    // unchanged rather than double-encoded.
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &test_card::bgra8(TILE),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(TILE * 4),
                        rows_per_image: Some(TILE),
                    },
                    wgpu::Extent3d {
                        width: TILE,
                        height: TILE,
                        depth_or_array_layers: 1,
                    },
                );
                let handle = texture
                    .create_view(&wgpu::TextureViewDescriptor::default())
                    .to_imported_handle()?;
                Ok(Self {
                    _device: device,
                    _texture: texture,
                    handle,
                })
            }
        }

        pub fn tiles(
            slot: &mut Option<Producer>,
            format: usize,
            _window: &mut Window,
        ) -> Vec<AnyElement> {
            if slot.is_none() {
                match Producer::new() {
                    Ok(producer) => *slot = Some(producer),
                    Err(error) => {
                        log::error!("surface: cannot produce a dma-buf: {error:#}");
                        return vec![error_panel(format!(
                            "dma-buf producer unavailable: {error:#}"
                        ))];
                    }
                }
            }
            let producer = slot.as_ref().expect("the producer was just built");

            match format {
                0..=3 => {
                    let (handle, surface_caption, canvas_caption) = match format {
                        0 => (
                            producer.bgra.clone(),
                            "surface() · SurfaceFormatKind::bgra8()",
                            "gpu_canvas(..) · SurfaceFormatKind::bgra8()",
                        ),
                        1 => (
                            producer.rgba.clone(),
                            "surface() · SurfaceFormatKind::rgba8()",
                            "gpu_canvas(..) · SurfaceFormatKind::rgba8()",
                        ),
                        2 => (
                            producer.nv12.clone(),
                            "surface() · Nv12 · bilinear chroma",
                            "gpu_canvas(..) · Nv12 · bilinear chroma",
                        ),
                        _ => (
                            producer.nv12_sharp.clone(),
                            "surface() · Nv12 · luma-guided chroma",
                            "gpu_canvas(..) · Nv12 · luma-guided chroma",
                        ),
                    };
                    let canvas = handle.clone();
                    vec![
                        panel(surface(handle).size_full(), surface_caption),
                        panel(
                            gpu_canvas(move |gpu| gpu.paint_surface(canvas)).size_full(),
                            canvas_caption,
                        ),
                    ]
                }
                4 => {
                    let imported = producer.imported.clone();
                    vec![panel(
                        gpu_canvas(move |gpu| {
                            let mut tile = imported.borrow_mut();
                            if tile.is_none() {
                                let Some((device, queue)) =
                                    gpu.try_device::<gpui_wgpu::WgpuRenderer>()
                                else {
                                    log::error!(
                                        "surface: the window's renderer lends no wgpu device"
                                    );
                                    return;
                                };
                                match ImportedTile::new(device, queue) {
                                    Ok(built) => *tile = Some(built),
                                    Err(error) => {
                                        log::error!(
                                            "surface: cannot produce the imported texture: {error:#}"
                                        );
                                        return;
                                    }
                                }
                            }
                            let handle = tile
                                .as_ref()
                                .expect("the tile was just built")
                                .handle
                                .clone();
                            gpu.paint_texture(handle, Corners::default(), 1.0, false);
                        })
                        .size_full(),
                        "gpu_canvas(..) · wgpu texture view",
                    )]
                }
                _ => {
                    // The decoded tab: a `Y_TILED` NV12 surface a VA-API decoder wrote, the shape a
                    // hardware decoder hands over. Its memory is the driver's, so there is no
                    // `test_card` to draw into it — a real decode fills it, or the stand-in does.
                    let Some(handle) = producer.decoded_handle.clone() else {
                        return vec![error_panel(
                            "no VA-API surface: libva, its driver, or the GPU is missing".into(),
                        )];
                    };
                    let source = match producer.decoded.as_ref() {
                        Some(DecodedSurface::Decoded(_)) => "H.264 keyframe decoded by VA-API",
                        _ => "GPU-filled stand-in (no libavcodec)",
                    };
                    let canvas = handle.clone();
                    vec![
                        panel(
                            surface(handle).size_full(),
                            format!("surface() · Nv12 · Y_TILED — {source}"),
                        ),
                        panel(
                            gpu_canvas(move |gpu| gpu.paint_surface(canvas)).size_full(),
                            format!("gpu_canvas(..) · Nv12 · Y_TILED — {source}"),
                        ),
                    ]
                }
            }
        }

    }

    /// Windows' producer: Direct3D 11 on the device the window's renderer lends — the route a
    /// hardware decoder or a Direct3D engine takes.
    #[cfg(target_os = "windows")]
    mod directx_backend {
        use std::cell::RefCell;
        use std::rc::Rc;

        use anyhow::{Context as _, Result};
        use gpui::{
            AnyElement, DirectXRenderer, DirectXSource, GpuCanvasContext, SharedDirectXFence,
            SharedDirectXSurface, SurfaceSource, Window, gpu_canvas, prelude::*, surface,
        };
        use windows::Win32::Foundation::{HANDLE, HMODULE};
        use windows::Win32::Graphics::Direct3D::{
            D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0, D3D_SRV_DIMENSION_TEXTURE2D,
        };
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_BIND_SHADER_RESOURCE, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_FENCE_FLAG_SHARED,
            D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_SDK_VERSION,
            D3D11_SHADER_RESOURCE_VIEW_DESC, D3D11_SHADER_RESOURCE_VIEW_DESC_0, D3D11_TEX2D_SRV,
            D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11CreateDevice, ID3D11Device,
            ID3D11Device5, ID3D11DeviceContext, ID3D11DeviceContext4, ID3D11Fence, ID3D11Resource,
            ID3D11ShaderResourceView, ID3D11Texture2D,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        use windows::Win32::Graphics::Dxgi::{
            DXGI_SHARED_RESOURCE_READ, DXGI_SHARED_RESOURCE_WRITE, IDXGIResource1,
        };
        use windows::core::{Interface as _, PCWSTR};

        use super::{TILE, error_panel, panel, test_card};

        /// The format dimension: one tab per payload this backend carries.
        pub const FORMATS: &[&str] = &["Texture", "View", "Shared"];

        /// `GENERIC_ALL`, the access a shared fence handle is created with.
        const GENERIC_ALL: u32 = 0x1000_0000;

        /// The two device-made payloads. They are built on the first paint, when the renderer's
        /// device exists, and shared with every canvas callback.
        struct Payloads {
            texture_variant: ID3D11Texture2D,
            view_variant: ID3D11ShaderResourceView,
        }

        pub struct Producer {
            payloads: Rc<RefCell<Option<Payloads>>>,
            /// The device-independent payload: a shared texture (and fence) made on the producer's
            /// *own* device at construction — no window involved.
            shared: Option<SharedProducer>,
        }

        impl Producer {
            fn new() -> Self {
                Self {
                    payloads: Rc::new(RefCell::new(None)),
                    shared: match SharedProducer::new() {
                        Ok(shared) => Some(shared),
                        Err(error) => {
                            log::error!(
                                "surface: cannot produce a shared Direct3D texture: {error:#}"
                            );
                            None
                        }
                    },
                }
            }
        }

        /// A shared Direct3D texture made on the producer's *own* device.
        ///
        /// This is the payload `surface()` exists for on Windows: the NT handle is
        /// device-independent, so the element carries it before the window's renderer has a device,
        /// and the renderer opens and views it at draw. The handle is same-adapter, so the producer
        /// device is created on the default adapter — the one the window renderer uses by default.
        struct SharedProducer {
            /// Held so the device that made the texture and the fence outlives them.
            _device: ID3D11Device,
            _texture: ID3D11Texture2D,
            _fence: ID3D11Fence,
            /// The texture's NT handle, for `DirectXSource::Shared`.
            handle: HANDLE,
            /// The fence's NT handle, and the value the renderer waits for before it samples.
            fence: HANDLE,
            value: u64,
        }

        impl SharedProducer {
            fn new() -> Result<Self> {
                let mut device = None;
                let mut context = None;
                unsafe {
                    D3D11CreateDevice(
                        None,
                        D3D_DRIVER_TYPE_HARDWARE,
                        HMODULE::default(),
                        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                        Some(&[D3D_FEATURE_LEVEL_11_0]),
                        D3D11_SDK_VERSION,
                        Some(&mut device),
                        None,
                        Some(&mut context),
                    )
                }
                .context("creating the shared producer's Direct3D 11 device")?;
                let device: ID3D11Device =
                    device.context("D3D11CreateDevice returned no device")?;
                let context: ID3D11DeviceContext =
                    context.context("D3D11CreateDevice returned no context")?;

                let texture: ID3D11Texture2D = unsafe {
                    let mut texture = None;
                    device
                        .CreateTexture2D(
                            &D3D11_TEXTURE2D_DESC {
                                Width: TILE,
                                Height: TILE,
                                MipLevels: 1,
                                ArraySize: 1,
                                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                                SampleDesc: DXGI_SAMPLE_DESC {
                                    Count: 1,
                                    Quality: 0,
                                },
                                Usage: D3D11_USAGE_DEFAULT,
                                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                                CPUAccessFlags: 0,
                                MiscFlags: D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 as u32,
                            },
                            None,
                            Some(&mut texture),
                        )
                        .context("creating the shared texture")?;
                    texture.context("CreateTexture2D returned no texture")?
                };

                let fence: ID3D11Fence = unsafe {
                    let device5: ID3D11Device5 =
                        device.cast().context("not a Direct3D 11.5 device")?;
                    let mut fence = None;
                    device5
                        .CreateFence(0, D3D11_FENCE_FLAG_SHARED, &mut fence)
                        .context("creating the shared fence")?;
                    fence.context("CreateFence returned no fence")?
                };

                // Upload the card on the producer's own context and signal the fence, so the
                // renderer's first sample is ordered behind the upload.
                upload_card(&device, &texture)?;
                unsafe {
                    let context4: ID3D11DeviceContext4 =
                        context.cast().context("not a Direct3D 11.4 context")?;
                    context4
                        .Signal(&fence, 1)
                        .context("signalling the shared fence")?;
                    context.Flush();
                }

                let resource: IDXGIResource1 = texture
                    .cast()
                    .context("the texture is not a DXGI resource")?;
                let handle = unsafe {
                    resource
                        .CreateSharedHandle(
                            None,
                            (DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE).0,
                            PCWSTR::null(),
                        )
                        .context("sharing the texture")?
                };
                let fence_handle = unsafe {
                    fence
                        .CreateSharedHandle(None, GENERIC_ALL, PCWSTR::null())
                        .context("sharing the fence")?
                };

                Ok(Self {
                    _device: device,
                    _texture: texture,
                    _fence: fence,
                    handle,
                    fence: fence_handle,
                    value: 1,
                })
            }
        }

        /// Build the payloads on the window renderer's own device.
        ///
        /// The device is reachable only at paint, so this runs inside a `gpu_canvas` callback: a
        /// `surface()` element takes its source before painting, which a same-device source cannot
        /// satisfy.
        fn build(gpu: &mut GpuCanvasContext) -> Result<Payloads> {
            let device = gpu
                .try_device::<DirectXRenderer>()
                .context("the window's renderer is not Direct3D")?;

            let texture_variant = card_texture(&device)?;
            let view_texture = card_texture(&device)?;
            let view_variant = shader_resource_view(&device, &view_texture)?;

            Ok(Payloads {
                texture_variant,
                view_variant,
            })
        }

        /// Build the shared payloads (once) and paint one of them through `source`.
        fn paint(
            payloads: &Rc<RefCell<Option<Payloads>>>,
            gpu: &mut GpuCanvasContext,
            source: impl FnOnce(&Payloads) -> SurfaceSource,
        ) {
            let mut built = payloads.borrow_mut();
            if built.is_none() {
                match build(gpu) {
                    Ok(payloads) => *built = Some(payloads),
                    Err(error) => {
                        log::error!("surface: cannot produce a Direct3D texture: {error:#}");
                        return;
                    }
                }
            }
            let built = built.as_ref().expect("the payloads were just built");
            gpu.paint_surface(source(built));
        }

        pub fn tiles(
            slot: &mut Option<Producer>,
            format: usize,
            _window: &mut Window,
        ) -> Vec<AnyElement> {
            let producer = slot.get_or_insert_with(Producer::new);
            let payloads = producer.payloads.clone();
            let shared = producer.shared.as_ref().map(|shared| SharedDirectXSurface {
                texture: shared.handle,
                fence: Some(SharedDirectXFence {
                    handle: shared.fence,
                    value: shared.value,
                }),
                width: TILE,
                height: TILE,
            });

            match format {
                0 => vec![panel(
                    gpu_canvas({
                        let payloads = payloads.clone();
                        move |gpu| {
                            paint(&payloads, gpu, |payloads| {
                                SurfaceSource::DirectX(DirectXSource::Texture(
                                    payloads.texture_variant.clone(),
                                ))
                            });
                        }
                    })
                    .size_full(),
                    "gpu_canvas(..) · DirectXSource::Texture",
                )],
                1 => vec![panel(
                    gpu_canvas({
                        let payloads = payloads.clone();
                        move |gpu| {
                            paint(&payloads, gpu, |payloads| {
                                SurfaceSource::DirectX(DirectXSource::View(
                                    payloads.view_variant.clone(),
                                ))
                            });
                        }
                    })
                    .size_full(),
                    "gpu_canvas(..) · DirectXSource::View",
                )],
                _ => match shared {
                    Some(source) => {
                        let canvas_source = source.clone();
                        vec![
                            panel(
                                surface(SurfaceSource::DirectX(DirectXSource::Shared(source)))
                                    .size_full(),
                                "surface() · DirectXSource::Shared",
                            ),
                            panel(
                                gpu_canvas(move |gpu| {
                                    gpu.paint_surface(SurfaceSource::DirectX(
                                        DirectXSource::Shared(canvas_source),
                                    ));
                                })
                                .size_full(),
                                "gpu_canvas(..) · DirectXSource::Shared",
                            ),
                        ]
                    }
                    None => vec![error_panel("no shared Direct3D producer".to_string())],
                },
            }
        }

        /// An offscreen texture on `device`, holding the test card.
        ///
        /// `SHADER_RESOURCE` because both layers sample it. The format is the renderer's own target
        /// format, `B8G8R8A8_UNORM`: the renderer views the texture as non-sRGB and the fragment
        /// samples its bytes straight through.
        fn card_texture(device: &ID3D11Device) -> Result<ID3D11Texture2D> {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: TILE,
                Height: TILE,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let mut texture = None;
            unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture))? };
            let texture = texture.context("CreateTexture2D returned no texture")?;
            upload_card(device, &texture)?;
            Ok(texture)
        }

        /// Writes the test card into `texture` through the device's immediate context.
        ///
        /// A `DEFAULT` texture cannot be mapped, so the card is uploaded with `UpdateSubresource`,
        /// which writes a whole subresource: one row pitch, no depth pitch, no destination box.
        fn upload_card(device: &ID3D11Device, texture: &ID3D11Texture2D) -> Result<()> {
            let bytes = test_card::bgra8(TILE);
            let context = unsafe { device.GetImmediateContext() }
                .context("the device has no immediate context")?;
            let resource: ID3D11Resource = texture
                .cast()
                .context("the texture is not a Direct3D resource")?;
            unsafe {
                context.UpdateSubresource(
                    &resource,
                    0,
                    None,
                    bytes.as_ptr() as *const core::ffi::c_void,
                    TILE * 4,
                    0,
                );
            }
            Ok(())
        }

        /// A non-sRGB `B8G8R8A8_UNORM` view of `texture`, the payload of a view tile.
        fn shader_resource_view(
            device: &ID3D11Device,
            texture: &ID3D11Texture2D,
        ) -> Result<ID3D11ShaderResourceView> {
            unsafe {
                let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    ViewDimension: D3D_SRV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D11_TEX2D_SRV {
                            MostDetailedMip: 0,
                            MipLevels: 1,
                        },
                    },
                };
                let mut view = None;
                device.CreateShaderResourceView(texture, Some(&desc), Some(&mut view))?;
                view.context("CreateShaderResourceView returned no view")
            }
        }
    }

    /// macOS' producer: a CoreVideo buffer for two tiles, and a wgpu texture resolved to its
    /// `id<MTLTexture>` for the third.
    #[cfg(target_os = "macos")]
    mod metal_backend {
        use std::sync::Arc;

        use anyhow::{Context as _, Result, ensure};
        use core_foundation::base::{CFType, TCFType as _};
        use core_foundation::boolean::CFBoolean;
        use core_foundation::dictionary::CFDictionary;
        use core_foundation::string::CFString;
        use core_video::pixel_buffer::{
            CVPixelBuffer, kCVPixelBufferIOSurfacePropertiesKey,
            kCVPixelBufferMetalCompatibilityKey, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        };
        use gpui::{
            AnyElement, Corners, ImportedTextureHandle, SurfaceSource, Window, gpu_canvas,
            prelude::*, surface,
        };

        use super::{TILE, error_panel, panel, test_card};

        /// The format dimension: one tab per payload this backend carries.
        pub const FORMATS: &[&str] = &["CoreVideo", "Metal texture"];

        pub struct Producer {
            /// The CoreVideo buffer, for `surface()` and `gpu_canvas(..)`.
            buffer: CVPixelBuffer,
            /// Held so the device outlives the view taken from the texture.
            _device: Arc<wgpu::Device>,
            _texture: wgpu::Texture,
            /// The imported-texture tile's handle.
            imported: ImportedTextureHandle,
        }

        impl Producer {
            fn new() -> Result<Self> {
                // A decoder's frame: an IOSurface-backed, two-plane NV12 buffer at full range,
                // holding the test card made from the renderer's own BT.601 conversion.
                let buffer = new_nv12_buffer(TILE, TILE)?;
                fill_card(&buffer)?;

                // The imported-texture tile: a wgpu texture on the one Metal device macOS hands every
                // creator, so it is the device the renderer samples on.
                let (device, queue) = device()?;
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("surface_showcase"),
                    size: wgpu::Extent3d {
                        width: TILE,
                        height: TILE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // sRGB, the format the imported-texture path requires: the sampler decodes to
                    // linear and the fragment re-encodes, so the card's bytes come back unchanged.
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &test_card::bgra8(TILE),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(TILE * 4),
                        rows_per_image: Some(TILE),
                    },
                    wgpu::Extent3d {
                        width: TILE,
                        height: TILE,
                        depth_or_array_layers: 1,
                    },
                );
                let imported = metal_handle(&texture)?;

                Ok(Self {
                    buffer,
                    _device: device,
                    _texture: texture,
                    imported,
                })
            }
        }

        pub fn tiles(
            slot: &mut Option<Producer>,
            format: usize,
            _window: &mut Window,
        ) -> Vec<AnyElement> {
            if slot.is_none() {
                match Producer::new() {
                    Ok(producer) => *slot = Some(producer),
                    Err(error) => {
                        log::error!("surface: cannot produce the macOS payloads: {error:#}");
                        return vec![error_panel(format!("producer unavailable: {error:#}"))];
                    }
                }
            }
            let producer = slot.as_ref().expect("the producer was just built");

            match format {
                0 => {
                    let surface_buffer = producer.buffer.clone();
                    let canvas_buffer = producer.buffer.clone();
                    vec![
                        panel(
                            surface(SurfaceSource::CoreVideo(surface_buffer)).size_full(),
                            "surface() · CoreVideo CVPixelBuffer",
                        ),
                        panel(
                            gpu_canvas(move |gpu| {
                                gpu.paint_surface(SurfaceSource::CoreVideo(canvas_buffer));
                            })
                            .size_full(),
                            "gpu_canvas(..) · CoreVideo",
                        ),
                    ]
                }
                _ => {
                    let imported = producer.imported.clone();
                    vec![panel(
                        gpu_canvas(move |gpu| {
                            gpu.paint_texture(imported, Corners::default(), 1.0, false);
                        })
                        .size_full(),
                        "gpu_canvas(..) · id<MTLTexture>",
                    )]
                }
            }
        }

        /// An `IOSurface`-backed, two-plane `NV12` full-range buffer — the layout a hardware decoder
        /// produces and the renderer expects.
        fn new_nv12_buffer(width: u32, height: u32) -> Result<CVPixelBuffer> {
            // A `CVPixelBuffer` is IOSurface-backed only when its attributes ask for it; the renderer's
            // texture cache needs the IOSurface behind it. Metal compatibility is declared alongside.
            let iosurface_properties: CFDictionary<CFString, CFType> =
                CFDictionary::from_CFType_pairs(&[]);
            let iosurface_key: CFString =
                unsafe { CFString::wrap_under_get_rule(kCVPixelBufferIOSurfacePropertiesKey) };
            let metal_key: CFString =
                unsafe { CFString::wrap_under_get_rule(kCVPixelBufferMetalCompatibilityKey) };
            let attributes = CFDictionary::from_CFType_pairs(&[
                (iosurface_key, iosurface_properties.as_CFType()),
                (metal_key, CFBoolean::true_value().as_CFType()),
            ]);

            CVPixelBuffer::new(
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                width as usize,
                height as usize,
                Some(&attributes),
            )
            .map_err(|code| anyhow::anyhow!("CVPixelBuffer::new returned CVReturn {code}"))
        }

        /// Fill both planes with the test card, padding included, so no stale bytes are sampled.
        fn fill_card(buffer: &CVPixelBuffer) -> Result<()> {
            let width = buffer.get_width() as u32;
            let height = buffer.get_height();
            let luma = test_card::luma(width);
            let chroma = test_card::chroma(width);

            // 0 is the read-write lock: the CPU fills the planes, the GPU samples them.
            let lock = 0u64;
            let result = buffer.lock_base_address(lock);
            ensure!(
                result == 0,
                "CVPixelBufferLockBaseAddress returned CVReturn {result}"
            );

            // Safety: the buffer is locked, so both planes' base addresses are valid for the plane's
            // height and stride, and the source rows are `width` bytes of the card.
            unsafe {
                let luma_base = buffer.get_base_address_of_plane(0) as *mut u8;
                let luma_stride = buffer.get_bytes_per_row_of_plane(0);
                let chroma_base = buffer.get_base_address_of_plane(1) as *mut u8;
                let chroma_stride = buffer.get_bytes_per_row_of_plane(1);
                let chroma_height = buffer.get_height_of_plane(1);

                for row in 0..height {
                    let source = &luma[row * width as usize..(row + 1) * width as usize];
                    std::ptr::copy_nonoverlapping(
                        source.as_ptr(),
                        luma_base.add(row * luma_stride),
                        width as usize,
                    );
                }
                // Cb and Cr are interleaved, two bytes a sample, so a row of `width/2` samples is
                // `width` bytes.
                for row in 0..chroma_height {
                    let source = &chroma[row * width as usize..(row + 1) * width as usize];
                    std::ptr::copy_nonoverlapping(
                        source.as_ptr(),
                        chroma_base.add(row * chroma_stride),
                        width as usize,
                    );
                }
            }

            let result = buffer.unlock_base_address(lock);
            ensure!(
                result == 0,
                "CVPixelBufferUnlockBaseAddress returned CVReturn {result}"
            );
            Ok(())
        }

        /// wgpu's own Metal device, which is the one the window's Metal renderer also draws through.
        fn device() -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::METAL,
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .map_err(|error| anyhow::anyhow!("no Metal adapter: {error}"))?;
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("surface_showcase"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                }))
                .map_err(|error| anyhow::anyhow!("no wgpu device: {error}"))?;
            Ok((Arc::new(device), Arc::new(queue)))
        }

        /// The texture's own `id<MTLTexture>`, the handle the renderer resolves.
        fn metal_handle(texture: &wgpu::Texture) -> Result<ImportedTextureHandle> {
            use foreign_types::ForeignTypeRef as _;
            use gpui::MetalTextureExt as _;

            unsafe {
                let hal = texture
                    .as_hal::<wgpu::hal::api::Metal>()
                    .context("the texture has no Metal handle")?;
                let texture = metal::TextureRef::from_ptr(hal.raw_handle() as *const _ as *mut _);
                texture.to_imported_handle()
            }
        }
    }
}
