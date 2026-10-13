//! A wall of independently decoded videos, to show what zero-copy presentation buys over a CPU
//! readback.
//!
//! Each pane is its own decoder and its own frame — not one frame drawn everywhere — and by default
//! the panes are the *same* video started at different points, so the wall is a spread of stages. Flip
//! the handover on screen (click the mode chip) and watch the frame rate and CPU in the header.
//!
//! Each pane is one element, sized from a fixed cell computed once, so the grid fills the window with
//! no flexible layout and nothing to measure per pane.
//!
//! ```sh
//! cargo run -p gpui_video --example grid                              # 16x16, the bundled clip, zero-copy
//! cargo run -p gpui_video --example grid -- --readback               # start in readback mode
//! cargo run -p gpui_video --example grid -- --cols 4 --rows 4         # a 4x4 wall
//! cargo run -p gpui_video --example grid -- --size 1920x1080          # a fixed window the grid fills
//! cargo run -p gpui_video --example grid -- movie.mp4                 # stages of `movie.mp4`
//! cargo run -p gpui_video --example grid -- a.mp4 b.mp4               # mix sources across the panes
//! ```
//!
//! The header's "panes" figures are the *worker* side: decode plus the handover, summed across panes,
//! as a fraction of a core. The readback path additionally uploads each frame on the UI thread, which
//! those figures do not count — the render fps (from the instrumented pipeline) is where an upload
//! cost shows up. The header also shows the process's RSS and thread count, so a leak shows as RSS
//! that will not settle. `GPUI_GRID_AUTOFLIP=<secs>` flips the handover every `<secs>` seconds, to
//! exercise the round-trip without clicking.
//!
//! `GPUI_GRID_PACE=keep|drop` (default `drop`) sets how a pane keeps up when the display is slower
//! than its stream. `drop` holds the stream's clock — a pane whole periods late takes that many
//! frames at once, shows the last and drops the rest — which is what live playback wants; `keep`
//! shows every frame however slowly, which is what an offscreen consumer that sets the rate wants.
//! The readout's `dropped/s` counts what a pane could not show.
//!
//! `GPUI_GRID_SCALE=fit|native` (default `fit`) sets how a readback frame is converted for its
//! pane. `fit` converts straight to the pane's size — swscale scales as it converts — so the UI
//! thread uploads only the pixels the pane shows, which is where readback costs the most; `native`
//! converts at the stream's size and lets the GPU scale it when it draws. Only the readback path
//! can downscale: a dma-buf is the decoder's own surface, which the GPU already scales on sample.
//!
//! A wall of hundreds presents through a per-render budget: at most `GPUI_GRID_BUDGET` panes (default
//! [`BUDGET`]) advance in one frame, taking turns, so the UI thread's per-render layout and upload
//! stay bounded however fast the workers are, and the panes play at a shared rate rather than the
//! wall dropping to a few frames a second. `GPUI_GRID_BUDGET=0` lifts the cap.
//!
//! The figures are a translucent panel over the wall: the render rate, the per-pane rate, and the
//! process's RAM and VRAM and thread count. VRAM is the kernel's own per-client DRM accounting, read
//! from `fdinfo`, so it sees what the renderer and the video engine have reserved without reaching
//! into either; on an integrated part it is system memory the GPU has reserved, not a separate pool.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use gpui::{
    AnyElement, App, Bounds, Context, FramePipelineExt, ImageSource, IntoElement, ObjectFit,
    ParentElement, PhaseMetrics, Render, RenderImage, StatefulInteractiveElement,
    StandardImmediatePipeline, Styled, SurfaceSource, Window, WindowBounds, WindowOptions, div,
    img, prelude::*, px, rgb, rgba, size, surface,
};
use gpui_video_platform::SurfaceProducer;

use gpui_va::{Demuxer, Playback};

/// How many frames a zero-copy pane may keep alive at once.
const DEPTH: usize = 3;
/// Gap between panes and page padding.
const GAP: f32 = 4.0;
const PAD: f32 = 8.0;
/// The window's size when `--size` is not given. The grid fills it, its cells share it equally, and
/// each frame is scaled proportionally (letterboxed) into its cell.
const WINDOW_W: f32 = 1280.0;
const WINDOW_H: f32 = 800.0;
/// A pane's assumed period until its stream declares its own.
const DEFAULT_PERIOD: Duration = Duration::from_millis(40);
/// How long a pane waits for a frame it asked for before giving up on the reply. Well above any real
/// frame's latency, so it only fires for a request whose answer is never coming.
const ADVANCE_TIMEOUT: Duration = Duration::from_millis(500);
/// The most frames a pane asks for at once to catch up under [`Pace::Drop`], bounding the burst.
const MAX_CATCH_UP: u32 = 4;
/// The most panes that may present in one render unless `GPUI_GRID_BUDGET` overrides it. Bounding the
/// per-render work keeps a wall of hundreds responsive: the UI thread lays the grid out and uploads
/// the frames that changed, so with no cap every pane changes every frame as soon as the workers can
/// keep up, and the render rate collapses under its own upload weight. The panes then take turns, so
/// they share the render rate instead of starving each other.
const BUDGET: usize = 64;

/// Which handover the panes use.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Hand the renderer the decoded surface as a dma-buf to import.
    Surface,
    /// Copy the decoded pixels to the CPU, convert, and hand over a packed image.
    Readback,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Surface => "zero-copy (dma-buf) — click to flip",
            Mode::Readback => "readback (CPU) — click to flip",
        }
    }

    fn accent(self) -> gpui::Rgba {
        match self {
            Mode::Surface => rgb(0x2f7d4f),
            Mode::Readback => rgb(0x8a3b3b),
        }
    }
}

/// How a pane keeps up when the display cannot take frames as fast as its stream makes them.
#[derive(Clone, Copy, PartialEq)]
enum Pace {
    /// Show every frame, however slowly: the pane takes one frame per render and never skips a
    /// period, so nothing is dropped and a slow wall plays in slow motion. What an offscreen
    /// consumer that drives the rate itself wants.
    Keep,
    /// Keep the stream's clock: when the pane falls whole periods behind, it asks for that many
    /// frames at once, shows the last and drops the rest, so it stays in real time. What live
    /// playback wants.
    Drop,
}

impl Pace {
    fn label(self) -> &'static str {
        match self {
            Pace::Keep => "keep every frame",
            Pace::Drop => "drop to keep time",
        }
    }
}

/// The pacing policy: `GPUI_GRID_PACE=keep|drop`, default `drop`.
fn pane_pace() -> Pace {
    match std::env::var("GPUI_GRID_PACE").as_deref() {
        Ok("keep") => Pace::Keep,
        _ => Pace::Drop,
    }
}

/// How a readback frame is converted for the pane it will be drawn in.
#[derive(Clone, Copy, PartialEq)]
enum Scale {
    /// Convert it straight to the pane's size, so the UI thread uploads only the pixels the pane
    /// shows. Only the readback path can do this; a dma-buf is the decoder's own surface.
    Fit,
    /// Convert it at the stream's own size and let the GPU scale it when it draws.
    Native,
}

impl Scale {
    fn label(self) -> &'static str {
        match self {
            Scale::Fit => "fit to pane",
            Scale::Native => "native size",
        }
    }
}

/// `GPUI_GRID_SCALE=fit|native`, default `fit`.
fn pane_scale() -> Scale {
    match std::env::var("GPUI_GRID_SCALE").as_deref() {
        Ok("native") => Scale::Native,
        _ => Scale::Fit,
    }
}

/// Where a frame of `source` pixels fits into a `box_size` box, keeping its aspect and never
/// upscaling — the GPU's bilinear upscale is cheaper than a CPU one, so there is nothing to gain.
fn contain_size(source: (u32, u32), box_size: (f32, f32)) -> (u32, u32) {
    let (source_w, source_h) = (source.0.max(1) as f32, source.1.max(1) as f32);
    let scale = (box_size.0 / source_w).min(box_size.1 / source_h).min(1.0);
    (
        (source_w * scale).round().max(1.0) as u32,
        (source_h * scale).round().max(1.0) as u32,
    )
}

/// What a pane's worker tells the UI.
enum ToUi {
    /// The pane opened, with its frame period.
    Opened(Duration),
    /// A frame to present.
    Frame(Content),
    /// This turn produced no frame — the source had none ready. It answers the request anyway, so a
    /// pane is never left believing a frame is in flight that will never come.
    Idle,
}

/// What the UI asks a pane's worker for.
enum ToWorker {
    Advance,
}

/// A frame, as a pane will present it.
#[derive(Clone)]
enum Content {
    Surface(SurfaceSource),
    Image(Arc<RenderImage>),
}

/// One pane: the worker it drives, the frame it last got, and the pacing.
struct Pane {
    worker: flume::Sender<ToWorker>,
    content: Option<Content>,
    /// Whether the current content has been painted. A frame displaced before it was painted was
    /// never shown, so it is dropped — released and counted — rather than left in the dark.
    painted: bool,
    /// Frames this pane did not present: periods it was too slow to take, and frames displaced
    /// before they reached the screen.
    dropped: u64,
    /// How many frames the pane has asked for and not yet had answered — more than one only while
    /// catching up under [`Pace::Drop`], where the extra frames are dropped as they arrive.
    outstanding: u32,
    /// When the outstanding requests were sent, so a reply that never comes can be given up on.
    pending_since: Option<Instant>,
    due: Instant,
    period: Duration,
}

impl Pane {
    /// Ask for the next frame if one is due, returning whether one was asked for — so a caller can
    /// spend a per-render budget of asks.
    fn tick(&mut self, pace: Pace) -> bool {
        let now = Instant::now();
        if self.outstanding > 0 {
            // A reply that never arrives — a worker that died, a message lost across a mode switch —
            // would freeze the pane for good. After long enough, give up and ask again.
            if !self
                .pending_since
                .is_some_and(|at| now.duration_since(at) >= ADVANCE_TIMEOUT)
            {
                return false;
            }
            self.outstanding = 0;
            self.pending_since = None;
        }
        if now < self.due {
            return false;
        }
        // Dropping keeps the stream's clock: a pane whole periods late asks for that many frames at
        // once, shows the last and drops the rest. Keeping asks for one, so every frame is shown
        // however slowly the display takes them.
        let mut wanted = 1;
        if pace == Pace::Drop {
            let late = now.duration_since(self.due).as_nanos();
            let periods = (late / self.period.as_nanos().max(1)) as u32;
            wanted = (1 + periods).min(MAX_CATCH_UP);
        }
        for _ in 0..wanted {
            if self.worker.send(ToWorker::Advance).is_err() {
                return false;
            }
        }
        self.outstanding = wanted;
        self.pending_since = Some(now);
        // Move the deadline on by the periods just asked for, so the pane keeps the stream's rate.
        self.due += self.period * wanted;
        if pace == Pace::Drop && now.duration_since(self.due) >= self.period {
            // Still behind even after catching up to the cap: resync, and count what could not be
            // taken.
            self.dropped += 1;
            self.due = now;
        }
        true
    }
}

/// The panes' cost, summed: how many frames the workers handed over, and what their work cost.
#[derive(Default)]
struct Metrics {
    frames: AtomicU64,
    /// Wall time per frame, summed: the worker's latency, waiting on the hardware included.
    work_ns: AtomicU64,
    /// CPU time per frame, summed: the worker's real cost, waiting excluded.
    cpu_ns: AtomicU64,
}

impl Metrics {
    fn record(&self, work: Duration, cpu: Duration) {
        self.frames.fetch_add(1, Ordering::Relaxed);
        self.work_ns
            .fetch_add(work.as_nanos() as u64, Ordering::Relaxed);
        self.cpu_ns
            .fetch_add(cpu.as_nanos() as u64, Ordering::Relaxed);
    }
}

struct Grid {
    panes: Vec<Pane>,
    /// Bumped whenever the panes are rebuilt, so a reply from a superseded pane's worker is ignored
    /// rather than delivered to the pane that reused its index.
    generation: u64,
    sources: Vec<PathBuf>,
    cols: usize,
    rows: usize,
    /// The fixed pane size, computed once from the window so the grid fills it and nothing is
    /// measured per pane.
    cell: (f32, f32),
    /// The window's scale factor, so a readback's target is computed in device pixels.
    scale: f32,
    mode: Mode,
    /// How readback frames are converted for their pane.
    scale_mode: Scale,
    /// The pane workers' cost.
    metrics: Arc<Metrics>,
    /// The frame pipeline's own timing, for the render rate.
    phases: Rc<RefCell<PhaseMetrics>>,
    /// How many drain tasks are still live, so a leak shows as a count that will not settle.
    live_tasks: Arc<AtomicUsize>,
    /// The last worker reading, for a per-second delta.
    snapshot: (u64, u64, u64, Instant),
    /// The panes' dropped-frame total at the last reading, for a per-second delta.
    dropped_snapshot: u64,
    /// The last render-frame count and pass times, for a per-second delta.
    phases_snapshot: (PhaseMetrics, Instant),
    /// What the header shows.
    readout: Readout,
    /// Readback images whose panes have moved on, kept until the next render releases their atlas
    /// tiles. A readback frame is a fresh image with a fresh id, and only `Window::drop_image`
    /// releases the tile it was uploaded to — retain them and the atlas grows without bound until
    /// the renderer grinds to a stop.
    replaced: Vec<Arc<RenderImage>>,
    /// The most panes that may present in one render; `0` lifts the cap.
    budget: usize,
    /// Where the last render's budget stopped asking, so the panes take turns.
    cursor: usize,
    /// How panes keep up when the display cannot take frames as fast as their streams.
    pace: Pace,
    /// TEMP: seconds after start to auto-flip to readback (then back at twice that).
    autoflip: Option<f64>,
    started: Instant,
    flips: u32,
}

/// The header's figures, kept between renders.
#[derive(Default)]
struct Readout {
    /// Frames the pipeline drove a second (the whole window's redraw rate).
    render_fps: f64,
    /// The frame's root passes, averaged over the last second, in milliseconds.
    render_eval_ms: f64,
    render_layout_ms: f64,
    render_paint_ms: f64,
    /// Frames the pane workers handed over a second, all panes together.
    pane_fps: f64,
    /// The workers' work per frame, in milliseconds.
    ms_per_frame: f64,
    /// The workers' CPU a second, as a fraction of one core.
    cores: f64,
    /// Frames a second the panes dropped: periods too slow to take, and frames displaced before they
    /// were painted.
    dropped_fps: f64,
    /// The process's RAM (resident) and VRAM (GPU-attributed) memory, in MiB, and its thread count,
    /// sampled once a second alongside the rates. On an integrated part, VRAM is system memory the
    /// GPU has reserved rather than a separate pool.
    ram_mib: f64,
    vram_mib: f64,
    threads: usize,
}

impl Grid {
    fn new(
        sources: Vec<PathBuf>,
        cols: usize,
        rows: usize,
        window: (f32, f32),
        mode: Mode,
        scale: f32,
        phases: Rc<RefCell<PhaseMetrics>>,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut grid = Self {
            panes: Vec::new(),
            generation: 0,
            sources,
            cols,
            rows,
            cell: cell_size(window, cols, rows),
            scale,
            mode,
            scale_mode: pane_scale(),
            metrics: Arc::new(Metrics::default()),
            live_tasks: Arc::new(AtomicUsize::new(0)),
            phases,
            snapshot: (0, 0, 0, Instant::now()),
            dropped_snapshot: 0,
            phases_snapshot: (PhaseMetrics::default(), Instant::now()),
            readout: Readout {
                ram_mib: process_rss_mib(),
                vram_mib: process_gpu_mib(),
                threads: process_threads(),
                ..Readout::default()
            },
            replaced: Vec::new(),
            budget: pane_budget(),
            cursor: 0,
            pace: pane_pace(),
            autoflip: std::env::var("GPUI_GRID_AUTOFLIP").ok().and_then(|v| v.parse().ok()),
            started: Instant::now(),
            flips: 0,
        };
        grid.start_panes(cx);
        grid
    }

    /// Start one worker and one drain task per pane. Pane `i` runs `sources[i % n]` from a fraction
    /// `i / panes` into it, so the wall is a spread of stages rather than copies of one frame.
    fn start_panes(&mut self, cx: &mut Context<Self>) {
        // The panes being replaced may hold readback images; keep them for the next render to
        // release, since clearing the pane drops the handle without freeing its atlas tile.
        let abandoned: Vec<Arc<RenderImage>> = self
            .panes
            .iter_mut()
            .filter_map(|pane| match pane.content.take() {
                Some(Content::Image(image)) => Some(image),
                _ => None,
            })
            .collect();
        self.replaced.extend(abandoned);
        self.panes.clear();
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        self.metrics = Arc::new(Metrics::default());
        self.snapshot = (0, 0, 0, Instant::now());
        self.phases_snapshot = (*self.phases.borrow(), Instant::now());

        let count = self.cols * self.rows;
        for index in 0..count {
            let path = self.sources[index % self.sources.len()].clone();
            let frac = index as f64 / count as f64;
            let mode = self.mode;
            let metrics = self.metrics.clone();
            let fit = match self.scale_mode {
                Scale::Fit => Some((self.cell.0 * self.scale, self.cell.1 * self.scale)),
                Scale::Native => None,
            };
            let (to_ui, from_worker) = flume::unbounded::<ToUi>();
            let (to_worker, for_worker) = flume::unbounded::<ToWorker>();

            let spawned = std::thread::Builder::new()
                .name(format!("grid-pane-{index}"))
                .spawn(move || worker(path, frac, mode, for_worker, to_ui, metrics, fit));
            if spawned.is_err() {
                eprintln!("grid: could not start a worker thread");
                continue;
            }

            let live = self.live_tasks.clone();
            self.live_tasks.fetch_add(1, Ordering::Relaxed);

            cx.spawn(async move |this, cx| {
                while let Ok(event) = from_worker.recv_async().await {
                    if this
                        .update(cx, |grid, cx| {
                            grid.on_event(generation, index, event);
                            cx.notify();
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                live.fetch_sub(1, Ordering::Relaxed);
            })
            .detach();

            self.panes.push(Pane {
                worker: to_worker,
                content: None,
                painted: false,
                dropped: 0,
                outstanding: 0,
                pending_since: None,
                due: Instant::now(),
                period: DEFAULT_PERIOD,
            });
        }
    }

    fn on_event(&mut self, generation: u64, index: usize, event: ToUi) {
        // A reply from a superseded pane would otherwise land on the pane that reused its index,
        // replacing a frame that was just handed out — and so never painted — which strands the new
        // playback's ring: the frame it displaced never signals its release.
        if generation != self.generation {
            return;
        }
        let mut replaced = None;
        {
            let Some(pane) = self.panes.get_mut(index) else {
                return;
            };
            match event {
                ToUi::Opened(period) => pane.period = period,
                ToUi::Frame(content) => {
                    match pane.content.replace(content) {
                        // The producer re-served the frame already showing (its ring had no free
                        // slot): nothing changed, nothing dropped, nothing to release.
                        Some(displaced) if same_surface(&displaced, pane.content.as_ref()) => {
                            if let Content::Image(image) = displaced {
                                replaced = Some(image);
                            }
                        }
                        // The frame it displaces was shown: a readback image still needs its atlas
                        // tile released at the next render.
                        Some(displaced) if pane.painted => {
                            if let Content::Image(image) = displaced {
                                replaced = Some(image);
                            }
                        }
                        // The frame it displaces was never painted — a dropped frame. A dma-buf the
                        // renderer never sampled is never released, so release it here, and count it.
                        Some(displaced) => {
                            pane.dropped += 1;
                            match displaced {
                                Content::Surface(source) => release_surface(&source),
                                Content::Image(image) => replaced = Some(image),
                            }
                        }
                        None => {}
                    }
                    pane.painted = false;
                    pane.outstanding = pane.outstanding.saturating_sub(1);
                    pane.pending_since = None;
                }
                ToUi::Idle => {
                    pane.outstanding = pane.outstanding.saturating_sub(1);
                    pane.pending_since = None;
                }
            }
        }
        if let Some(previous) = replaced {
            self.replaced.push(previous);
        }
    }

    /// Flip the handover and rebuild the panes on it.
    fn switch_mode(&mut self, cx: &mut Context<Self>) {
        self.mode = match self.mode {
            Mode::Surface => Mode::Readback,
            Mode::Readback => Mode::Surface,
        };
        self.start_panes(cx);
        cx.notify();
    }

    /// Ask up to `budget` due panes for their next frame, starting where the last render stopped, so
    /// the panes take turns rather than the same leading ones updating every frame. `budget` of `0`
    /// lifts the cap and asks every due pane.
    fn tick_panes(&mut self) {
        let count = self.panes.len();
        if count == 0 {
            return;
        }
        let budget = if self.budget == 0 {
            count
        } else {
            self.budget.min(count)
        };
        let pace = self.pace;
        let mut advanced = 0;
        let mut scanned = 0;
        while advanced < budget && scanned < count {
            if self.panes[(self.cursor + scanned) % count].tick(pace) {
                advanced += 1;
            }
            scanned += 1;
        }
        self.cursor = (self.cursor + scanned) % count;
    }

    /// Read the workers and the pipeline, keeping a per-second delta for both.
    fn measure(&mut self) {
        // The pane workers.
        let frames = self.metrics.frames.load(Ordering::Relaxed);
        let work_ns = self.metrics.work_ns.load(Ordering::Relaxed);
        let cpu_ns = self.metrics.cpu_ns.load(Ordering::Relaxed);
        let (last_frames, last_work_ns, last_cpu_ns, at) = self.snapshot;
        let elapsed = at.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            let delta_frames = frames.saturating_sub(last_frames) as f64;
            let delta_work = work_ns.saturating_sub(last_work_ns) as f64;
            let delta_cpu = cpu_ns.saturating_sub(last_cpu_ns) as f64;
            self.readout.pane_fps = delta_frames / elapsed;
            self.readout.ms_per_frame = if delta_frames == 0.0 {
                0.0
            } else {
                // Wall time per frame: the worker's latency, hardware waits included.
                delta_work / 1e6 / delta_frames
            };
            // CPU time as a fraction of a core: what the workers actually burn, excluding waits.
            self.readout.cores = delta_cpu / 1e9 / elapsed;
            let dropped: u64 = self.panes.iter().map(|pane| pane.dropped).sum();
            self.readout.dropped_fps =
                dropped.saturating_sub(self.dropped_snapshot) as f64 / elapsed;
            self.dropped_snapshot = dropped;
            self.snapshot = (frames, work_ns, cpu_ns, Instant::now());
        }

        // The frame pipeline: how often the window redraws, and what its passes cost.
        let phases = *self.phases.borrow();
        let (last, since) = self.phases_snapshot;
        let render_elapsed = since.elapsed().as_secs_f64();
        if render_elapsed >= 1.0 {
            let frames = phases.frames.saturating_sub(last.frames);
            self.readout.render_fps = frames as f64 / render_elapsed;
            if frames > 0 {
                let per = |total: Duration, prev: Duration| {
                    (total - prev).as_secs_f64() * 1000.0 / frames as f64
                };
                self.readout.render_eval_ms = per(phases.evaluate, last.evaluate);
                self.readout.render_layout_ms = per(phases.layout, last.layout);
                self.readout.render_paint_ms = per(phases.paint, last.paint);
            }
            self.readout.ram_mib = process_rss_mib();
            self.readout.vram_mib = process_gpu_mib();
            self.readout.threads = process_threads();
            self.phases_snapshot = (phases, Instant::now());
            eprintln!(
                "grid: {} panes, {} — render {:.0} fps (eval {:.2} layout {:.2} paint {:.2} ms/frame); \
                 panes {:.0} frames/s ({:.1}/pane), {:.2} ms/frame, {:.0}% cpu, {:.0} dropped/s; {:.0} MiB ram, {:.0} MiB vram, {} threads, {} tasks",
                self.panes.len(),
                self.mode.label(),
                self.readout.render_fps,
                self.readout.render_eval_ms,
                self.readout.render_layout_ms,
                self.readout.render_paint_ms,
                self.readout.pane_fps,
                self.readout.pane_fps / self.panes.len().max(1) as f64,
                self.readout.ms_per_frame,
                self.readout.cores * 100.0,
                self.readout.dropped_fps,
                self.readout.ram_mib,
                self.readout.vram_mib,
                self.readout.threads,
                self.live_tasks.load(Ordering::Relaxed),
            );
        }
    }
}

impl Render for Grid {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Release the atlas tiles of the readback images the panes have replaced, so the atlas does
        // not grow with every frame.
        for image in std::mem::take(&mut self.replaced) {
            let _ = window.drop_image(image);
        }
        if let Some(at) = self.autoflip {
            let elapsed = self.started.elapsed().as_secs_f64();
            let flips = (elapsed / at) as u32;
            if flips > self.flips {
                self.flips = flips;
                self.switch_mode(cx);
                eprintln!("grid: autoflip {flips} -> {}", self.mode.label());
            }
        }
        // Ask for the next frame of at most `budget` due panes, taking turns from where the last
        // render stopped, so a wall of hundreds shares the render rate instead of every pane
        // presenting at once and the UI thread choking on the uploads.
        self.tick_panes();
        // Keep the frame loop alive: the panes always have a next frame due.
        window.request_animation_frame();

        self.measure();

        let toggle = cx.entity();
        let readout = &self.readout;
        // The readout floats over the wall, on a translucent panel, so the panes get the whole
        // window and the numbers sit over the video rather than beside it.
        let overlay = div()
            .absolute()
            .top(px(PAD))
            .left(px(PAD))
            .flex()
            .flex_col()
            .gap_1()
            .px_3()
            .py_2()
            .rounded_md()
            .bg(rgba(0x101014d9))
            .text_color(rgb(0xe0e0e8))
            .text_sm()
            .child(
                div().flex().child(
                    div()
                        .id("grid-mode")
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(self.mode.accent())
                        .cursor_pointer()
                        .child(self.mode.label())
                        .on_click(move |_, _window, cx| {
                            toggle.update(cx, |grid, cx| grid.switch_mode(cx));
                        }),
                ),
            )
            .child(format!(
                "{} panes ({}x{}) · {} · {} · {}",
                self.panes.len(),
                self.cols,
                self.rows,
                if self.budget == 0 {
                    "budget off".to_string()
                } else {
                    format!("budget {}/render", self.budget.min(self.panes.len()))
                },
                self.pace.label(),
                self.scale_mode.label(),
            ))
            .child(format!(
                "render {:.0} fps (eval {:.1} + layout {:.1} + paint {:.1} ms)",
                readout.render_fps,
                readout.render_eval_ms,
                readout.render_layout_ms,
                readout.render_paint_ms,
            ))
            .child(format!(
                "panes {:.0} frames/s ({:.1}/pane) · {:.2} ms/frame · {:.0}% cpu · {:.0} dropped/s",
                readout.pane_fps,
                readout.pane_fps / self.panes.len().max(1) as f64,
                readout.ms_per_frame,
                readout.cores * 100.0,
                readout.dropped_fps,
            ))
            .child(format!(
                "ram {:.0} MiB · vram {:.0} MiB · {} threads",
                readout.ram_mib, readout.vram_mib, readout.threads,
            ));

        let mut rows = Vec::with_capacity(self.rows);
        for chunk in self.panes.chunks(self.cols) {
            // Fixed pane sizes, computed once: the row is as tall as a cell, and the panes as wide as
            // one, so the grid fills the fixed window with no flexible layout to recompute.
            let mut row = div().flex().gap(px(GAP)).h(px(self.cell.1));
            for pane in chunk {
                row = row.child(pane_view(pane, self.cell));
            }
            rows.push(row);
        }

        // Everything with content is painted this frame, so a frame that arrives before the next
        // render displaces a painted one rather than a dropped one.
        for pane in &mut self.panes {
            if pane.content.is_some() {
                pane.painted = true;
            }
        }

        div()
            .size_full()
            .relative()
            .bg(rgb(0x101014))
            .child(
                div()
                    .size_full()
                    .flex()
                    .flex_col()
                    .gap(px(GAP))
                    .p(px(PAD))
                    .overflow_hidden()
                    .children(rows),
            )
            .child(overlay)
    }
}

/// One pane's pixels, scaled proportionally into its cell of `cell` size. The painted element is
/// given the cell size directly — a `surface` carries no size of its own — with no wrapper: at a wall
/// of hundreds of panes, one element per pane instead of two keeps the frame's layout down.
fn pane_view(pane: &Pane, cell: (f32, f32)) -> AnyElement {
    match &pane.content {
        Some(Content::Surface(source)) => surface(source.clone())
            .object_fit(ObjectFit::Contain)
            .w(px(cell.0))
            .h(px(cell.1))
            .into_any_element(),
        Some(Content::Image(image)) => img(ImageSource::Render(image.clone()))
            .object_fit(ObjectFit::Contain)
            .w(px(cell.0))
            .h(px(cell.1))
            .into_any_element(),
        None => div()
            .w(px(cell.0))
            .h(px(cell.1))
            .bg(rgb(0x202028))
            .into_any_element(),
    }
}

/// The calling thread's CPU time, in nanoseconds, from the per-thread clock. Unlike wall time it
/// does not count blocking — a worker waiting on the hardware decoder is idle, not busy — so what is
/// built from it is real CPU.
fn thread_cpu_ns() -> u64 {
    let mut spec = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `spec` is a valid, writable `timespec`, and the clock id is a constant.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut spec) } != 0 {
        return 0;
    }
    (spec.tv_sec as u64) * 1_000_000_000 + (spec.tv_nsec as u64)
}

/// Whether two contents are the same dma-buf, so a producer that re-serves the frame already showing
/// is not mistaken for a new one the pane should drop and release.
fn same_surface(a: &Content, b: Option<&Content>) -> bool {
    match (a, b) {
        (Content::Surface(a), Some(Content::Surface(b))) => a == b,
        _ => false,
    }
}

/// Release a surface a pane dropped without painting it, so the producer's bounded ring does not
/// wait on a frame nothing ever sampled.
fn release_surface(source: &SurfaceSource) {
    match source {
        #[cfg(target_os = "linux")]
        SurfaceSource::DmaBuf(handle) => handle.signal_release(),
        #[allow(
            unreachable_patterns,
            reason = "the other platform sources do not exist here"
        )]
        _ => {}
    }
}

/// The fixed pane size that fits `cols` by `rows` panes into a `window` of `(width, height)`, given
/// the gaps, padding and header.
fn cell_size(window: (f32, f32), cols: usize, rows: usize) -> (f32, f32) {
    let inner_w = window.0 - 2.0 * PAD - (cols.saturating_sub(1)) as f32 * GAP;
    let inner_h = window.1 - 2.0 * PAD - (rows.saturating_sub(1)) as f32 * GAP;
    (
        (inner_w / cols as f32).max(1.0),
        (inner_h / rows as f32).max(1.0),
    )
}

/// A pane's worker: open its source, start `frac` of the way in, then decode one frame per request.
fn worker(
    path: PathBuf,
    frac: f64,
    mode: Mode,
    requests: flume::Receiver<ToWorker>,
    to_ui: flume::Sender<ToUi>,
    metrics: Arc<Metrics>,
    fit: Option<(f32, f32)>,
) {
    match mode {
        Mode::Surface => surface_worker(path, frac, requests, to_ui, metrics),
        Mode::Readback => readback_worker(path, frac, requests, to_ui, metrics, fit),
    }
}

/// The zero-copy worker: a bounded playback, handing over the exported surface.
fn surface_worker(
    path: PathBuf,
    frac: f64,
    requests: flume::Receiver<ToWorker>,
    to_ui: flume::Sender<ToUi>,
    metrics: Arc<Metrics>,
) {
    let Some(mut playback) = Playback::open_path(&path, DEPTH) else {
        eprintln!("grid: cannot open {}", path.display());
        return;
    };
    start_at(playback.duration(), frac).map(|at| playback.seek(at));
    let _ = to_ui.send(ToUi::Opened(playback.frame_period().unwrap_or(DEFAULT_PERIOD)));
    while let Ok(ToWorker::Advance) = requests.recv() {
        let started = Instant::now();
        let cpu_before = thread_cpu_ns();
        match playback.surface() {
            Some(source) => {
                metrics.record(
                    started.elapsed(),
                    Duration::from_nanos(thread_cpu_ns().saturating_sub(cpu_before)),
                );
                if to_ui.send(ToUi::Frame(Content::Surface(source))).is_err() {
                    break;
                }
            }
            // No frame ready this turn: answer the request anyway, so the pane is never left waiting
            // on a frame that will not come.
            None => {
                if to_ui.send(ToUi::Idle).is_err() {
                    break;
                }
            }
        }
    }
}

/// The per-render pane-update budget: `GPUI_GRID_BUDGET`, or [`BUDGET`]. `0` lifts the cap.
fn pane_budget() -> usize {
    std::env::var("GPUI_GRID_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(BUDGET)
}

/// A pool of permits bounding how many readbacks run at once, so a wall of readback panes cannot
/// oversubscribe the cores and starve the UI thread (which still has to upload every frame). The
/// default is one per core; `GPUI_GRID_PERMITS` overrides it.
fn readback_permits() -> &'static (flume::Sender<()>, flume::Receiver<()>) {
    use std::sync::OnceLock;
    static PERMITS: OnceLock<(flume::Sender<()>, flume::Receiver<()>)> = OnceLock::new();
    PERMITS.get_or_init(|| {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let count = std::env::var("GPUI_GRID_PERMITS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(cores)
            .max(1);
        let (sender, receiver) = flume::bounded(count);
        for _ in 0..count {
            let _ = sender.send(());
        }
        (sender, receiver)
    })
}

/// The readback worker: decode, copy the pixels to the CPU, convert, and hand over an image.
fn readback_worker(
    path: PathBuf,
    frac: f64,
    requests: flume::Receiver<ToWorker>,
    to_ui: flume::Sender<ToUi>,
    metrics: Arc<Metrics>,
    fit: Option<(f32, f32)>,
) {
    let Some(mut demuxer) = Demuxer::open(&path) else {
        eprintln!("grid: cannot open {}", path.display());
        return;
    };
    let Some(mut decoder) = demuxer.decoder() else {
        eprintln!("grid: cannot decode {}", path.display());
        return;
    };
    if let Some(at) = start_at(demuxer.duration(), frac) {
        demuxer.seek(at);
        decoder.flush();
    }
    let _ = to_ui.send(ToUi::Opened(demuxer.frame_period().unwrap_or(DEFAULT_PERIOD)));
    let (permits, permit_pool) = readback_permits();
    let mut scratch: Option<gpui_va::Readback> = None;
    while let Ok(ToWorker::Advance) = requests.recv() {
        // Pull the next decoded frame, rewinding in place at its end so the pane loops.
        let frame = loop {
            if let Some(frame) = decoder.receive() {
                break Some(frame);
            }
            match demuxer.send_next(&mut decoder) {
                Ok(true) => {}
                _ => {
                    // Loop by rewinding the same demuxer and decoder, not by reopening them: a fresh
                    // decoder builds a fresh VA-API device and surface pool the driver does not hand
                    // back, so reopening once per lap leaks until the machine swaps. If the stream
                    // will not seek back, the pane stops rather than spinning.
                    if demuxer.seek(Duration::ZERO).is_some() {
                        decoder.rewind();
                    } else {
                        break None;
                    }
                }
            }
        };
        let Some(frame) = frame else {
            if to_ui.send(ToUi::Idle).is_err() {
                break;
            }
            continue;
        };
        // Take a permit around the readback, so it runs no more than the pool allows even when every
        // pane wants one at once. The wait is not part of the measured work.
        let _permit = permit_pool.recv();
        if scratch.is_none() {
            scratch = frame.readback();
        }
        let started = Instant::now();
        let cpu_before = thread_cpu_ns();
        // Convert straight to the pane's size where asked, so the upload carries only what it shows.
        let (dst_width, dst_height) = match fit {
            Some(box_size) => contain_size(
                (frame.handle().width, frame.handle().height),
                box_size,
            ),
            None => (0, 0),
        };
        let read = scratch
            .as_mut()
            .and_then(|scratch| scratch.read_to(&frame, dst_width, dst_height));
        metrics.record(
            started.elapsed(),
            Duration::from_nanos(thread_cpu_ns().saturating_sub(cpu_before)),
        );
        let _ = permits.send(());
        let Some(read) = read else {
            if to_ui.send(ToUi::Idle).is_err() {
                break;
            }
            continue;
        };
        // The image owns its pixels, so the scratch's buffer is copied into it.
        let image = bgra_image(read.width, read.height, read.bgra.to_vec());
        if to_ui.send(ToUi::Frame(Content::Image(image))).is_err() {
            break;
        }
    }
}

/// Where in the stream a pane at `frac` of the way should start, once its duration is known.
fn start_at(duration: Option<Duration>, frac: f64) -> Option<Duration> {
    let duration = duration?;
    (frac > 0.0).then(|| duration.mul_f64(frac.clamp(0.0, 1.0)))
}

/// Wrap read-back `BGRA8` bytes as the packed image the image element uploads.
///
/// The renderer samples the buffer as `BGRA`, so the bytes stay as the readback wrote them; the image
/// crate has no `Bgra` pixel type, so the buffer is tagged `Rgba` and its channel order is the same
/// reinterpretation the SVG renderer makes after swapping its pixels.
fn bgra_image(width: u32, height: u32, bgra: Vec<u8>) -> Arc<RenderImage> {
    let buffer = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, bgra)
        .expect("a readback is width * height * 4 bytes");
    let frame = image::Frame::new(buffer);
    Arc::new(RenderImage::new(smallvec::SmallVec::from_vec(vec![frame])))
}

/// This process's resident set, in MiB, from `/proc/self/statm`.
fn process_rss_mib() -> f64 {
    let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else {
        return 0.0;
    };
    let pages: u64 = statm
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    pages as f64 * 4096.0 / (1024.0 * 1024.0)
}

/// This process's GPU memory, in MiB, from the DRM clients it holds.
///
/// The kernel accounts, per client, every buffer a DRM client has allocated — what the Vulkan
/// renderer and the video engine reserve — and reports it in that client's descriptor `fdinfo`,
/// under `drm-total-<region>`, in KiB. Several descriptors can name one client (a device dup'd for
/// each decoder), so each client is totalled once; several clients (one per decoder) are each
/// counted, which is exactly the point.
fn process_gpu_mib() -> f64 {
    use std::collections::HashMap;

    let Ok(entries) = std::fs::read_dir("/proc/self/fd") else {
        return 0.0;
    };
    let mut clients: HashMap<u64, u64> = HashMap::new();
    for entry in entries.flatten() {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        // `/proc/self/fd/N` is a symlink to the target, so read the shadow `fdinfo` entry instead.
        let Ok(info) = std::fs::read_to_string(format!("/proc/self/fdinfo/{name}")) else {
            continue;
        };
        let mut client = None;
        let mut kib = 0u64;
        for line in info.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            if key == "drm-client-id" {
                client = value.trim().parse().ok();
            } else if key.starts_with("drm-total-") {
                // The kernel prints the value in KiB, with the unit after the number.
                let number = value.trim().split_whitespace().next().unwrap_or("");
                if let Ok(v) = number.parse::<u64>() {
                    kib += v;
                }
            }
        }
        if let Some(client) = client {
            clients.insert(client, kib);
        }
    }
    clients.values().sum::<u64>() as f64 / 1024.0
}

/// This process's thread count, from `/proc/self/status`.
fn process_threads() -> usize {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    status
        .lines()
        .find_map(|line| line.strip_prefix("Threads:").and_then(|v| v.trim().parse().ok()))
        .unwrap_or(0)
}

/// Parse a `WxH` size, as `--size` takes it.
fn parse_size(value: &str) -> Option<(f32, f32)> {
    let (w, h) = value.split_once(['x', 'X'])?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

fn main() {
    let mut sources: Vec<PathBuf> = Vec::new();
    let mut mode = Mode::Surface;
    let mut cols = 16usize;
    let mut rows = 16usize;
    let mut window = (WINDOW_W, WINDOW_H);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--readback" => mode = Mode::Readback,
            "--cols" => cols = args.next().and_then(|n| n.parse().ok()).unwrap_or(cols),
            "--rows" => rows = args.next().and_then(|n| n.parse().ok()).unwrap_or(rows),
            "--size" => {
                if let Some((w, h)) = args.next().and_then(|v| parse_size(&v)) {
                    window = (w, h);
                }
            }
            path => sources.push(PathBuf::from(path)),
        }
    }
    cols = cols.max(1);
    rows = rows.max(1);
    if sources.is_empty() {
        // The checked-in clip: every pane is still its own decoder, started at its own point.
        sources = vec![
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../gpui_va/fixtures/clip.mp4"),
        ];
    }

    // The window is a fixed size; the grid fills it, whatever shape it is.
    let (window_w, window_h) = window;

    let phases = Rc::new(RefCell::new(PhaseMetrics::default()));
    let pipeline_phases = phases.clone();
    let grid_sources = sources.clone();

    gpui::application()
        .with_frame_pipeline(move |_| {
            Box::new(StandardImmediatePipeline.instrumented(pipeline_phases.clone()))
        })
        .run(move |cx: &mut App| {
            let bounds = Bounds::centered(None, size(px(window_w), px(window_h)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    is_resizable: false,
                    ..Default::default()
                },
                move |window, cx| {
                    let scale = window.scale_factor();
                    cx.new(|cx| {
                        Grid::new(
                            grid_sources.clone(),
                            cols,
                            rows,
                            (window_w, window_h),
                            mode,
                            scale,
                            phases.clone(),
                            cx,
                        )
                    })
                },
            )
            .expect("a window for the grid");
            eprintln!(
                "grid: {}x{} = {} panes, starting in {}",
                cols,
                rows,
                cols * rows,
                mode.label(),
            );
            cx.activate(true);
        });
}
