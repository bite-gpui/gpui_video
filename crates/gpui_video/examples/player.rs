//! A video player: a stream hardware-decoded by VA-API, played into a window with HTML5-shaped
//! controls.
//!
//! ```sh
//! cargo run -p gpui_video --example player
//! cargo run -p gpui_video --example player -- path/to/movie.mp4
//! ```
//!
//! Run with no argument it plays the bundled clip — sixty frames of SMPTE-style bars scrolling a
//! few pixels per frame, so the frames differ and the stream carries real inter-frame (`P`) frames.
//! Give it a path to play any stream libavformat opens: a container such as MP4 or Matroska, or a raw
//! H.264 elementary stream (Annex B). A container's packets are framed and its parameter sets kept
//! out of band, so it is decoded through the demuxer rather than the bitstream parser.
//!
//! # What this is
//!
//! The player end of the escape hatch, now almost entirely `gpui_video`: one element —
//! `video().source(path).controls(true)` — and the facade does the rest. `gpui_va` decodes the
//! stream, each frame exports its VA-API surface as a dma-buf under the DRM format modifier the driver
//! chose, and the window's renderer imports that buffer and converts it — no copy through the CPU, no
//! copy on the GPU. The facade opens the decoder (off the frame path, on the background executor),
//! paces it to the stream's rate, and drives the transport bar; the window only lays the picture out
//! and names the source.
//!
//! # Bounded, not all-in-memory
//!
//! The playback keeps only a few frames alive and recycles each surface once the renderer says it is
//! done with it. Every handle the producer hands out carries a **release descriptor**; the renderer
//! signals it from the completion callback of the submission that sampled the frame, and the producer
//! waits on it before returning the surface to its pool — and never recycles the frame still on
//! screen, which the scene re-samples every frame. That is what lets a stream play in constant memory.
//!
//! # Colour
//!
//! The renderer converts NV12 with the matrix and range the *producer declares on the frame's
//! handle*, and `gpui_va` reads both off the stream — so footage plays as it was graded rather than
//! through one assumed matrix. The bundled clip is full-range BT.601. A stream that names neither (as
//! most do) is resolved by convention: limited range, and BT.601 below 576 lines, BT.709 above.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!(
        "player is a Linux example: it plays a stream through the VA-API surface path. Run it on \
         Linux."
    );
}

#[cfg(target_os = "linux")]
fn main() {
    demo::run();
}

#[cfg(target_os = "linux")]
mod demo {
    use gpui::{
        App, Bounds, Context, IntoElement, Render, StandardImmediatePipeline, Window,
        WindowBounds, WindowOptions, div, prelude::*, px, rgb, size,
    };
    use gpui_video::{Source, video};

    /// The window redraws at the display's rate. The player samples that grid and picks up a frame
    /// when its period comes due, so pacing is never coarser than the content — a cap here would
    /// coarsen the grid and make a stream whose rate is not a divisor of the cap judge (see
    /// [`OwnedPlayer::tick`](gpui_video::Player)).
    pub fn run() {
        gpui::application()
            .with_frame_pipeline(|_window_id| Box::new(StandardImmediatePipeline))
            .run(|cx: &mut App| {
                let bounds = Bounds::centered(None, size(px(720.), px(480.)), cx);
                cx.open_window(
                    WindowOptions {
                        window_bounds: Some(WindowBounds::Windowed(bounds)),
                        ..Default::default()
                    },
                    |_, cx| cx.new(|_| PlayerView::new()),
                )
                .expect("a window to play into");
                cx.activate(true);
            });
    }

    /// The view, and nothing more: the video element opens the decoder and drives the transport, so
    /// there is no player entity here to manage.
    struct PlayerView {
        source: Source,
        /// What to show under the picture.
        label: String,
    }

    impl PlayerView {
        fn new() -> Self {
            let (source, label) = match std::env::args().nth(1) {
                // A named file goes through libavformat, so a container (MP4, Matroska, …) plays as
                // readily as a raw elementary stream.
                Some(path) => (path.as_str().into(), path),
                None => (gpui_va::CLIP.into(), "bundled clip".to_owned()),
            };
            eprintln!("gpui_video player: playing {label}");
            Self { source, label }
        }
    }

    impl Render for PlayerView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div()
                .size_full()
                .bg(rgb(0x101014))
                .flex()
                .flex_col()
                .child(
                    video()
                        .source(self.source.clone())
                        // The default bar, drawn over the picture. `.controls(false)` hides it, and
                        // any element of your own — or `.overlay(..)` for free placement — replaces it.
                        .controls(true)
                        .flex_1()
                        .w_full(),
                )
                .child(
                    div()
                        .px_3()
                        .py_2()
                        .text_sm()
                        .text_color(rgb(0x9a9aa2))
                        .child(self.label.clone()),
                )
        }
    }
}
