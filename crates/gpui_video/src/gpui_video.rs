//! The video element: whatever a [`SurfaceProducer`] hands over, composited every frame.
//!
//! This is the consumer the surface seam exists for. It names no platform — no
//! `#[cfg(target_os = …)]` fork, no concrete decoder — because it holds the seam rather than a
//! producer: an application builds the producer its platform has (`gpui_va::Playback` on Linux,
//! `gpui_cv::Decoder` on macOS) and hands it here.
//!
//! What it does *not* do is pace itself. A producer's cadence and its pool are the producer's: a
//! playback knows its frame period and recycles its own surfaces only once the renderer released
//! them, and a 3D producer has neither. Lifting those into the element would mean widening the seam,
//! which is a decision to make when a second producer needs it, not before.

use gpui_authoring::{AnyElement, App, AppContext as _, Context, IntoElement, Render, Window, gpu_canvas};
use gpui_surface_producer::SurfaceProducer;

/// A surface producer's output, as an element.
pub struct Video {
    producer: Box<dyn SurfaceProducer>,
}

impl Video {
    /// Wrap `producer`.
    ///
    /// The producer is held for the element's whole life, so a decoder's pool and position survive
    /// between frames rather than restarting each one.
    pub fn new(producer: impl SurfaceProducer + 'static) -> Self {
        Self {
            producer: Box::new(producer),
        }
    }

    /// Build the element in `cx`, for an application's `render`.
    pub fn into_element(self, cx: &mut App) -> AnyElement {
        cx.new(|_| self).into_any_element()
    }
}

impl Render for Video {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        // Asked once, at paint time: the producer either has a surface for this frame or it does not,
        // and a frame with nothing new keeps showing the last one, because the canvas is not cleared
        // between frames of the same element.
        let source = self.producer.surface();
        gpu_canvas(move |canvas| {
            if let Some(source) = source {
                canvas.paint_surface(source);
            }
        })
    }
}
