//! The producer side of the surface path.
//!
//! `gpui_engine` owns the consumer's half: [`SurfaceSource`], the payload a renderer samples, and
//! `SurfaceFormat`, the shape it is read as. This crate owns the half that feeds it — what something
//! outside GPUI implements — so a video decoder, a capture pipeline and a 3D renderer reach the
//! `surface()` element the same way, and a consumer is written once against all of them rather than
//! once per producer.
//!
//! It is deliberately the **surface** seam and not a video one. A video producer adds a *cadence* (a
//! frame every period) and a *pool* (recycle a surface only once the renderer released it); a 3D
//! producer adds neither. What they share, and all this seam carries, is a surface.

use gpui_engine::SurfaceSource;

/// Something outside GPUI that produces surfaces for GPUI to composite.
///
/// One method, because a surface is the whole of what crosses: the transport-specific payload is
/// already described by [`SurfaceSource`], and how the next one is made — a frame decoded, a pass
/// rendered, a buffer captured — is the producer's own business.
///
/// A producer with a cadence or a pool keeps them *below* this trait. Those are video shapes; a 3D
/// producer has neither, and a consumer that only wants surfaces should not have to know about them.
///
/// Object-safe on purpose: an element holds a producer it does not know the concrete type of.
pub trait SurfaceProducer {
    /// The surface to composite now, or `None` when the producer has nothing ready.
    ///
    /// `None` is not a failure. A decoder waiting on its input and a producer between frames both
    /// have nothing this frame; a caller that already has a surface keeps showing it.
    fn surface(&mut self) -> Option<SurfaceSource>;
}

#[cfg(test)]
mod tests {
    use super::SurfaceProducer;

    #[test]
    fn a_producer_may_have_nothing_and_is_object_safe() {
        /// The smallest producer: one that never has a surface, which is a caller's first case.
        struct Nothing;

        impl SurfaceProducer for Nothing {
            fn surface(&mut self) -> Option<gpui_engine::SurfaceSource> {
                None
            }
        }

        let mut producer: Box<dyn SurfaceProducer> = Box::new(Nothing);
        assert!(producer.surface().is_none());
    }
}
