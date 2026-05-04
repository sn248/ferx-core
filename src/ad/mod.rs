#[cfg(feature = "autodiff")]
pub mod ad_gradients;
#[cfg(feature = "autodiff")]
pub mod event_driven_ad;
pub mod dual;

pub use dual::Dual;
