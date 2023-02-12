pub trait UIState: Send + Sync + 'static {}

impl UIState for () {}
