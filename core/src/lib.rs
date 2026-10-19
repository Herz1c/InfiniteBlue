// Classical infinite chess: positions, exact legal moves, game-end rules, network input.
pub mod encode;
pub mod encode_ref;
pub mod mcts;
pub mod moves;
pub mod pos;
pub mod rules;
#[cfg(feature = "python")]
mod py;
