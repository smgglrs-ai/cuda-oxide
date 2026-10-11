/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! NVVM dialect operations.
//!
//! `mir-importer` creates these operations from recognized device intrinsic
//! calls. `mir-lower` then selects the reviewed backend route: a typed LLVM
//! NVVM intrinsic or inline PTX.
//!
//! Most leaf operations are generated under `generated`. The top-level
//! handwritten modules are limited to compiler infrastructure, composite
//! lowerings, and public compatibility types.
//!
//! All operations are re-exported here for convenience:
//!
//! ```ignore
//! use dialect_nvvm::ops::{ReadPtxSregTidXOp, Barrier0Op, ShflSyncBflyI32Op};
//! ```

mod asm;
pub mod atomic;
mod cluster;
mod debug;
mod generated;
mod grid;
mod mbarrier;
mod memory;
mod mma_sync;
mod stmatrix;
mod tcgen05;
mod thread;
mod tma;
mod warp;
mod wgmma;

use pliron::context::Context;

// Re-export all operations for public API
pub use asm::*;
pub use atomic::*;
pub use cluster::*;
pub use debug::*;
pub use generated::*;
pub use grid::*;
pub use mbarrier::*;
pub use memory::*;
pub use mma_sync::*;
pub use stmatrix::*;
pub use tcgen05::*;
pub use thread::*;
pub use tma::*;
pub use warp::*;
pub use wgmma::*;

/// Register all NVVM dialect operations with the context.
///
/// This function registers all operation types so they can be parsed,
/// verified, and printed. Must be called during dialect initialization.
pub fn register(ctx: &mut Context) {
    atomic::register(ctx);
    asm::register(ctx);
    cluster::register(ctx);
    generated::register(ctx);
    grid::register(ctx);
    wgmma::register(ctx);
    tcgen05::register(ctx);
    mma_sync::register(ctx);
    stmatrix::register(ctx);
    debug::register(ctx);
}
