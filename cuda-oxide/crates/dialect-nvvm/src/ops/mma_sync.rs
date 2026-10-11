/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ampere/Ada mma.sync warp-level matrix multiply-accumulate operations.
//!
//! mma.sync.aligned.m16n8k16.row.col.f32.f32:
//! - 10 operands: a0,a1,a2,a3 (i32/v2f16), b0,b1 (i32/v2f16), c0,c1,c2,c3 (f32)
//! - 4 results: d0,d1,d2,d3 (f32)

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// mma.sync.aligned.m16n8k16.row.col.f32.f32
///
/// Warp-level matrix multiply-accumulate for Ampere+ (sm_80+).
///
/// LLVM intrinsic: `@llvm.nvvm.mma.m16n8k16.row.col.f32.f32`
///
/// # Operands (10)
///
/// - a0, a1, a2, a3 (i32): A fragments (packed f16×2, bitcast to v2f16 at LLVM level)
/// - b0, b1 (i32): B fragments (packed f16×2, bitcast to v2f16 at LLVM level)
/// - c0, c1, c2, c3 (f32): C accumulator fragments
///
/// # Results (4)
///
/// - d0, d1, d2, d3 (f32): D output fragments
#[pliron_op(
    name = "nvvm.mma_sync_m16n8k16_f32_f16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<10>, NResultsInterface<4>],
)]
pub struct MmaSyncM16N8K16F32F16Op;

impl MmaSyncM16N8K16F32F16Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaSyncM16N8K16F32F16Op { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    MmaSyncM16N8K16F32F16Op::register(ctx);
}
