/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ampere/Ada mma.sync warp-level matrix multiply-accumulate (sm_80+).
//!
//! mma.sync is the pre-Hopper tensor core interface where 32 threads in a warp
//! collectively issue a matrix multiply. Each thread holds a fragment of the
//! operands and results in registers.
//!
//! # Supported Shape
//!
//! - **m16n8k16** with f16 inputs and f32 accumulator
//!   - A: 16×16 (f16), distributed as 4 packed f16×2 per thread (u32)
//!   - B: 16×8 (f16), distributed as 2 packed f16×2 per thread (u32)
//!   - C/D: 16×8 (f32), distributed as 4 f32 per thread
//!
//! # Hardware Support
//!
//! - **sm_80+**: Ampere (A100), Ada (L40S), Hopper (H100), Blackwell (B200)

/// Warp-level matrix multiply-accumulate: D = A × B + C.
///
/// Computes mma.sync.aligned.m16n8k16.row.col.f32.f32 where A and B are
/// f16 inputs packed as u32 (f16×2 pairs) and C/D are f32 accumulators.
///
/// The `acc` array serves as both C input and D output (accumulate in place).
///
/// # Fragment Distribution
///
/// Each of the 32 threads in the warp holds:
/// - `a[0..3]`: 4 × u32, each containing 2 packed f16 values (8 f16 total)
/// - `b[0..1]`: 2 × u32, each containing 2 packed f16 values (4 f16 total)
/// - `acc[0..3]`: 4 × f32 values
///
/// # Safety
///
/// - Must be called by ALL 32 threads in a warp together (warp-synchronous)
/// - Fragment values must be correctly distributed across threads
/// - Only available on sm_80+ (Ampere and later)
/// Marker for MIR importer dispatch. Body writes via volatile to prevent
/// the optimizer from inlining or eliminating the call.
#[inline(never)]
pub unsafe fn mma_sync_m16n8k16_f32_f16(
    acc: &mut [f32; 4],
    a: &[u32; 4],
    b: &[u32; 2],
) {
    core::ptr::write_volatile(&mut acc[0], a[0] as f32 + b[0] as f32);
}
