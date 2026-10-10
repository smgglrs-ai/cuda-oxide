/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU correctness tests for the vec_width=4 optimization hint.
//!
//! Three kernels verify correctness under vec_width=4:
//! 1. f32 flat tensor — simple scale
//! 2. u32 flat tensor — identity copy
//! 3. u32 stride-9 access — mirrors Q2_0 weight layout

use cutile::prelude::*;
use crate::common;
use crate::audit_common::upload;

// ---------------------------------------------------------------------------
// Test 1: f32 flat tensor with vec_width=4
// ---------------------------------------------------------------------------

#[cutile::module]
mod vw_f32_mod {
    use cutile::core::*;

    /// Scale a 1-D f32 tensor by 2.0, loading tiles with vec_width=4 hint.
    #[cutile::entry(optimization_hints = (sm_120 = (vec_width = 4,)))]
    pub fn scale_f32<const N: i32>(
        z: &mut Tensor<f32, { [N] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile: Tile<f32, { [N] }> = x.load_tile(shape![N], [pid]);
        let two: Tile<f32, { [N] }> = broadcast_scalar(2.0_f32, shape![N]);
        z.store(tile * two);
    }
}

#[test]
fn vec_width_f32_load_correct() {
    common::with_test_stack(|| {
        const N: i32 = 128;
        let n_elems = 512usize;

        let x_host: Vec<f32> = (0..n_elems).map(|i| i as f32).collect();
        let x = upload(x_host.clone());
        let z = api::zeros::<f32>(&[n_elems]).partition([N as usize]);

        let result = vw_f32_mod::scale_f32(z, x)
            .generics(vec![N.to_string()])
            .grid((n_elems as u32 / N as u32, 1, 1))
            .first()
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("scale_f32 should succeed");

        for (i, &v) in result.iter().enumerate() {
            let expected = i as f32 * 2.0;
            assert!(
                (v - expected).abs() < 1e-5,
                "f32 load wrong at i={i}: got {v}, expected {expected}"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Test 2: u32 flat tensor with vec_width=4
// ---------------------------------------------------------------------------

#[cutile::module]
mod vw_u32_mod {
    use cutile::core::*;

    /// Identity copy of u32 elements with vec_width=4 hint.
    #[cutile::entry(optimization_hints = (sm_120 = (vec_width = 4,)))]
    pub fn copy_u32<const N: i32>(
        z: &mut Tensor<u32, { [N] }>,
        x: &Tensor<u32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile: Tile<u32, { [N] }> = x.load_tile(shape![N], [pid]);
        z.store(tile);
    }
}

#[test]
fn vec_width_u32_load_correct() {
    common::with_test_stack(|| {
        const N: i32 = 128;
        let n_elems = 512usize;

        let x_host: Vec<u32> = (0..n_elems).map(|i| i as u32 * 0x1234 + 0xDEAD).collect();
        let x = upload(x_host.clone());
        let z = api::zeros::<u32>(&[n_elems]).partition([N as usize]);

        let result = vw_u32_mod::copy_u32(z, x)
            .generics(vec![N.to_string()])
            .grid((n_elems as u32 / N as u32, 1, 1))
            .first()
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("copy_u32 should succeed");

        for (i, &v) in result.iter().enumerate() {
            let expected = i as u32 * 0x1234 + 0xDEAD;
            assert_eq!(
                v, expected,
                "u32 load wrong at i={i}: got 0x{v:08x}, expected 0x{expected:08x}"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Test 3: stride-9 u32 access with vec_width=4 — mirrors Q2_0 layout
// ---------------------------------------------------------------------------

#[cutile::module]
mod vw_stride9_mod {
    use cutile::core::*;

    /// Simulate Q2_0 access: each output pid loads 9 u32 words per block at
    /// offset `pid * K_BLOCKS * 9 + blk * 9 + w`. Offset NOT divisible by 4.
    #[cutile::entry(optimization_hints = (sm_120 = (vec_width = 4,)))]
    pub fn stride9<const K_BLOCKS: i32>(
        z: &mut Tensor<u32, { [1] }>,
        packed: &Tensor<u32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let part = packed.partition(shape![1]);
        let mut acc: Tile<u32, { [1] }> = broadcast_scalar(0_u32, shape![1]);

        for blk in 0i32..K_BLOCKS {
            let base = pid * K_BLOCKS * 9 + blk * 9;
            for w in 0i32..9 {
                let word: Tile<u32, { [1] }> = part.load([base + w]);
                acc = acc + word;
            }
        }
        z.store(acc);
    }
}

#[test]
fn vec_width_stride9_load_correct() {
    common::with_test_stack(|| {
        const K_BLOCKS: i32 = 16;
        let n_out = 4usize;
        let n_packed = n_out * K_BLOCKS as usize * 9;

        let packed_host: Vec<u32> = (0..n_packed).map(|i| i as u32).collect();
        let packed = upload(packed_host.clone());
        let z = api::zeros::<u32>(&[n_out]).partition([1usize]);

        let result = vw_stride9_mod::stride9(z, packed)
            .generics(vec![K_BLOCKS.to_string()])
            .grid((n_out as u32, 1, 1))
            .first()
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("stride9 should succeed");

        assert_eq!(result.len(), n_out);
        for pid in 0..n_out {
            let mut expected: u32 = 0;
            for blk in 0..K_BLOCKS as usize {
                for w in 0..9usize {
                    expected = expected.wrapping_add((pid * K_BLOCKS as usize * 9 + blk * 9 + w) as u32);
                }
            }
            assert_eq!(
                result[pid], expected,
                "stride9 pid={pid}: got {}, expected {expected}",
                result[pid]
            );
        }
    });
}
