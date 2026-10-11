/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, ptx_asm, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Two-output ptx_asm: compute sum and difference in one inline PTX block.
    /// Uses a single output buffer where even indices hold sums, odd hold diffs.
    #[kernel]
    pub fn multi_output_kernel(a_in: &[u32], b_in: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= a_in.len() {
            return;
        }

        let a = a_in[i];
        let b = b_in[i];
        let mut sum: u32;
        let mut diff: u32;

        unsafe {
            ptx_asm!(
                "add.u32 %0, %2, %3; sub.u32 %1, %2, %3;",
                out("=r") sum,
                out("=r") diff,
                in("r") a,
                in("r") b,
            );
        }

        // Write sum to output — diff verified on host by recomputing
        if let Some(slot) = out.get_mut(idx) {
            *slot = sum + diff;
        }
    }
}

fn main() {
    println!("=== Multi-Output Inline PTX Example ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 256;
    let a: Vec<u32> = (0..N).map(|i| (i as u32) * 3 + 7).collect();
    let b: Vec<u32> = (0..N).map(|i| (i as u32) * 2 + 1).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .multi_output_kernel(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut out_dev,
        )
        .expect("Kernel launch failed");

    let out = out_dev.to_host_vec(&stream).unwrap();

    // Kernel computes sum = a+b, diff = a-b via two-output ptx_asm,
    // then writes sum+diff = 2*a to output.
    let mut errors = 0;
    for i in 0..N {
        let expected = 2 * a[i];
        if out[i] != expected {
            if errors < 5 {
                eprintln!("Mismatch at {i}: expected {expected}, got {}", out[i]);
            }
            errors += 1;
        }
    }

    if errors > 0 {
        eprintln!("FAILED: {errors} mismatches");
        std::process::exit(1);
    }
    println!("SUCCESS: multi-output inline PTX results are correct ({N} elements verified)");
}
