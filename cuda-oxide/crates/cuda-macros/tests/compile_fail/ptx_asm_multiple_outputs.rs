// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::ptx_asm;

fn main() {
    let a: u32;
    let b: u32;

    // Multiple outputs: second out must come before any in operand.
    // This tests that the "out after in" check still works.
    unsafe {
        ptx_asm!(
            "mov.u32 %0, %2; mov.u32 %1, %2;",
            out("=r") a,
            out("=r") b,
            in("r") 42u32,
        );
    }

    // Now test that out-after-in is still rejected.
    let c: u32;
    unsafe {
        ptx_asm!("add.u32 %0, %1, %1;", in("r") 1u32, out("=r") c);
        //~^ ERROR `out` operands must appear before `in` operands
    }
}
