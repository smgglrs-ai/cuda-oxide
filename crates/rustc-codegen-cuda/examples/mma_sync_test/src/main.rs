use cuda_device::{kernel, cuda_module, DisjointSlice};

#[cuda_module]
mod kernels {
    use super::*;

    /// Compute a 16x8 MMA using SharedArray for A (16x16) and B (8x16 for BT).
    /// A[r,c] = r + c*0.01 (each element unique)
    /// B[n,k] = n*0.1 + k*0.01
    /// This lets us verify the intrinsic with known data.
    #[kernel]
    pub fn mma_test(mut out: DisjointSlice<f32>) {
        static mut A_SM: cuda_device::SharedArray<f32, 256> = cuda_device::SharedArray::UNINIT;
        static mut B_SM: cuda_device::SharedArray<f32, 128> = cuda_device::SharedArray::UNINIT;

        let lane = cuda_device::thread::threadIdx_x() as usize;

        // A = identity 16x16, B[n,k] = n*100+k
        // Expected: D[m,n] = B_mma[m,n] = B_orig[n,m] = n*100+m
        let mut i = lane;
        while i < 256 {
            let r = i / 16;
            let c = i % 16;
            unsafe { A_SM[i] = if r == c { 1.0 } else { 0.0 }; }
            i += 32;
        }
        i = lane;
        while i < 128 {
            let n = i / 16;
            let k = i % 16;
            unsafe { B_SM[i] = (n * 100 + k) as f32; }
            i += 32;
        }
        cuda_device::thread::sync_threads();

        let group_id = lane >> 2;
        let tid = lane & 3;

        // Pack A fragments
        let a_frag: [u32; 4];
        unsafe {
            let r0 = group_id;
            let r1 = group_id + 8;
            // Try: A row=tid, col=groupID*2 (swap groupID/tid roles)
            let ar0 = tid;
            let ar1 = tid + 8; // Wait, tid only goes 0..3, +8 = 4..11 — wrong range
            // Actually, the PTX spec might use a different grouping
            // PTX ISA 8.7, page 415, section 9.7.14.5.8:
            // row = groupID            for ai where 0 <= i < 2 || 4 <= i < 6
            //       groupID + 8        Otherwise (i in {2,3,6,7})
            // col = tid*2 + (i & 1)              for ai where i < 4
            //       tid*2 + (i & 1) + 8           for ai where i >= 4
            //
            // Register layout (elements low to high): a0,a1,a2,a3,a4,a5,a6,a7
            // Packed into 4 u32s: {a0,a1}, {a2,a3}, {a4,a5}, {a6,a7}
            //
            // So:
            //   reg0 = {a0,a1} = A[groupID,   tid*2], A[groupID,   tid*2+1]
            //   reg1 = {a2,a3} = A[groupID+8, tid*2], A[groupID+8, tid*2+1]
            //   reg2 = {a4,a5} = A[groupID,   tid*2+8], A[groupID,   tid*2+9]
            //   reg3 = {a6,a7} = A[groupID+8, tid*2+8], A[groupID+8, tid*2+9]
            a_frag = [
                pack_f16x2(A_SM[r0 * 16 + tid * 2 + 1], A_SM[r0 * 16 + tid * 2]),         // {a0,a1} swapped
                pack_f16x2(A_SM[r1 * 16 + tid * 2 + 1], A_SM[r1 * 16 + tid * 2]),         // {a2,a3} swapped
                pack_f16x2(A_SM[r0 * 16 + tid * 2 + 9], A_SM[r0 * 16 + tid * 2 + 8]),    // {a4,a5} swapped
                pack_f16x2(A_SM[r1 * 16 + tid * 2 + 9], A_SM[r1 * 16 + tid * 2 + 8]),    // {a6,a7} swapped
            ];
        }

        // B fragment with SWAPPED pack order (hi, lo instead of lo, hi)
        let b_frag: [u32; 2];
        unsafe {
            let n = group_id;
            b_frag = [
                pack_f16x2(B_SM[n * 16 + tid * 2 + 1], B_SM[n * 16 + tid * 2]),
                pack_f16x2(B_SM[n * 16 + tid * 2 + 9], B_SM[n * 16 + tid * 2 + 8]),
            ];
        }

        let mut acc = [0.0f32; 4];
        unsafe {
            cuda_device::mma_sync::mma_sync_m16n8k16_f32_f16(&mut acc, &a_frag, &b_frag);
        }

        // Write all 32 threads' acc[0] values (D[groupID, tid*2])
        // into output[lane] for verification
        let idx = cuda_device::thread::index_1d();
        unsafe {
            *out.get_unchecked_mut(lane) = acc[0];
            *out.get_unchecked_mut(32 + lane) = acc[1];
            *out.get_unchecked_mut(64 + lane) = acc[2];
            *out.get_unchecked_mut(96 + lane) = acc[3];
        }
    }

    #[inline(always)]
    fn pack_f16x2(lo: f32, hi: f32) -> u32 {
        let result: u32;
        unsafe {
            cuda_device::ptx_asm!(
                "cvt.rn.f16x2.f32 %0, %2, %1;",
                out("=r") result,
                in("f") hi,
                in("f") lo,
            );
        }
        result
    }
}

fn main() {
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 128).unwrap();
    let module = kernels::load(&ctx).unwrap();
    module.mma_test(&stream, LaunchConfig { grid_dim: (1,1,1), block_dim: (32,1,1), shared_mem_bytes: 0 }, &mut out).unwrap();
    let gpu = out.to_host_vec(&stream).unwrap();

    // Reconstruct 16x8 D matrix from per-lane acc values.
    // acc[0] for lane L → D[groupID, tid*2] where groupID=L>>2, tid=L&3
    // acc[1] for lane L → D[groupID, tid*2+1]
    // acc[2] for lane L → D[groupID+8, tid*2]
    // acc[3] for lane L → D[groupID+8, tid*2+1]
    let mut d_gpu = vec![0.0f32; 16 * 8];
    for lane in 0..32 {
        let gid = lane >> 2;
        let tid = lane & 3;
        d_gpu[gid * 8 + tid * 2] = gpu[lane];            // acc[0]
        d_gpu[gid * 8 + tid * 2 + 1] = gpu[32 + lane];   // acc[1]
        d_gpu[(gid + 8) * 8 + tid * 2] = gpu[64 + lane];  // acc[2]
        d_gpu[(gid + 8) * 8 + tid * 2 + 1] = gpu[96 + lane]; // acc[3]
    }

    // CPU reference: C = A × B^T where A[r,c]=r+c*0.01, B[n,k]=n*0.1+k*0.01
    let mut d_cpu = vec![0.0f32; 16 * 8];
    for m in 0..16 {
        for n in 0..8 {
            let mut sum = 0.0f32;
            for k in 0..16 {
                let a = if m == k { 1.0f32 } else { 0.0 };
                let b = (n * 100 + k) as f32;
                sum += a * b;
            }
            d_cpu[m * 8 + n] = sum;
        }
    }

    println!("=== MMA 16x8 verification (A[r,c]=r+c*0.01, B[n,k]=n*0.1+k*0.01) ===");
    let mut max_diff = 0.0f32;
    let mut max_reldiff = 0.0f32;
    for i in 0..128 {
        let diff = (d_cpu[i] - d_gpu[i]).abs();
        max_diff = max_diff.max(diff);
        if d_cpu[i].abs() > 0.01 {
            max_reldiff = max_reldiff.max(diff / d_cpu[i].abs());
        }
    }
    for r in 0..8 {
        print!("  row {r:2}: CPU=[");
        for c in 0..8 { print!("{:7.2}", d_cpu[r*8+c]); }
        print!("]  GPU=[");
        for c in 0..8 { print!("{:7.2}", d_gpu[r*8+c]); }
        println!("]");
    }
    println!("  Max abs diff: {max_diff:.6}");
    println!("  Max rel diff: {:.4}%", max_reldiff * 100.0);
    if max_reldiff < 0.01 {
        println!("  PASS (<1% relative error)");
    } else {
        println!("  FAIL (>{:.1}% relative error)", max_reldiff * 100.0);
    }
}
