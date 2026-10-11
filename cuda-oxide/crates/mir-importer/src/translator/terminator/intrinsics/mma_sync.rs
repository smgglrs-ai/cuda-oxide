/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ampere/Ada mma.sync warp-level matrix multiply-accumulate intrinsics.

use super::super::helpers::{emit_goto, insert_op};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::ops::{MirExtractArrayElementOp, MirLoadOp, MirStoreOp};
use dialect_nvvm::ops::MmaSyncM16N8K16F32F16Op;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::value::Value;
use rustc_public::mir;

fn create_usize_const(
    ctx: &mut Context,
    value: usize,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    let usize_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let const_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstantOp::get_concrete_op_info(),
        vec![usize_ty.into()],
        vec![],
        vec![],
        0,
    );
    const_op.deref_mut(ctx).set_loc(loc);
    let const_wrapped = dialect_mir::ops::MirConstantOp::new(const_op);
    let apint = pliron::utils::apint::APInt::from_i64(
        value as i64,
        std::num::NonZeroUsize::new(64).unwrap(),
    );
    const_wrapped.set_attr_value(
        ctx,
        pliron::builtin::attributes::IntegerAttr::new(usize_ty, apint),
    );
    insert_op(ctx, const_op, block_ptr, prev_op);
    let val = const_op.deref(ctx).get_result(0);
    (val, const_op)
}

fn extract_array_element(
    ctx: &mut Context,
    array_val: Value,
    index: usize,
    elem_ty: Ptr<pliron::r#type::TypeObj>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Ptr<Operation>,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    let (idx_val, idx_op) =
        create_usize_const(ctx, index, block_ptr, Some(prev_op), loc.clone());

    let extract_op = Operation::new(
        ctx,
        MirExtractArrayElementOp::get_concrete_op_info(),
        vec![elem_ty],
        vec![array_val, idx_val],
        vec![],
        0,
    );
    extract_op.deref_mut(ctx).set_loc(loc);
    extract_op.insert_after(ctx, idx_op);
    let val = extract_op.deref(ctx).get_result(0);
    (val, extract_op)
}

/// Emit mma_sync_m16n8k16_f32_f16: warp-level MMA via LLVM intrinsic.
///
/// Args:
/// - `args[0]`: &mut [f32; 4] (acc — both C input and D output)
/// - `args[1]`: &[u32; 4] (a — A fragments, packed f16×2)
/// - `args[2]`: &[u32; 2] (b — B fragments, packed f16×2)
///
/// The handler loads whole arrays from pointers, extracts individual elements,
/// creates the MmaSyncM16N8K16F32F16Op with 10 inputs producing 4 results,
/// constructs a new array with the results, and stores it back into acc.
pub fn emit_mma_sync_m16n8k16_f32_f16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_sync_m16n8k16_f32_f16 expects 3 arguments (acc, a, b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (acc_ptr, last_op_after) = rvalue::translate_operand(
        ctx, body, &args[0], value_map, block_ptr, last_op, loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_ptr, last_op_after) = rvalue::translate_operand(
        ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_ptr, last_op_after) = rvalue::translate_operand(
        ctx, body, &args[2], value_map, block_ptr, last_op, loc.clone(),
    )?;
    last_op = last_op_after;

    let u32_ty: Ptr<pliron::r#type::TypeObj> =
        IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let f32_ty: Ptr<pliron::r#type::TypeObj> = FP32Type::get(ctx).into();
    let u32_arr4_ty: Ptr<pliron::r#type::TypeObj> =
        dialect_mir::types::MirArrayType::get(ctx, u32_ty, 4).into();
    let u32_arr2_ty: Ptr<pliron::r#type::TypeObj> =
        dialect_mir::types::MirArrayType::get(ctx, u32_ty, 2).into();
    let f32_arr4_ty: Ptr<pliron::r#type::TypeObj> =
        dialect_mir::types::MirArrayType::get(ctx, f32_ty, 4).into();

    // Load whole arrays from pointers
    // Load a: [u32; 4]
    let load_a = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![u32_arr4_ty],
        vec![a_ptr],
        vec![],
        0,
    );
    load_a.deref_mut(ctx).set_loc(loc.clone());
    insert_op(ctx, load_a, block_ptr, last_op);
    let a_arr = load_a.deref(ctx).get_result(0);

    // Load b: [u32; 2]
    let load_b = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![u32_arr2_ty],
        vec![b_ptr],
        vec![],
        0,
    );
    load_b.deref_mut(ctx).set_loc(loc.clone());
    load_b.insert_after(ctx, load_a);
    let b_arr = load_b.deref(ctx).get_result(0);

    // Load acc: [f32; 4] (C input)
    let load_acc = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![f32_arr4_ty],
        vec![acc_ptr],
        vec![],
        0,
    );
    load_acc.deref_mut(ctx).set_loc(loc.clone());
    load_acc.insert_after(ctx, load_b);
    let acc_arr = load_acc.deref(ctx).get_result(0);

    // Extract individual elements from arrays
    let mut last = load_acc;

    // a[0..3]
    let mut a_vals = Vec::with_capacity(4);
    for i in 0..4 {
        let (val, op) = extract_array_element(ctx, a_arr, i, u32_ty, block_ptr, last, loc.clone());
        a_vals.push(val);
        last = op;
    }

    // b[0..1]
    let mut b_vals = Vec::with_capacity(2);
    for i in 0..2 {
        let (val, op) = extract_array_element(ctx, b_arr, i, u32_ty, block_ptr, last, loc.clone());
        b_vals.push(val);
        last = op;
    }

    // acc[0..3] (C inputs)
    let mut c_vals = Vec::with_capacity(4);
    for i in 0..4 {
        let (val, op) =
            extract_array_element(ctx, acc_arr, i, f32_ty, block_ptr, last, loc.clone());
        c_vals.push(val);
        last = op;
    }

    // Create the MmaSyncOp with 10 operands: [a0,a1,a2,a3, b0,b1, c0,c1,c2,c3]
    let operands = vec![
        a_vals[0], a_vals[1], a_vals[2], a_vals[3],
        b_vals[0], b_vals[1],
        c_vals[0], c_vals[1], c_vals[2], c_vals[3],
    ];

    let result_types: Vec<Ptr<pliron::r#type::TypeObj>> = (0..4).map(|_| f32_ty).collect();

    let mma_op = Operation::new(
        ctx,
        MmaSyncM16N8K16F32F16Op::get_concrete_op_info(),
        result_types,
        operands,
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());
    mma_op.insert_after(ctx, last);

    // Get the 4 f32 results (d0-d3)
    let d_vals: Vec<Value> = (0..4).map(|i| mma_op.deref(ctx).get_result(i)).collect();

    // Construct new [f32; 4] array from results
    let arr_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructArrayOp::get_concrete_op_info(),
        vec![f32_arr4_ty],
        d_vals,
        vec![],
        0,
    );
    arr_op.deref_mut(ctx).set_loc(loc.clone());
    arr_op.insert_after(ctx, mma_op);
    let result_arr = arr_op.deref(ctx).get_result(0);

    // Store result array back to acc pointer
    let store_op = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![acc_ptr, result_arr],
        vec![],
        0,
    );
    store_op.deref_mut(ctx).set_loc(loc.clone());
    store_op.insert_after(ctx, arr_op);

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, store_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "mma_sync_m16n8k16_f32_f16 call without target block".to_string()
            )
        )
    }
}
