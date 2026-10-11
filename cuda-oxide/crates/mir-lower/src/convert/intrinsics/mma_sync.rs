/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ampere/Ada mma.sync intrinsic conversion to LLVM IR.
//!
//! Converts `nvvm.mma_sync_m16n8k16_f32_f16` to the LLVM intrinsic
//! `@llvm.nvvm.mma.m16n8k16.row.col.f32.f32`.
//!
//! The intrinsic expects `<2 x half>` for A/B operands (bitcast from i32)
//! and returns `{ float, float, float, float }`.

use crate::convert::intrinsics::common::*;
use llvm_export::op_interfaces::CastOpInterface;
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP16Type, FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeObj;

/// Convert `nvvm.mma_sync_m16n8k16_f32_f16` to the LLVM intrinsic call.
///
/// The op has 10 operands: [a0,a1,a2,a3, b0,b1, c0,c1,c2,c3]
/// where a0-a3, b0-b1 are i32 (packed f16x2) and c0-c3 are f32.
///
/// Steps:
/// 1. Bitcast a0-a3, b0-b1 from i32 to <2 x half>
/// 2. Declare the LLVM intrinsic function
/// 3. Call with 10 args: 4x v2f16, 2x v2f16, 4x float
/// 4. Extract 4 floats from returned struct
pub(crate) fn convert_mma_sync_m16n8k16_f32_f16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 10 {
        return pliron::input_err_noloc!("mma_sync_m16n8k16_f32_f16 requires 10 operands");
    }

    let a0 = operands[0];
    let a1 = operands[1];
    let a2 = operands[2];
    let a3 = operands[3];
    let b0 = operands[4];
    let b1 = operands[5];
    let c0 = operands[6];
    let c1 = operands[7];
    let c2 = operands[8];
    let c3 = operands[9];

    let half_ty = FP16Type::get(ctx);
    let v2f16_ty: Ptr<TypeObj> =
        llvm_types::VectorType::get(ctx, half_ty.into(), 2, llvm_types::VectorTypeKind::Fixed)
            .into();
    let f32_ty = FP32Type::get(ctx);

    // Bitcast A operands: i32 -> <2 x half>
    let a0_v = llvm::BitcastOp::new(ctx, a0, v2f16_ty);
    rewriter.insert_operation(ctx, a0_v.get_operation());
    let a0_val = a0_v.get_operation().deref(ctx).get_result(0);

    let a1_v = llvm::BitcastOp::new(ctx, a1, v2f16_ty);
    rewriter.insert_operation(ctx, a1_v.get_operation());
    let a1_val = a1_v.get_operation().deref(ctx).get_result(0);

    let a2_v = llvm::BitcastOp::new(ctx, a2, v2f16_ty);
    rewriter.insert_operation(ctx, a2_v.get_operation());
    let a2_val = a2_v.get_operation().deref(ctx).get_result(0);

    let a3_v = llvm::BitcastOp::new(ctx, a3, v2f16_ty);
    rewriter.insert_operation(ctx, a3_v.get_operation());
    let a3_val = a3_v.get_operation().deref(ctx).get_result(0);

    // Bitcast B operands: i32 -> <2 x half>
    let b0_v = llvm::BitcastOp::new(ctx, b0, v2f16_ty);
    rewriter.insert_operation(ctx, b0_v.get_operation());
    let b0_val = b0_v.get_operation().deref(ctx).get_result(0);

    let b1_v = llvm::BitcastOp::new(ctx, b1, v2f16_ty);
    rewriter.insert_operation(ctx, b1_v.get_operation());
    let b1_val = b1_v.get_operation().deref(ctx).get_result(0);

    // Build the intrinsic function type:
    //   (<2 x half>, <2 x half>, <2 x half>, <2 x half>,
    //    <2 x half>, <2 x half>,
    //    float, float, float, float) -> { float, float, float, float }
    let result_field_types: Vec<Ptr<TypeObj>> = (0..4).map(|_| f32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, result_field_types);

    let param_types: Vec<Ptr<TypeObj>> = vec![
        v2f16_ty, v2f16_ty, v2f16_ty, v2f16_ty, // A0-A3
        v2f16_ty, v2f16_ty, // B0-B1
        f32_ty.into(),
        f32_ty.into(),
        f32_ty.into(),
        f32_ty.into(), // C0-C3
    ];

    let func_ty = llvm_types::FuncType::get(ctx, struct_ty.into(), param_types, false);

    let intrinsic_name = "llvm_nvvm_mma_m16n8k16_row_col_f32_f32";

    let call_args = vec![
        a0_val, a1_val, a2_val, a3_val, b0_val, b1_val, c0, c1, c2, c3,
    ];

    let call_op = call_intrinsic(ctx, rewriter, op, intrinsic_name, func_ty, call_args)?;

    // Extract 4 float results from the struct
    let struct_result = call_op.deref(ctx).get_result(0);
    let mut extracted_values = Vec::with_capacity(4);
    for i in 0..4u32 {
        let extract_op = llvm::ExtractValueOp::new(ctx, struct_result, vec![i])
            .map_err(|e| pliron::input_error_noloc!("{}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        let field_val = extract_op.get_operation().deref(ctx).get_result(0);
        extracted_values.push(field_val);
    }

    rewriter.replace_operation_with_values(ctx, op, extracted_values);
    Ok(())
}
