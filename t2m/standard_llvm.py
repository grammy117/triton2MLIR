"""Retain the standard LLVM volatile-load field missing from pinned xDSL."""

from xdsl.dialects import builtin, llvm
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, opt_prop_def, prop_def, result_def


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "llvm.load"
    ptr = operand_def(llvm.LLVMPointerType)
    dereferenced_value = result_def()
    alignment = opt_prop_def(builtin.IntegerAttr)
    ordering = prop_def(builtin.IntegerAttr, default_value=builtin.IntegerAttr(0, builtin.i64))
    volatile_ = opt_prop_def(builtin.UnitAttr)

    def __init__(self, pointer, result_type):
        super().__init__(operands=[pointer], result_types=[result_type])


def llvm_memory_dialect():
    # Only the portable memory leaf is admitted here. No inline assembly,
    # target intrinsic, physical address-space or arbitrary LLVM program.
    return Dialect("llvm", [LoadOp, llvm.IntToPtrOp], [llvm.LLVMPointerType])
