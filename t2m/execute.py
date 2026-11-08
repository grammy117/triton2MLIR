"""Reference execution of the emitted standard-op subset, independent of provenance.

This is an IR interpreter, not a claim of LLVM or accelerator execution. Forall is
sequentially realized by the interpreter after conversion proved independence.
"""

import numpy as np
from xdsl.dialects import arith, func, tensor
from xdsl.dialects.linalg.ops import FillOp, GenericOp, YieldOp
from xdsl.dialects.builtin import DYNAMIC_INDEX
from xdsl.ir.affine import AffineMap

from .ir import parse
from .standard_ops import ForallOp, InParallelOp, ParallelInsertSliceOp


def execute(mlir_text, inputs):
    module = parse(mlir_text)
    functions = [op for op in module.body.block.ops if isinstance(op, func.FuncOp)]
    if len(functions) != 1:
        raise ValueError("interpreter requires one function")
    entry = functions[0].body.block
    if len(inputs) != len(entry.args):
        raise ValueError("input count differs from function ABI")
    arrays = [np.asarray(value) for value in inputs]
    for arg, array in zip(entry.args, arrays, strict=True):
        if array.dtype != np.float32 or tuple(array.shape) != arg.type.get_shape():
            raise ValueError("input must match declared f32 tensor shape exactly")
    if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1:]):
        raise ValueError("independent-allocation precondition violated")

    def run(block, arguments, outer):
        env = dict(outer)
        env.update(zip(block.args, arguments, strict=True))
        for op in block.ops:
            args = [env[arg] for arg in op.operands]
            results = ()
            if isinstance(op, (func.ReturnOp, YieldOp, tensor.YieldOp)):
                return tuple(args)
            if isinstance(op, arith.ConstantOp):
                results = (op.value.value.data,)
            elif isinstance(op, (arith.MuliOp, arith.AddiOp, arith.SubiOp, arith.MinSIOp, arith.MaxSIOp)):
                fn = {arith.MuliOp: lambda a, b: a * b, arith.AddiOp: lambda a, b: a + b,
                      arith.SubiOp: lambda a, b: a - b, arith.MinSIOp: min, arith.MaxSIOp: max}[type(op)]
                results = (fn(*args),)
            elif isinstance(op, (arith.AddfOp, arith.SubfOp, arith.MulfOp)):
                fn = {arith.AddfOp: np.add, arith.SubfOp: np.subtract, arith.MulfOp: np.multiply}[type(op)]
                results = (fn(args[0], args[1], dtype=np.float32),)
            elif isinstance(op, tensor.EmptyOp):
                results = (np.full(op.tensor.type.get_shape(), np.nan, dtype=np.float32),)
            elif isinstance(op, tensor.ExtractSliceOp):
                dynamic = iter(args[1:])
                off = next(dynamic) if op.static_offsets.get_values()[0] == DYNAMIC_INDEX else op.static_offsets.get_values()[0]
                length = next(dynamic) if op.static_sizes.get_values()[0] == DYNAMIC_INDEX else op.static_sizes.get_values()[0]
                if off < 0 or length < 0 or off + length > len(args[0]):
                    raise ValueError("interpreter encountered an out-of-bounds slice")
                results = (args[0][off:off + length].copy(),)
            elif isinstance(op, tensor.PadOp):
                dynamic = iter(args[1:])
                low = next(dynamic) if op.static_low.get_values()[0] == DYNAMIC_INDEX else op.static_low.get_values()[0]
                high = next(dynamic) if op.static_high.get_values()[0] == DYNAMIC_INDEX else op.static_high.get_values()[0]
                if low < 0 or high < 0:
                    raise ValueError("negative tensor padding")
                result = np.empty(low + len(args[0]) + high, dtype=np.float32)
                result[low:low + len(args[0])] = args[0]
                for index in (*range(low), *range(low + len(args[0]), len(result))):
                    result[index], = run(op.region.block, (index,), env)
                results = (result,)
            elif isinstance(op, tensor.CastOp):
                shape = op.results[0].type.get_shape()
                if any(expected != DYNAMIC_INDEX and expected != actual for expected, actual in zip(shape, args[0].shape, strict=True)):
                    raise ValueError("tensor.cast shape does not match the actual padded tile")
                results = (args[0],)
            elif isinstance(op, FillOp):
                results = (np.full(args[-1].shape, args[0], dtype=np.float32),)
            elif isinstance(op, GenericOp):
                if any(mapping.data != AffineMap.identity(1) for mapping in op.indexing_maps):
                    raise ValueError("interpreter supports identity indexing maps only")
                results = run(op.body.block, args, env)
            elif isinstance(op, ForallOp):
                output = args[0].copy()
                for i in range(op.staticUpperBound.get_values()[0]):
                    run(op.body.block, (i, output), env)
                results = (output,)
            elif isinstance(op, InParallelOp):
                run(op.body.block, (), env)
                return ()
            elif isinstance(op, ParallelInsertSliceOp):
                source, dest, off, *dynamic_sizes = args
                length = (dynamic_sizes[0] if op.static_sizes.get_values()[0] == DYNAMIC_INDEX
                          else op.static_sizes.get_values()[0])
                if off < 0 or length != len(source) or off + length > len(dest):
                    raise ValueError("interpreter encountered an out-of-bounds insertion")
                dest[off:off + len(source)] = source
            else:
                raise ValueError(f"interpreter does not implement {op.name}")
            env.update(zip(op.results, results, strict=True))
        return ()
    return run(entry, arrays, {})
