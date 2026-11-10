"""CAS through official structured pointers and standard atomic regions.

Reuse Meta 17f0c234's rank-one gather descriptor, retaining the official source
tt.atomic_cas until expansion (the pinned binary does not register tts.atomic_cas).
The parallel tensor event follows its UnstructuredAtomicRMWConverter pattern.
LLVM a47bddc's MemRefOps.td/GenericAtomicRMWOpLowering supplies the atomic region
semantics and cmpxchg continuation. No custom operation reaches the consumer.

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from math import prod

from xdsl.dialects import arith, builtin, memref, scf, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .standard_memref import AtomicYieldOp, GenericAtomicRMWOp


def dispatch(op, roots, offset, selector, *, pointer_type=None):
    """Select logical bases with scalar branches inside the original parallel tile."""
    type_ = op.results[0].type
    tiled = isinstance(type_, builtin.TensorType)
    element = type_.element_type if tiled else type_
    operations = []
    def raw(name, operands, result_types, attributes=None):
        return builtin.UnregisteredOp.with_name(name).create(operands=operands, result_types=result_types,
            properties={key: op.properties.get(key, op.attributes.get(key)) for key in ("sem", "scope")} if name == "tt.atomic_cas" else {},
            attributes=attributes or {}, location=op.location)
    if tiled:
        if not isinstance(selector.type, builtin.TensorType):
            splat = raw("tt.splat", [selector], [builtin.TensorType(builtin.i32, type_.get_shape())])
            operations.append(splat)
            selector = splat.results[0]
        inputs = [offset, selector, *op.operands[1:]]
        empty = tensor.EmptyOp([], type_)
        operations.append(empty)
        body = Block(arg_types=[v.type.element_type for v in inputs] + [element])
        local_offset, local_selector, comparison, value = body.args[:4]
    else:
        body = Block()
        local_offset, local_selector, comparison, value = offset, selector, *op.operands[1:]
    def atomic_on(block, ordinal):
        base = roots[ordinal]
        scalar = pointer_type.element_type if isinstance(pointer_type, builtin.TensorType) else pointer_type
        viewed = base
        if scalar is not None and scalar != base.type:
            cast = raw("tt.bitcast", [base], [scalar])
            block.add_op(cast)
            viewed = cast.results[0]
        pointer = raw("tt.addptr", [viewed, local_offset], [viewed.type])
        atomic = raw("tt.atomic_cas", [pointer.results[0], comparison, value], [element],
                     {"prov.pointer_base_ordinal": builtin.IntegerAttr(base.index, builtin.i32)})
        block.add_ops([pointer, atomic])
        return atomic.results[0]
    def select(block, ordinal):
        base = roots[ordinal]
        if ordinal == len(roots) - 1:
            return atomic_on(block, ordinal)
        constant = arith.ConstantOp.from_int_and_width(base.index, builtin.i32)
        equal = arith.CmpiOp(local_selector, constant.result, "eq")
        yes, no = Block(), Block()
        previous = atomic_on(yes, ordinal)
        yes.add_op(scf.YieldOp(previous))
        no.add_op(scf.YieldOp(select(no, ordinal + 1)))
        branch = scf.IfOp(equal.result, [element], Region(yes), Region(no))
        block.add_ops([constant, equal, branch])
        return branch.results[0]
    result = select(body, 0)
    if tiled:
        body.add_op(YieldOp(result))
        rank = len(type_.get_shape())
        identity = builtin.AffineMapAttr(AffineMap.identity(rank))
        generic = GenericOp(inputs, [empty.tensor], Region(body), [identity] * 5,
                            [IteratorTypeAttr.parallel()] * rank, [type_])
        operations.append(generic)
        result = generic.results[0]
    else:
        for child in list(body.ops):
            body.detach_op(child)
            operations.append(child)
    for outer in operations:
        for nested in outer.walk():
            nested.location = op.location
    Rewriter.replace_op(op, operations, [result])


def template(body, op, arguments):
    """Normalize one scalar/rank-N source event to an official rank-one CAS."""
    base, offset, comparison, value = arguments
    type_ = op.results[0].type
    tiled = isinstance(type_, builtin.TensorType)
    shape = type_.get_shape() if tiled else ()
    count = prod(shape)
    def emit(name, operands, result_types, properties=None):
        generated = builtin.UnregisteredOp.with_name(name).create(operands=operands, result_types=result_types,
            properties=properties or {}, location=op.location)
        body.add_op(generated)
        return generated.results[0]
    def flat(v):
        if not tiled:
            wrapped = tensor.FromElementsOp(v)
            wrapped.location = op.location
            body.add_op(wrapped)
            return wrapped.result
        if len(shape) == 1:
            return v
        return emit("tt.reshape", [v], [builtin.TensorType(v.type.element_type, [count])])
    offset, comparison, value = map(flat, (offset, comparison, value))
    properties = {"sizes": builtin.DenseArrayBase.from_list(builtin.i64, [count]),
        "gather_scatter_dim": builtin.IntegerAttr(0, builtin.i32),
        "static_strides": builtin.DenseArrayBase.from_list(builtin.i64, [1]),
        "static_offsets": builtin.DenseArrayBase.from_list(builtin.i64, [0]),
        "operandSegmentSizes": builtin.DenseArrayBase.from_list(builtin.i32, [1, 1, 0, 0, 0])}
    pointer = emit("tts.make_gather_scatter_tptr", [base, offset], [builtin.TensorType(base.type, [count])], properties)
    settings = {key: op.properties.get(key, op.attributes.get(key)) for key in ("sem", "scope")}
    result = emit("tt.atomic_cas", [pointer, comparison, value], [comparison.type], settings)
    if not tiled:
        zero = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        extracted = tensor.ExtractOp(result, [zero.result], type_)
        for generated in (zero, extracted):
            generated.location = op.location
            body.add_op(generated)
        return extracted.result
    if len(shape) != 1:
        result = emit("tt.reshape", [result], [type_])
    return result


def expand(module, source_ids):
    """Expand admitted official CAS events before the native memory continuation."""
    from .shared_tensor import _array, opcode
    reports = []
    for op in list(module.walk()):
        if opcode(op) != "tt.atomic_cas":
            continue
        descriptor = op.operands[0].owner
        if (not hasattr(descriptor, "operands") or opcode(descriptor) != "tts.make_gather_scatter_tptr"
                or _array(descriptor, "operandSegmentSizes") != (1, 1, 0, 0, 0)
                or _array(descriptor, "static_strides") != (1,) or _array(descriptor, "static_offsets") != (0,)):
            raise ConversionError("CAS requires its exact normalized rank-one logical index descriptor", opcode=opcode(op))
        base, offset = descriptor.operands
        comparison, value = op.operands[1:]
        type_ = op.results[0].type
        element = type_.element_type
        storage, word_view = element, None
        viewed_base = base.owner
        if hasattr(viewed_base, "operands") and opcode(viewed_base) == "tt.bitcast":
            from .pointer_views import element as pointer_element
            from .atomic_word_views import view_info
            base, = viewed_base.operands
            storage = pointer_element(base.type)
            word_view = view_info(storage, element)
        cast = builtin.UnrealizedConversionCastOp.get([base], [builtin.UnrankedMemRefType.from_type(storage)])
        ranked = memref.CastOp.get(cast.results[0], builtin.MemRefType(storage, [builtin.DYNAMIC_INDEX]))
        empty = tensor.EmptyOp([], type_)
        block = Block(arg_types=[offset.type.element_type, element, element, element])
        index = arith.IndexCastOp(block.args[0], builtin.IndexType())
        atomic_body = Block(arg_types=[storage])
        current, = atomic_body.args
        if word_view:
            viewed = arith.BitcastOp(current, element)
            atomic_body.add_op(viewed)
            current = viewed.result
        equal = arith.CmpiOp(current, block.args[1], "eq")
        selected = arith.SelectOp(equal.result, block.args[2], current)
        atomic_body.add_ops([equal, selected])
        updated = selected.result
        if word_view:
            stored = arith.BitcastOp(updated, storage)
            atomic_body.add_op(stored)
            updated = stored.result
        atomic_body.add_op(AtomicYieldOp.create(operands=[updated]))
        atomic = GenericAtomicRMWOp.create(operands=[ranked.dest, index.result], result_types=[storage], regions=[Region(atomic_body)])
        block.add_ops([index, atomic])
        returned = atomic.result
        if word_view:
            old_word = arith.BitcastOp(returned, element)
            block.add_op(old_word)
            returned = old_word.result
        block.add_op(YieldOp(returned))
        identity = builtin.AffineMapAttr(AffineMap.identity(1))
        generic = GenericOp([offset, comparison, value], [empty.tensor], Region(block), [identity] * 4,
                            [IteratorTypeAttr.parallel()], [type_])
        generated = [cast, ranked, empty, generic]
        for outer in generated:
            for nested in outer.walk():
                nested.location = op.location
        reports.append({"kind": "standard-compare-exchange-region", "source_ids": source_ids(op.location),
            "comparison": "integer-equality", "success_value": "original-source-value", "failure_value": "unchanged-current-word",
            "result": "original-atomic-old-value", "iteration": "parallel-logical-tile",
            **({"word_view": word_view} if word_view else {})})
        Rewriter.replace_op(op, generated, generic.results)
        if not descriptor.results[0].uses:
            descriptor.parent.erase_op(descriptor)
    return reports
