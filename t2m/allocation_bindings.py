"""Bind source pointer arguments to allocation state with executable TTIR calls.

The upstream inliner substitutes the call operands before memory analysis. All
aliased observations therefore use one root, including its guards and versions.
No physical overlap or provenance annotation supplies rewrite authority.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path

from xdsl.dialects import arith, builtin
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from .diagnostics import ConversionError
from .shared_tensor import convert_shared, opcode, parse_shared
from .upstream import inspect_optimizer, lower_with_shared

PROFILE = "t2m.source-allocation-bindings.v1"
ARGUMENTS_SCHEMA = "t2m.source-allocation-arguments.v1"


@dataclass(frozen=True)
class PointerArgument:
    allocation: str
    offset: int | str = 0

    def __post_init__(self):
        if type(self.allocation) is not str or not self.allocation.isidentifier():
            raise ValueError("pointer allocation requires a declared identifier")


@dataclass(frozen=True)
class ScalarArgument:
    name: str

    def __post_init__(self):
        if type(self.name) is not str or not self.name.isidentifier():
            raise ValueError("scalar argument requires a declared identifier")


def arguments_from_dict(value):
    """Read the strict source-order argument configuration used by the CLI."""
    if (not isinstance(value,dict) or set(value) != {"schema","source_entry","arguments"}
            or value["schema"] != ARGUMENTS_SCHEMA or type(value["source_entry"]) is not str
            or not value["source_entry"] or not isinstance(value["arguments"],list)):
        raise ValueError("source allocation arguments require their explicit schema, entry and complete roster")
    bindings = []
    for argument in value["arguments"]:
        if not isinstance(argument,dict) or "kind" not in argument:
            raise ValueError("source allocation arguments must be typed records")
        kind = argument.get("kind")
        if kind == "pointer" and set(argument) in ({"kind","allocation"},{"kind","allocation","offset"}):
            binding = PointerArgument(argument["allocation"],argument.get("offset",0))
        elif kind == "scalar" and set(argument) == {"kind","name"}:
            binding = ScalarArgument(argument["name"])
        elif kind == "constant" and set(argument) == {"kind","value"} and type(argument["value"]) is int:
            binding = argument["value"]
        else:
            raise ValueError("unknown or incomplete source allocation argument record")
        bindings.append(binding)
    return value["source_entry"],tuple(bindings)


def bind_module(module, source_entry, abi, bindings):
    """Return a cloned source module with an explicit public allocation entry."""
    module = module.clone()
    functions = list(module.ops)
    if any(opcode(op) != "tt.func" for op in functions):
        raise ConversionError("allocation binding requires a TTIR function module")
    names = [op.properties["sym_name"].data for op in functions]
    if names.count(source_entry) != 1 or abi.entry in names:
        raise ConversionError("allocation entry must be fresh and source entry unique")
    source = functions[names.index(source_entry)]
    if (len(source.regions[0].blocks) != 1 or source.properties["function_type"].outputs
            or len(bindings) != len(source.regions[0].block.args)):
        raise ConversionError("allocation binding requires the complete void source signature")
    if any(op is not source and op.properties.get("sym_visibility") != builtin.StringAttr("private")
           for op in functions):
        raise ConversionError("allocation binding cannot hide another public source entry")
    if abi.scalar_constants:
        raise ConversionError("use explicit source constant bindings instead of unnamed wrapper constants")
    roots = {buffer.name:buffer for buffer in abi.buffers}
    scalars = {scalar.name:scalar for scalar in abi.runtime_scalars}
    pointer_types, records = {}, []
    arguments = source.regions[0].block.args
    for arg,binding in zip(arguments,bindings,strict=True):
        if isinstance(binding,PointerArgument):
            root = roots.get(binding.allocation)
            if root is None or str(arg.type) != f"!tt.ptr<{root.dtype}>":
                raise ConversionError("source pointer binding requires a declared allocation of the same element type")
            if type(binding.offset) is int:
                if not -(2**63) <= binding.offset < 2**63:
                    raise ConversionError("source allocation origin must fit signed i64")
            elif (type(binding.offset) is not str or binding.offset not in scalars
                  or scalars[binding.offset].dtype not in ("i32","i64")):
                raise ConversionError("source allocation origin requires an integer or declared i32/i64 scalar")
            pointer_types[root.name] = arg.type
            records.append({"kind":"pointer",**asdict(binding)})
        elif isinstance(binding,ScalarArgument):
            scalar = scalars.get(binding.name)
            if scalar is None or arg.type != scalar.type:
                raise ConversionError("source scalar binding requires its exact declared wrapper type")
            records.append({"kind":"scalar",**asdict(binding)})
        elif type(binding) is int and isinstance(arg.type,builtin.IntegerType):
            width = arg.type.width.data
            if not (binding in (0,1) if width == 1 else -(2**(width-1)) <= binding < 2**(width-1)):
                raise ConversionError("specialized source scalar does not fit its signed source type")
            records.append({"kind":"constant","value":binding,"type":str(arg.type)})
        else:
            raise ConversionError("source argument needs an explicit typed allocation/scalar binding")
    if set(pointer_types) != set(roots):
        raise ConversionError("every declared allocation must bind at least one source pointer")
    types = [pointer_types[value.name] if kind == "buffer" else value.type for kind,value in abi.source_bindings]
    block = Block(arg_types=types)
    values = {value.name:arg for arg,(kind,value) in zip(block.args,abi.source_bindings,strict=True)}
    operands = []
    for arg,binding in zip(arguments,bindings,strict=True):
        if isinstance(binding,PointerArgument):
            value = values[binding.allocation]
            if binding.offset != 0:
                if type(binding.offset) is int:
                    constant = arith.ConstantOp.from_int_and_width(binding.offset,builtin.i64)
                    block.add_op(constant)
                    offset = constant.result
                else:
                    offset = values[binding.offset]
                address = builtin.UnregisteredOp.with_name("tt.addptr").create(
                    operands=[value,offset],result_types=[arg.type])
                block.add_op(address)
                value = address.results[0]
            operands.append(value)
        elif isinstance(binding,ScalarArgument):
            operands.append(values[binding.name])
        else:
            constant = arith.ConstantOp.from_int_and_width(binding,arg.type)
            block.add_op(constant)
            operands.append(constant.result)
    block.add_op(builtin.UnregisteredOp.with_name("tt.call").create(operands=operands,
        properties={"callee":builtin.SymbolRefAttr(source_entry)}))
    block.add_op(builtin.UnregisteredOp.with_name("tt.return").create())
    report = {"schema":PROFILE,"source_entry":source_entry,"allocation_entry":abi.entry,
              "bindings":records,"grants_semantics":False,
              "representation":"executable-tt-call-before-upstream-inlining"}
    wrapper = builtin.UnregisteredOp.with_name("tt.func").create(
        properties={"sym_name":builtin.StringAttr(abi.entry),"sym_visibility":builtin.StringAttr("public"),
                    "function_type":builtin.FunctionType.from_lists(types,[])},
        attributes={"prov.family":builtin.StringAttr("abi"),"prov.op":builtin.StringAttr("source-allocation-binding"),
                    "prov.allocation_binding":builtin.StringAttr(json.dumps(report,sort_keys=True))},
        regions=[Region(block)])
    source.properties["sym_visibility"] = builtin.StringAttr("private")
    module.body.block.insert_op_before(wrapper,functions[0])
    # Native Triton/shared verifies the serialized wrapper before conversion.
    # xDSL's scalar-only arith constraints cannot verify raw TTIR tensor casts.
    return module,report


def convert_allocations(source, abi, bindings, work, *, source_entry, optimizer=None, **options):
    """Compile a source kernel with an explicit allocation-state calling ABI.

    The package's source is the bound TTIR program, including the private source
    body. The original source bytes and binding report remain in this work area.
    """
    work = Path(work)
    work.mkdir(parents=True,exist_ok=False)
    optimizer = inspect_optimizer(optimizer)
    original = Path(source).read_bytes()
    saved = work / "original.ttir"
    saved.write_bytes(original)
    lower_with_shared(saved,work / "source",optimizer=optimizer,stage="source")
    module = parse_shared((work / "source/lowered.mlir").read_text())
    module,report = bind_module(module,source_entry,abi,bindings)
    stream = StringIO()
    Printer(stream=stream,print_generic_format=True,print_debuginfo=True).print_op(module)
    bound = work / "bound.ttir"
    bound.write_text(stream.getvalue()+"\n")
    report.update(original_source_sha256=hashlib.sha256(original).hexdigest(),
                  bound_source_sha256=hashlib.sha256(bound.read_bytes()).hexdigest())
    (work / "bindings.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    result = convert_shared(bound,abi,work / "translate",optimizer=optimizer,**options)
    result.trace["source_argument_binding"] = report
    result.trace["abi_difference"] += "; explicit allocation-state entry binds source pointers before inlining; updated roots returned once"
    return result
