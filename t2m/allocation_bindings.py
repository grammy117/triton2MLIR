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

from .descriptor_bindings import DescriptorArgument, ExtendScalar
from .diagnostics import ConversionError
from .shared_tensor import SharedParser, convert_shared, opcode, parse_shared
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
            or value["schema"] not in (ARGUMENTS_SCHEMA, "t2m.source-allocation-arguments.v2") or type(value["source_entry"]) is not str
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
        elif (kind == "descriptor" and value["schema"] == "t2m.source-allocation-arguments.v2"
              and {"kind", "allocation", "shape", "strides"} <= set(argument)
              and set(argument) <= {"kind", "allocation", "shape", "strides", "offset", "padding_nan", "round_f32_to_tf32"}):
            binding = DescriptorArgument(**{key: item for key, item in argument.items() if key != "kind"})
        else:
            raise ValueError("unknown or incomplete source allocation argument record")
        bindings.append(binding)
    if (value["schema"] == "t2m.source-allocation-arguments.v2") != any(isinstance(binding, DescriptorArgument) for binding in bindings):
        raise ValueError("descriptor source arguments require their explicit v2 schema")
    return value["source_entry"],tuple(bindings)


def arguments_as_dict(source_entry, bindings):
    records = []
    for binding in bindings:
        if isinstance(binding, (PointerArgument, ScalarArgument, DescriptorArgument)):
            kind = "pointer" if isinstance(binding, PointerArgument) else "scalar" if isinstance(binding, ScalarArgument) else "descriptor"
            records.append({"kind": kind, **asdict(binding)})
        elif type(binding) is int:
            records.append({"kind": "constant", "value": binding})
        else:
            raise ValueError("unknown original source allocation argument")
    return {"schema": "t2m.source-allocation-arguments.v2" if any(isinstance(binding, DescriptorArgument) for binding in bindings) else ARGUMENTS_SCHEMA,
            "source_entry": source_entry, "arguments": records}


def bind_module(module, source_entry, abi, bindings, *, byte_order=None):
    """Return a cloned source module with an explicit public allocation entry."""
    if byte_order is not None and byte_order not in ("little", "big"):
        raise ValueError("byte-buffer bindings require explicit little or big byte order")
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
    from .ir import context
    from .pointer_views import byte_compatible, compatible
    ctx = context()
    ctx.allow_unregistered = True
    pointer_types = {name: SharedParser(ctx, f"!tt.ptr<{root.dtype}>").parse_type()
                     for name, root in roots.items()}
    used_roots, records = set(), []
    arguments = source.regions[0].block.args
    for arg,binding in zip(arguments,bindings,strict=True):
        if isinstance(binding,PointerArgument):
            root = roots.get(binding.allocation)
            if root is None:
                raise ConversionError("source pointer binding requires a declared allocation")
            storage_type = pointer_types[root.name]
            viewed = str(arg.type) != str(storage_type)
            byte_view = (viewed and byte_order is not None and byte_compatible(arg.type, storage_type, typed_root=True))
            if viewed and (isinstance(arg.type, builtin.TensorType) or not compatible(arg.type, storage_type) and not byte_view):
                raise ConversionError("source pointer binding requires the same element type or an equal byte-aligned word view")
            if type(binding.offset) is int:
                if not -(2**63) <= binding.offset < 2**63:
                    raise ConversionError("source allocation origin must fit signed i64")
            elif (type(binding.offset) is not str or binding.offset not in scalars
                  or scalars[binding.offset].dtype not in ("i32","i64")):
                raise ConversionError("source allocation origin requires an integer or declared i32/i64 scalar")
            used_roots.add(root.name)
            records.append({"kind":"pointer",**asdict(binding), **({
                "source_type": str(arg.type), "storage_type": str(storage_type),
                "representation": ("byte-buffer-view-v1" if root.dtype == "i8" else "typed-root-byte-view-v1") if byte_view else "equal-width-word-view-v1",
                **({"byte_order": byte_order} if byte_view else {})} if viewed else {})})
        elif isinstance(binding,ScalarArgument):
            scalar = scalars.get(binding.name)
            if scalar is None or arg.type != scalar.type:
                raise ConversionError("source scalar binding requires its exact declared wrapper type")
            records.append({"kind":"scalar",**asdict(binding)})
        elif isinstance(binding, ExtendScalar):
            scalar = scalars.get(binding.name)
            if scalar is None or scalar.dtype != "i32" or arg.type != builtin.i64:
                raise ConversionError("descriptor field extension requires an i32 scalar and i64 source field")
            records.append({"kind": "sext", "name": binding.name, "from": "i32", "type": "i64"})
        elif type(binding) is int and isinstance(arg.type,builtin.IntegerType):
            width = arg.type.width.data
            if not (binding in (0,1) if width == 1 else -(2**(width-1)) <= binding < 2**(width-1)):
                raise ConversionError("specialized source scalar does not fit its signed source type")
            records.append({"kind":"constant","value":binding,"type":str(arg.type)})
        else:
            raise ConversionError("source argument needs an explicit typed allocation/scalar binding")
    if used_roots != set(roots):
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
                    operands=[value,offset],result_types=[value.type])
                block.add_op(address)
                value = address.results[0]
            if str(value.type) != str(arg.type):
                cast = builtin.UnregisteredOp.with_name("tt.bitcast").create(
                    operands=[value], result_types=[arg.type])
                block.add_op(cast)
                value = cast.results[0]
            operands.append(value)
        elif isinstance(binding,ScalarArgument):
            operands.append(values[binding.name])
        elif isinstance(binding, ExtendScalar):
            extended = arith.ExtSIOp(values[binding.name], builtin.i64)
            block.add_op(extended)
            operands.append(extended.result)
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

    The package source is the executable bound program. Its portable source
    capsule retains exact original, normalized and generic input bytes separately.
    """
    work = Path(work)
    work.mkdir(parents=True,exist_ok=False)
    optimizer = inspect_optimizer(optimizer)
    original = Path(source).read_bytes()
    saved = work / "original.ttir"
    saved.write_bytes(original)
    bindings = tuple(bindings)
    configuration = arguments_as_dict(source_entry, bindings)
    normalized, intake, inventory = None, None, None
    normalization_source, protected_artifacts = saved, {}
    if options.get("fp8_overflow") is not None or options.get("assembly_provider") is not None:
        from .source_protection import protect
        normalization_source, protected_artifacts, _ = protect(saved, work / "protection", source_entry=source_entry,
            **{key: options[key] for key in ("fp8_overflow", "effect_profile", "f32_atomic_add", "assembly_provider") if key in options})

    def normalize():
        from .frontend import expand_tensor_pointers
        report = expand_tensor_pointers(normalization_source, work / "intake")
        return work / "intake/lowered.mlir", report, json.loads((work / "intake/inventory.json").read_text())

    source_stage_dir = work / "source"
    if any(isinstance(binding, DescriptorArgument) for binding in bindings):
        normalized, intake, inventory = normalize()
    try:
        source_stage = lower_with_shared(normalized or normalization_source, source_stage_dir, optimizer=optimizer, stage="source")
    except ConversionError as original_error:
        if normalized is not None:
            raise
        try:
            normalized, intake, inventory = normalize()
        except ConversionError:
            raise original_error
        source_stage_dir = work / "normalized-source"
        source_stage = lower_with_shared(normalized, source_stage_dir, optimizer=optimizer, stage="source")
    generic = (source_stage_dir / "lowered.mlir").read_text()
    module = parse_shared(generic)
    from .descriptor_bindings import expand
    flat, descriptor_bindings = expand(bindings, intake.get("descriptor_argument_expansion") if intake else None, source_entry, abi)
    module,report = bind_module(module,source_entry,abi,flat,byte_order=options.get("byte_order"))
    stream = StringIO()
    Printer(stream=stream,print_generic_format=True,print_debuginfo=True).print_op(module)
    bound = work / "bound.ttir"
    bound.write_text(stream.getvalue()+"\n")
    report.update(original_source_sha256=hashlib.sha256(original).hexdigest(),
                  bound_source_sha256=hashlib.sha256(bound.read_bytes()).hexdigest())
    if descriptor_bindings:
        report["descriptor_bindings"] = descriptor_bindings
    from .source_capsule import SCHEMA, PROTECTED_SCHEMA, record, validate
    source_custody = {"schema": "t2m.source-stage-custody.v1", "source_stage_schema": source_stage["schema"],
                      "stage": source_stage["stage"], "stage_completed": source_stage["stage_completed"],
                      "flags": source_stage["command"][1:-1], "artifacts": source_stage["artifacts"],
                      "optimizer": {key: source_stage["optimizer"][key] for key in ("sha256", "bytes")},
                      "grants_semantics": False}
    capsule = {"allocation-original.ttir": original.decode(), "allocation-input.mlir": generic,
               "allocation-arguments.json": json.dumps(configuration, sort_keys=True, indent=2) + "\n",
               "allocation-source-stage.json": json.dumps(source_custody, sort_keys=True, indent=2) + "\n",
               "allocation-source-stderr.log": (source_stage_dir / "stderr.log").read_text()}
    if normalized is not None:
        capsule.update({"allocation-normalized.ttir": normalized.read_text(),
                        "allocation-intake.json": (work / "intake/report.json").read_text(),
                        "allocation-inventory.json": (work / "intake/inventory.json").read_text()})
    capsule.update(protected_artifacts)
    report["source_capsule"] = {"schema": PROTECTED_SCHEMA if protected_artifacts else SCHEMA, "normalized": normalized is not None,
                                "artifacts": {name: record(data) for name, data in capsule.items()}, "grants_semantics": False}
    (work / "bindings.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    result = convert_shared(bound,abi,work / "translate",optimizer=optimizer,**options)
    result.trace["source_argument_binding"] = report
    result.auxiliary_artifacts.update(capsule)
    result.trace["abi_difference"] += "; explicit allocation-state entry binds source pointers before inlining; updated roots returned once"
    if protected_artifacts:
        # These are the pre-wrapper contracts, distinct from the bound source
        # inspected by convert_shared. Descriptor normalization may already
        # have expanded the original dot before that converter sees it.
        protection = json.loads(protected_artifacts["allocation-protection.json"])
        entry = next(iter(result.module.ops))
        entry.attributes["prov.allocation_source_contracts"] = builtin.StringAttr(json.dumps({
            "schema": protection["schema"], "source_entry": source_entry,
            "source_sha256": report["original_source_sha256"],
            "fp8_conversions": protection["fp8_conversions"],
            "assembly_conversions": protection["assembly_conversions"],
            "grants_semantics": False}, sort_keys=True))
        precisions = {operation["id"]: {**operation["dot_precision"],
            "inputs": operation["inputs"], "input_types": operation["input_types"],
            "intent": operation.get("intent", {})}
            for operation in protection["source_graph"]["operations"] if "dot_precision" in operation}
        if precisions:
            intent = builtin.StringAttr(json.dumps({"source_entry": source_entry,
                "source_sha256": report["original_source_sha256"], "contractions": precisions,
                "grants_semantics": False}, sort_keys=True))
            for operation in result.module.walk():
                if operation is entry or operation.name in ("linalg.matmul", "linalg.batch_matmul"):
                    operation.attributes["prov.allocation_original_dot_precisions"] = intent
        result.trace["abi_difference"] += "; original allocation numeric intent remains separately annotated; source contracts execute as standard arithmetic"
    if descriptor_bindings:
        entry = next(iter(result.module.ops))
        entry.attributes["prov.kernel_descriptor_state_bindings"] = builtin.StringAttr(json.dumps(descriptor_bindings, sort_keys=True))
        original_operations = (json.loads(protected_artifacts["allocation-protection.json"])["source_graph"]["operations"]
                               if protected_artifacts else inventory["operations"])
        original_descriptors = [operation for operation in original_operations
                                if operation["opcode"].startswith("tt.descriptor_") or operation["opcode"] == "tt.make_tensor_descriptor"]
        entry.attributes["prov.kernel_original_descriptors"] = builtin.StringAttr(json.dumps(
            {"source_sha256": report["original_source_sha256"], "operations": original_descriptors, "grants_semantics": False}, sort_keys=True))
        result.trace["abi_difference"] += "; original descriptors bind declared root origins and logical fields after native normalization"
    if protected_artifacts or descriptor_bindings:
        from .ir import text
        result.trace["mlir_sha256"] = hashlib.sha256(text(result.module).encode()).hexdigest()
    validate(result.trace, result.auxiliary_artifacts, result.source_ttir)
    return result
