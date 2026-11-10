"""The only module importing Triton internals. Native parsing, never regex IR parsing."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import os
import struct
import subprocess
import sys

from .diagnostics import ConversionError

PINNED_TRITON = "3.7.1"
BLOCK_POINTER_OPS = frozenset(("tt.make_tensor_ptr", "tt.advance"))
TENSOR_DESCRIPTOR_OPS = frozenset(("tt.make_tensor_descriptor", "tt.descriptor_load", "tt.descriptor_store",
                                   "tt.descriptor_gather", "tt.descriptor_scatter", "tt.descriptor_reduce"))
DESCRIPTOR_OPS = BLOCK_POINTER_OPS | TENSOR_DESCRIPTOR_OPS
INTAKE_NORMALIZATION_OPS = DESCRIPTOR_OPS | frozenset(("tt.cat",))


def _expand_tensor_pointers(path, inventory_path):
    """Worker-only checked intake rewrites; never invoke a target backend."""
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    ir, context = _context()
    from triton._C.libtriton import passes
    module = ir.parse_mlir_module(str(Path(path).resolve()), context)
    if not module.verify():
        raise ConversionError("Triton rejected the original tensor-pointer source")
    operations = []
    module.walk(operations.append)
    names = {op.get_name() for op in operations}
    if not names & INTAKE_NORMALIZATION_OPS:
        raise ConversionError("input does not use a supported intake normalization")
    pipeline = []
    if "tt.cat" in names:
        pipeline.extend(("t2m-expand-cat-preserve-order", "canonicalize"))
    if names & TENSOR_DESCRIPTOR_OPS:
        pipeline.extend(("triton-rewrite-tensor-descriptor-to-pointer", "canonicalize"))
    if names & BLOCK_POINTER_OPS:
        pipeline.append("triton-rewrite-tensor-pointer")
    values, blocks, records, regions = {}, {}, [], {}

    def value_id(value):
        return values.setdefault(value.id(), f"original.ttir.value.{len(values):04d}")

    def region_id(region):
        return regions.setdefault(region.id(), f"original.ttir.region.{len(regions):04d}")

    for op in operations:
        owned_regions = [region_id(op.get_region(i)) for i in range(op.get_num_regions())]
        if op.get_name() == "builtin.module":
            continue
        block = op.get_block()
        if block.id() not in blocks:
            args = [block.get_argument(i) for i in range(block.get_num_arguments())]
            blocks[block.id()] = {"id": f"original.ttir.block.{len(blocks):04d}",
                                  "parent_region": region_id(block.get_parent()),
                                  "arguments": [{"id": value_id(v), "type": str(v.get_type())} for v in args]}
        inputs = [op.get_operand(i) for i in range(op.get_num_operands())]
        outputs = [op.get_result(i) for i in range(op.get_num_results())]
        records.append({"id": f"original.ttir.op.{len(records):04d}", "opcode": op.get_name(),
                        "block": blocks[block.id()]["id"], "regions": owned_regions,
                        "inputs": [value_id(v) for v in inputs], "outputs": [value_id(v) for v in outputs],
                        "input_types": [str(v.get_type()) for v in inputs],
                        "output_types": [str(v.get_type()) for v in outputs]})
    Path(inventory_path).write_text(json.dumps({"operations": records, "blocks": list(blocks.values()),
        "attributes": "exact-original-source; native binding does not expose complete attribute dictionaries",
        "normalization_passes": pipeline,
        "normalized_opcodes": sorted(names & INTAKE_NORMALIZATION_OPS)}, indent=2) + "\n")
    if "tt.cat" in names:
        _expand_cats(ir, context, module, operations)
    manager = ir.pass_manager(context)
    if names & TENSOR_DESCRIPTOR_OPS:
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(manager)
        # The official pass emits both padding alternatives and selects the
        # actual descriptor option. Fold unused NaN alternatives before intake.
        passes.common.add_canonicalizer(manager)
    if names & BLOCK_POINTER_OPS:
        passes.ttir.add_rewrite_tensor_pointer(manager)
    manager.run(module, "triton2mlir-expand-tensor-pointers")
    if not module.verify():
        raise ConversionError("official tensor-pointer rewrite produced invalid IR")
    return str(module)  # Keep source NameLocs; str_nodebug discards provenance.


def _expand_cats(ir, native_context, module, operations):
    """Choose lhs-then-rhs order allowed by TTIR cat, using native builders.

    Triton v3.7.1 language/core.py::cat uses join/trans/reshape for ordered
    concatenation. Meta 17f0c234's CatConverter chooses the same dimension-zero
    order via tensor.concat. No physical layout or element reordering is needed.
    """
    from xdsl.dialects.builtin import NoneAttr, TensorType
    from xdsl.parser import Parser
    from .ir import context
    type_context = context()
    type_context.allow_unregistered = True
    builder = ir.builder(native_context)
    for op in operations:
        if op.get_name() != "tt.cat":
            continue
        inputs = [op.get_operand(i) for i in range(op.get_num_operands())]
        result = op.get_result(0)
        # Native bindings expose types only as objects/assembly. Parse their
        # complete typed assembly structurally; do not infer IR from lines.
        types = [Parser(type_context, str(v.get_type())).parse_type() for v in (*inputs, result)]
        if (len(inputs) != 2 or not all(isinstance(t, TensorType) for t in types)
                or types[0] != types[1] or len(types[0].get_shape()) != 1
                or types[2].get_shape() != (2 * types[0].get_shape()[0],)
                or any(not isinstance(t.encoding, NoneAttr) for t in types)):
            raise ConversionError("cat intake requires equal rank-one pre-layout inputs and a doubled result", opcode="tt.cat")
        builder.set_loc(result.get_loc())
        builder.set_insertion_point_after(op)
        joined = builder.create_join(*inputs)
        transposed = builder.create_trans(joined, [1, 0])
        expanded = builder.create_reshape(transposed, list(types[2].get_shape()), False)
        result.replace_all_uses_with(expanded)
    if not module.verify():
        raise ConversionError("cat expansion produced invalid native TTIR", opcode="tt.cat")
    from triton._C.libtriton import passes
    manager = ir.pass_manager(native_context)
    passes.common.add_canonicalizer(manager)
    manager.run(module, "triton2mlir-expand-cat")


def expand_tensor_pointers(source, work):
    """Bound native intake rewrites in a child; retain exact input/output."""
    source, work = Path(source), Path(work)
    data = source.read_bytes()
    if not 0 < len(data) <= 4 * 1024 * 1024:
        raise ConversionError("tensor-pointer input must be nonempty and at most 4MiB")
    work.mkdir(parents=True, exist_ok=False)
    saved, output = work / "source.ttir", work / "lowered.mlir"
    saved.write_bytes(data)
    command = [sys.executable, "-c",
               "import sys; sys.path.insert(0, sys.argv[1]); "
               "from t2m.frontend import _expand_tensor_pointers; "
               "sys.stdout.write(_expand_tensor_pointers(sys.argv[2], sys.argv[3]))",
               str(Path(__file__).resolve().parents[1]), str(saved.resolve()), str((work / "inventory.json").resolve())]
    with output.open("xb") as stdout, (work / "stderr.log").open("xb") as stderr:
        try:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=30,
                                     env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                              TMPDIR=str(work.resolve())))
        except subprocess.TimeoutExpired as exc:
            raise ConversionError("official tensor-pointer rewrite exceeded 30 seconds") from exc
    if process.returncode or not output.stat().st_size:
        raise ConversionError("official tensor-pointer rewrite failed; inspect " + str(work / "stderr.log"))
    inventory = json.loads((work / "inventory.json").read_text())
    pipeline = inventory["normalization_passes"]
    report = {"schema": "t2m.intake-pass.v1", "triton_version": PINNED_TRITON,
              "pass": pipeline[0], "pipeline": pipeline, "target_backend": "not-invoked",
              "normalized_opcodes": inventory["normalized_opcodes"],
              "source_sha256": hashlib.sha256(data).hexdigest(),
              "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (work / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _context():
    try:
        found = version("triton")
    except PackageNotFoundError as exc:
        raise ConversionError("Triton is required for intake; install triton2mlir[triton]") from exc
    if found != PINNED_TRITON:
        raise ConversionError(f"expected triton=={PINNED_TRITON}, found {found}")
    from triton._C.libtriton import ir
    context = ir.context()
    ir.load_dialects(context)
    return ir, context


@dataclass
class TTIR:
    context: object
    module: object
    text: str
    operations: list
    entry_args: list
    origin: str
    constants: dict

    @property
    def digest(self):
        return hashlib.sha256(self.text.encode()).hexdigest()


def _wrap(module, context, spec, origin):
    if not module.verify():
        raise ConversionError("Triton rejected the input module")
    operations = []
    module.walk(operations.append)
    functions = [op for op in operations if op.get_name() == "tt.func"]
    if len(functions) != 1 or module.get_entry_func_name() != spec.entry:
        raise ConversionError("MVP requires one tt.func matching the declared entry")
    body = [op for op in operations if op.get_name() not in ("tt.func", "builtin.module")]
    if not body:
        raise ConversionError("empty TTIR entry")
    for index, op in enumerate(body):
        if op.get_num_regions():
            raise ConversionError("nested regions are a later capability; refusing to flatten them",
                                  source_id=f"ttir.op.{index:04d}", opcode=op.get_name())
    block = body[0].get_block()
    if any(op.get_block().id() != block.id() for op in body):
        raise ConversionError("multiple blocks are not supported in the MVP")
    args = [block.get_argument(i) for i in range(block.get_num_arguments())]
    if len(args) != len(spec.buffers):
        raise ConversionError("TTIR argument count differs from the tensor ABI")
    if any(str(arg.get_type()) != "!tt.ptr<f32>" for arg in args):
        raise ConversionError("MVP entry arguments must all be !tt.ptr<f32>")
    text = module.str_nodebug()
    return TTIR(context, module, text, operations, args, origin, _constants(text, operations))


def _constants(text, operations):
    """Read scalar/splat literals missing from the pinned native attribute API.

    Like Merlin's constant adapter, correlate the native and printed renderings
    only after checking counts, types and readable scalar integers. No IR control
    flow or operation semantics are inferred by textual pattern matching.
    """
    native = [op for op in operations if op.get_name() == "arith.constant"]
    printed = [line.partition(" = arith.constant ")[2] for line in text.splitlines()
               if " = arith.constant " in line]
    if len(native) != len(printed):
        raise ConversionError("native/printed constant inventories disagree")
    constants = {}
    for op, fragment in zip(native, printed, strict=True):
        literal, separator, type_text = fragment.rpartition(" : ")
        if not separator:
            literal, type_text = fragment, "i1"
        type_ = str(op.get_result(0).get_type())
        if type_ != type_text.strip():
            raise ConversionError("native/printed constant types disagree")
        body = literal.removeprefix("dense<").removesuffix(">")
        # A live unsupported constant will receive a located analysis refusal.
        if body.startswith("[") or "{" in body:
            continue
        try:
            if type_ == "f32" or type_.endswith("xf32>"):
                bits = (int(body, 16) if body.startswith("0x")
                        else struct.unpack("<I", struct.pack("<f", float(body)))[0])
                if not 0 <= bits < 1 << 32:
                    continue
                value = ("f32", bits)
            else:
                number = {"true": 1, "false": 0}.get(body)
                if number is None:
                    number = int(body)
                direct = op.get_int_attr("value")
                if direct is not None and (bool(direct) if type_ == "i1" else direct) != number:
                    raise ConversionError("native/printed constant values disagree")
                value = ("i1" if type_ == "i1" or type_.endswith("xi1>") else "integer", number)
        except (ValueError, OverflowError):
            continue
        constants[op.get_result(0).id()] = value
    return constants


def parse(path, spec):
    ir, context = _context()
    module = ir.parse_mlir_module(str(Path(path).resolve()), context)
    return _wrap(module, context, spec, "saved-ttir")


def capture(kernel, spec):
    _, context = _context()
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, make_backend
    if getattr(kernel, "__name__", None) != spec.entry:
        raise ConversionError("kernel function name differs from spec.entry")
    expected = [buffer.name for buffer in spec.buffers] + [name for name, _ in spec.constexprs]
    if list(getattr(kernel, "arg_names", ())) != expected:
        raise ConversionError("source argument order/names must match buffers followed by constexprs")
    # Nominal callbacks only: no driver, compile(), layout pass or device binary.
    target = GPUTarget("cuda", 80, 32)
    backend = make_backend(target)
    backend.load_dialects(context)
    options = backend.parse_options({})
    signature = {buffer.name: "*fp32" for buffer in spec.buffers}
    signature.update({name: "constexpr" for name, _ in spec.constexprs})
    source = ASTSource(fn=kernel, signature=signature, constexprs=dict(spec.constexprs))
    module = source.make_ir(target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    return _wrap(module, context, spec, "triton-source")


def capture_ttir(kernel, signature, constexprs, *, debug=None):
    """Capture general pre-layout TTIR; apply semantic ABI checks downstream.

    Unlike the reference frontend's `_wrap`, this preserves regions, arbitrary
    ranks and source types. ``debug`` selects Triton's source diagnostic option;
    None retains the backend default. No backend compilation or launch is run.
    """
    if debug is not None and type(debug) is not bool:
        raise ValueError("capture debug must be a Boolean source option")
    _, context = _context()
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, make_backend
    if set(signature) != set(kernel.arg_names):
        raise ConversionError("capture signature must describe every source parameter")
    target = GPUTarget("cuda", 80, 32)
    backend = make_backend(target)
    backend.load_dialects(context)
    options = backend.parse_options({} if debug is None else {"debug": debug})
    source = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    if not module.verify():
        raise ConversionError("Triton rejected captured TTIR")
    return module.str_nodebug()


def value_id(value):
    return value.id()
