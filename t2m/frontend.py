"""The only module importing Triton internals. Native parsing, never regex IR parsing."""

import hashlib
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .diagnostics import ConversionError

PINNED_TRITON = "3.7.1"
BLOCK_POINTER_OPS = frozenset(("tt.make_tensor_ptr", "tt.advance"))
TENSOR_DESCRIPTOR_OPS = frozenset(("tt.make_tensor_descriptor", "tt.descriptor_load", "tt.descriptor_store",
                                   "tt.descriptor_gather", "tt.descriptor_scatter", "tt.descriptor_reduce"))
DESCRIPTOR_OPS = BLOCK_POINTER_OPS | TENSOR_DESCRIPTOR_OPS
INTAKE_NORMALIZATION_OPS = DESCRIPTOR_OPS | frozenset(("tt.cat", "tt.dot"))


def _function_parameters(operations):
    """Read native function-entry block types without printed IR parsing."""
    blocks = {}
    for op in operations:
        if op.get_name() == "builtin.module":
            continue
        block = op.get_block()
        blocks.setdefault(block.get_parent().id(), {})[block.id()] = block
    functions = {}
    for op in operations:
        if op.get_name() != "tt.func" or not op.get_num_regions():
            continue
        owned = blocks.get(op.get_region(0).id(), {})
        if owned:
            entry = next(iter(owned.values()))
            functions[op.get_str_attr("sym_name")] = [str(entry.get_argument(i).get_type())
                                                      for i in range(entry.get_num_arguments())]
    return functions


def _native_operation_inventory(operations, precision_dots=()):
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
        intent = {name: op.get_str_attr(name) for name in ("prov.family", "prov.op", "prov.region_id")}
        if any(value is not None for value in intent.values()):
            records[-1]["intent"] = {name: value for name, value in intent.items() if value is not None}
        if op in precision_dots:
            records[-1]["dot_precision"] = {"input_precision": {0: "tf32", 1: "tf32x3", 3: "bf16x3", 4: "bf16x6"}[op.get_int_attr("inputPrecision")],
                "max_imprecise_accumulations": op.get_int_attr("maxNumImpreciseAcc") or 0,
                "accumulator": ("original-source-C" if op.get_int_attr("inputPrecision") == 0
                                else "source-C-added-after-official-composite"), "physical_mapping": "undecided"}
            if op.get_int_attr("inputPrecision") in (0, 1):
                records[-1]["dot_precision"]["logical_numeric_profile"] = "logical-tf32-e8m10-rtz-v1"
    return {"operations": records, "blocks": list(blocks.values()),
            "attributes": "exact-original-source; native binding does not expose complete attribute dictionaries"}


def _inspect_ttir(path, inventory_path):
    """Worker-only source verification/inventory without normalization."""
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    ir, context = _context()
    module = ir.parse_mlir_module(str(Path(path).resolve()), context)
    if not module.verify():
        raise ConversionError("Triton rejected the original source")
    operations = []
    module.walk(operations.append)
    if len(operations) > 200_000:
        raise ConversionError("source inventory exceeds its operation budget")
    result = _native_operation_inventory(operations)
    result.update(schema="t2m.native-source-inventory.v1", triton_version=PINNED_TRITON,
                  normalization="not-performed", target_backend="not-invoked")
    Path(inventory_path).write_text(json.dumps(result, indent=2) + "\n")


def inspect_ttir(source, work):
    """Retain exact source and native verification independently of conversion."""
    source, work = Path(source), Path(work)
    if source.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("source inspection exceeds its 16 MiB input budget")
    data = source.read_bytes()
    work.mkdir(parents=True, exist_ok=False)
    saved, inventory = work / "source.ttir", work / "inventory.json"
    saved.write_bytes(data)
    command = [sys.executable, "-c", ("import sys; sys.path.insert(0,sys.argv[1]); "
               "from t2m.frontend import _inspect_ttir; _inspect_ttir(sys.argv[2],sys.argv[3])"),
               str(Path(__file__).resolve().parents[1]), str(saved.resolve()), str(inventory.resolve())]
    with (work / "stdout.log").open("xb") as stdout, (work / "stderr.log").open("xb") as stderr:
        try:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=30, check=False,
                env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(work.resolve())))
        except subprocess.TimeoutExpired as exc:
            raise ConversionError("native source inspection exceeded 30 seconds") from exc
    if process.returncode or not inventory.is_file():
        raise ConversionError("native source verification failed; inspect " + str(work / "stderr.log"))
    if saved.read_bytes() != data:
        raise ConversionError("source changed during native inspection")
    return json.loads(inventory.read_text())


SOURCE_PRINT_FLAGS = ("--mlir-print-op-generic", "--mlir-print-local-scope", "--mlir-print-debuginfo")
SOURCE_PRINT_SCHEMA = "t2m.native-source-print.v1"


def _enable_generic_source_print(library):
    """Worker-only, pin-probed printer options in the already loaded library.

    LLVM's C API parses the fixed MLIR printer flags. The no-argument MLIR
    registration has the pinned Linux/Itanium exported name; no native object
    layouts or pointers cross ctypes. Missing exports/flags fail in this child.
    """
    import ctypes

    try:
        native = ctypes.CDLL(str(library))
        register = getattr(native, "_ZN4mlir27registerAsmPrinterCLOptionsEv")
        parse = native.LLVMParseCommandLineOptions
    except (OSError, AttributeError) as error:
        raise ConversionError("the pinned native source printer exports are unavailable") from error
    register.argtypes, register.restype = [], None
    parse.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_char_p]
    parse.restype = None
    register()
    args = [b"t2m-source-printer", *(flag.encode() for flag in SOURCE_PRINT_FLAGS)]
    parse(len(args), (ctypes.c_char_p * len(args))(*args), b"verified TTIR source printing")


def _print_ttir_generic(source, output, report_path):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    ir, context = _context()
    from triton._C import libtriton

    module = ir.parse_mlir_module(str(Path(source).resolve()), context)
    if not module.verify():
        raise ConversionError("native source printer rejected the original TTIR")
    count = [0]
    def count_operation(_operation):
        count[0] += 1
        if count[0] > 200_000:
            raise ConversionError("generic source exceeds its operation budget")
    module.walk(count_operation)
    library = Path(libtriton.__file__)
    _enable_generic_source_print(library)
    generic = module.str()
    if not generic.lstrip().startswith('"builtin.module"'):
        raise ConversionError("native printer did not honor generic source flags")
    Path(output).write_text(generic)
    # No pass runs. Reparse the complete native serialization, including all
    # properties, attributes, regions, types and exact constant spellings.
    restored = ir.parse_mlir_module(str(Path(output).resolve()), context)
    if not restored.verify() or not module.verify() or restored.str() != generic:
        raise ConversionError("native generic source did not round-trip exactly")
    digest = hashlib.sha256()
    with library.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {"schema": SOURCE_PRINT_SCHEMA, "triton_version": PINNED_TRITON,
        "flags": list(SOURCE_PRINT_FLAGS), "operations": count[0], "normalization": "not-performed", "target_backend": "not-invoked",
        "native_source_verified": True, "native_generic_roundtrip_verified": True, "grants_semantics": False,
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
        "implementation": {"binary_sha256": digest.hexdigest(), "binary_bytes": library.stat().st_size,
            "printer_options": "MLIR-registerAsmPrinterCLOptions; LLVM-C-LLVMParseCommandLineOptions",
            "source_reference_revision": "f797708c0626e5f9840ca5b0a98790e2c7cb09ad",
            "source_reference_is_build_receipt": False}}
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")


def read_ttir_generic(source, work):
    """Read complete verified source before any source-version normalization.

    Printer options are process-global in LLVM, so only an isolated bounded
    child may select them. The caller's Triton/MLIR printers remain untouched.
    """
    source, work = Path(source), Path(work)
    if not source.is_file() or not 0 < source.stat().st_size <= 4 * 1024 * 1024:
        raise ConversionError("generic source input must be nonempty and at most 4MiB")
    data = source.read_bytes()
    work.mkdir(parents=True, exist_ok=False)
    saved, output, report_path = work / "source.ttir", work / "lowered.mlir", work / "print.json"
    saved.write_bytes(data)
    command = [sys.executable, "-c", "import sys; sys.path.insert(0,sys.argv[1]); "
        "from t2m.frontend import _print_ttir_generic; _print_ttir_generic(*sys.argv[2:])",
        str(Path(__file__).resolve().parents[1]), str(saved.resolve()), str(output.resolve()), str(report_path.resolve())]
    with (work / "stdout.log").open("xb") as stdout, (work / "stderr.log").open("xb") as stderr:
        try:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=30, check=False,
                env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(work.resolve())))
        except subprocess.TimeoutExpired as error:
            raise ConversionError("native generic source printing exceeded 30 seconds") from error
    if process.returncode or not report_path.is_file() or not output.is_file():
        raise ConversionError("native generic source printing failed; inspect " + str(work / "stderr.log"))
    report = json.loads(report_path.read_text())
    if (saved.read_bytes() != data or report.get("schema") != SOURCE_PRINT_SCHEMA
            or report.get("flags") != list(SOURCE_PRINT_FLAGS)
            or report.get("native_source_verified") is not True
            or report.get("native_generic_roundtrip_verified") is not True
            or report.get("normalization") != "not-performed" or report.get("grants_semantics") is not False
            or report.get("source_sha256") != hashlib.sha256(data).hexdigest()
            or report.get("output_sha256") != hashlib.sha256(output.read_bytes()).hexdigest()):
        raise ConversionError("native generic source differs from its verified source/print records")
    return report


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
    original_parameters = _function_parameters(operations)
    from .descriptor_arguments import has_parameters
    descriptor_parameters = has_parameters(original_parameters)
    descriptors = bool(names & TENSOR_DESCRIPTOR_OPS) or descriptor_parameters
    # Parse only the typed value's complete type syntax, never printed IR lines.
    from xdsl.dialects.builtin import TensorType, f32
    from xdsl.parser import Parser

    from .ir import context as xdsl_context
    type_context = xdsl_context()
    precision_dots = []
    for op in operations:
        if op.get_name() != "tt.dot" or op.get_int_attr("inputPrecision") not in (0, 1, 3, 4):
            continue
        types = [Parser(type_context, str(op.get_operand(i).get_type())).parse_type() for i in range(3)]
        if not all(isinstance(type_, TensorType) and type_.element_type == f32 for type_ in types[:2]):
            continue  # The pinned pass ignores precision for other inputs.
        if types[2].element_type != f32 or (op.get_int_attr("maxNumImpreciseAcc") or 0) < 0:
            raise ConversionError("selected f32 dot precision requires f32 C and a nonnegative accumulation bound", opcode="tt.dot")
        precision_dots.append(op)
    normalized_names = names & (DESCRIPTOR_OPS | {"tt.cat"})
    if descriptor_parameters:
        normalized_names.add("tt.func")
    if precision_dots:
        if any(op.get_name() == "tt.load" and op.get_bool_attr("isVolatile") for op in operations):
            raise ConversionError("volatile observations cannot become stable tensor reads", opcode="tt.load")
        if names & {"tt.inline_asm", "tt.elementwise_inline_asm"}:
            raise ConversionError("source assembly needs a separately qualified numeric/effect profile", opcode="tt.elementwise_inline_asm")
        normalized_names.add("tt.dot")
    if not normalized_names:
        raise ConversionError("input does not use a supported intake normalization")
    pipeline = []
    if "tt.cat" in names:
        pipeline.extend(("t2m-expand-cat-preserve-order", "canonicalize"))
    if descriptors:
        pipeline.extend(("triton-rewrite-tensor-descriptor-to-pointer", "canonicalize"))
    if names & BLOCK_POINTER_OPS:
        pipeline.append("triton-rewrite-tensor-pointer")
    precision_modes = {op.get_int_attr("inputPrecision") for op in precision_dots}
    if precision_modes & {1, 3, 4}:
        pipeline.append("tritongpu-F32DotTC{emu-tf32=" + ("true" if 1 in precision_modes else "false") + "}")
    if precision_modes & {0, 1}:
        pipeline.extend(("t2m-expand-tf32-e8m10-rtz-v1", "canonicalize"))
    inventory = _native_operation_inventory(operations, precision_dots)
    normalizer_implementation = None
    if precision_dots or descriptors:
        from triton._C import libtriton
        binary = Path(libtriton.__file__)
        digest = hashlib.sha256()
        with binary.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        normalizer_implementation = {"version": PINNED_TRITON, "binary_sha256": digest.hexdigest(),
            "binary_bytes": binary.stat().st_size,
            "source_reference_revision": "f797708c0626e5f9840ca5b0a98790e2c7cb09ad",
            "source_reference_is_build_receipt": False}
    Path(inventory_path).write_text(json.dumps({**inventory, "normalization_passes": pipeline,
        "normalized_opcodes": sorted(normalized_names),
        **({"dot_precision_implementation": normalizer_implementation} if precision_dots else {}),
        **({"descriptor_implementation": normalizer_implementation} if descriptors else {})}, indent=2) + "\n")
    if "tt.cat" in names:
        _expand_cats(ir, context, module, operations)
    manager = ir.pass_manager(context)
    if descriptors:
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(manager)
        # The official pass emits both padding alternatives and selects the
        # actual descriptor option. Fold unused NaN alternatives before intake.
        passes.common.add_canonicalizer(manager)
    if names & BLOCK_POINTER_OPS:
        passes.ttir.add_rewrite_tensor_pointer(manager)
    if precision_modes & {1, 3, 4}:
        passes.ttgpuir.add_f32_dot_tc(manager, 1 in precision_modes)
    manager.run(module, "triton2mlir-expand-tensor-pointers")
    if precision_modes & {0, 1}:
        _expand_tf32(ir, context, module)
    if not module.verify():
        raise ConversionError("official tensor-pointer rewrite produced invalid IR")
    if descriptors:
        from .descriptor_arguments import expansions
        normalized = []
        module.walk(normalized.append)
        arguments = expansions(original_parameters, _function_parameters(normalized))
        if arguments is not None:
            inventory = json.loads(Path(inventory_path).read_text())
            inventory["descriptor_argument_expansion"] = arguments
            Path(inventory_path).write_text(json.dumps(inventory, indent=2) + "\n")
    return str(module)  # Keep source NameLocs; str_nodebug discards provenance.


def _expand_tf32(ir, native_context, module):
    """Standard arithmetic for the explicit logical TF32 E8M10/RTZ profile.

    Only the official F32DotTC pass may introduce the RNA assembly below;
    source assembly was rejected before invoking it. Preserve all partial dots
    and their accumulators. Physical instruction selection remains downstream.
    """
    from xdsl.dialects.builtin import TensorType, f32
    from xdsl.parser import Parser

    from .ir import context

    type_context = context()
    builder = ir.builder(native_context)
    operations = []
    module.walk(operations.append)
    # Reuse within a single block only: the first generated conversion then
    # dominates later uses, without hoisting across branches or loop scopes.
    converted = {}

    def tf32_dot(op):
        if op.get_name() != "tt.dot" or op.get_int_attr("inputPrecision") != 0:
            return False
        types = [Parser(type_context, str(op.get_operand(i).get_type())).parse_type() for i in range(2)]
        return all(isinstance(type_, TensorType) and type_.element_type == f32 for type_ in types)

    def quantize(value, block, *, rna=False):
        key = (block.id(), value.id(), rna)
        if key in converted:
            return converted[key]
        type_ = Parser(type_context, str(value.get_type())).parse_type()
        if not isinstance(type_, TensorType) or type_.element_type != f32:
            raise ConversionError("TF32 conversion requires a ranked f32 tensor")
        integer = builder.get_block_ty(builder.get_int32_ty(), list(type_.get_shape()))

        def constant(word):
            return builder.create_splat(integer, builder.get_uint32(word))

        bits = builder.create_bitcast(value, integer)
        magnitude = builder.create_and(bits, constant(0x7FFFFFFF))
        nan = builder.create_icmpUGT(magnitude, constant(0x7F800000))
        if rna:
            magnitude = builder.create_add(magnitude, constant(0x1000))
        magnitude = builder.create_and(magnitude, constant(0x7FFFE000))
        sign = builder.create_and(bits, constant(0x80000000))
        rounded = builder.create_or(magnitude, sign)
        # A low NaN payload must not turn into infinity after truncation.
        rounded = builder.create_select(nan, constant(0x7FC00000), rounded)
        result = builder.create_bitcast(rounded, value.get_type())
        converted[key] = result
        if rna:
            # The generated RNA result is already E8M10 aligned, including
            # canonical NaN. Its later RTZ conversion is an exact identity.
            converted[(block.id(), result.id(), False)] = result
        return result

    for op in operations:
        name = op.get_name()
        if name == "tt.elementwise_inline_asm":
            if (op.get_str_attr("asm_string") != "cvt.rna.tf32.f32 $0, $1;"
                    or op.get_str_attr("constraints") != "=r,r"
                    or op.get_bool_attr("pure") is not True
                    or op.get_int_attr("packed_element") != 1
                    or op.get_num_operands() != 1 or op.get_num_results() != 1
                    or op.get_operand(0).get_type() != op.get_result(0).get_type()):
                raise ConversionError("official TF32 expansion emitted an unqualified assembly form", opcode=name)
            result = op.get_result(0)
            builder.set_loc(result.get_loc())
            builder.set_insertion_point_after(op)
            result.replace_all_uses_with(quantize(op.get_operand(0), op.get_block(), rna=True))
        elif tf32_dot(op):
            result = op.get_result(0)
            builder.set_loc(result.get_loc())
            builder.set_insertion_point_after(op)
            a, b = (quantize(op.get_operand(i), op.get_block()) for i in range(2))
            replacement = builder.create_dot(a, b, op.get_operand(2), ir.INPUT_PRECISION.IEEE,
                                             op.get_int_attr("maxNumImpreciseAcc") or 0)
            result.replace_all_uses_with(replacement)
    from triton._C.libtriton import passes
    manager = ir.pass_manager(native_context)
    passes.common.add_canonicalizer(manager)
    manager.run(module, "triton2mlir-expand-logical-tf32")
    remaining = []
    module.walk(remaining.append)
    if any(op.get_name() == "tt.elementwise_inline_asm" or tf32_dot(op) for op in remaining):
        raise ConversionError("TF32 expansion left unexpanded precision or assembly")


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
               ("import sys; sys.path.insert(0, sys.argv[1]); "
               "from t2m.frontend import _expand_tensor_pointers; "
               "sys.stdout.write(_expand_tensor_pointers(sys.argv[2], sys.argv[3]))"),
               str(Path(__file__).resolve().parents[1]), str(saved.resolve()), str((work / "inventory.json").resolve())]
    with output.open("xb") as stdout, (work / "stderr.log").open("xb") as stderr:
        try:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=30, check=False,
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
    if "dot_precision_implementation" in inventory:
        report["dot_precision_implementation"] = inventory["dot_precision_implementation"]
    if "descriptor_implementation" in inventory:
        report["descriptor_implementation"] = inventory["descriptor_implementation"]
        report["descriptor_profile"] = {
            "name": "logical-tensor-descriptors-v1",
            "bounds": "signed-per-dimension; out-of-bounds loads use source padding; stores/reductions are masked",
            "addressing": "source base/shape/element-strides and logical indices",
            "effects": "ordinary stores require unique ownership; reducing stores retain atomic kind/order/scope",
            "physical_mapping": "undecided",
            "physical_descriptor_abi": "not-introduced",
            "accelerator_equivalence": "not-established",
            "grants_semantics": False,
        }
    if "descriptor_argument_expansion" in inventory:
        report["descriptor_argument_expansion"] = inventory["descriptor_argument_expansion"]
    if "t2m-expand-tf32-e8m10-rtz-v1" in pipeline:
        report["tf32_numeric_profile"] = {"name": "logical-tf32-e8m10-rtz-v1",
            "dot_inputs": "E8M10-toward-zero; gradual-underflow; computed-NaN-class",
            "composite_high_parts": "nearest-ties-away-from-zero",
            "accumulation": "standard-f32-products-ascending-K-adds",
            "physical_mapping": "undecided", "accelerator_equivalence": "not-established"}
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


def _extract_map_layouts(path, record_path, options):
    """Worker: derive packed value grouping with official layout conversion.

    Only source layout selection runs, never LLVM/device code generation. The
    caller adds unique map markers to the exact inlined pre-layout source.
    """
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    from triton._C.libtriton import gluon_ir, linear_layout, passes
    library = Path(sys.modules["triton._C.libtriton"].__file__).resolve()
    with library.open("rb") as stream:
        native = {"path": str(library), "bytes": library.stat().st_size,
                  "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    options = json.loads(options)
    ir, context = _context()
    module = ir.parse_mlir_module(str(Path(path).resolve()), context)
    if not module.verify():
        raise ConversionError("Triton rejected the source layout specialization")
    before = []
    module.walk(before.append)
    expected = [op.get_str_attr("t2m.map_key") for op in before if op.get_name() == "tt.map_elementwise" and op.get_int_attr("pack") != 1]
    if not expected or any(key is None for key in expected) or len(set(expected)) != len(expected):
        raise ConversionError("source layout extraction needs unique map identities")
    manager = ir.pass_manager(context)
    passes.ttir.add_convert_to_ttgpuir(manager, options["target"], options["num_warps"],
                                     options["threads_per_warp"], options["num_ctas"])
    manager.run(module, "t2m-source-layout")
    if not module.verify():
        raise ConversionError("Triton rejected the converted source layout")
    operations = []
    module.walk(operations.append)
    builder = gluon_ir.GluonOpBuilder(context)
    maps = []
    for op in operations:
        if op.get_name() != "tt.map_elementwise" or op.get_int_attr("pack") == 1:
            continue
        values = [op.get_operand(i) for i in range(op.get_num_operands())]
        values += [op.get_result(i) for i in range(op.get_num_results())]
        layouts, shape = [], None
        for value in values:
            current_shape = builder.get_shape_from_tensor(value)
            encoding = builder.get_gluon_layout_from_tensor(value)
            layout = builder.to_linear_layout(encoding._to_ir(builder), current_shape)
            fields = {name: getattr(layout, name) for name in ("reg_bases", "lane_bases", "warp_bases", "block_bases", "shape")}
            if shape is None:
                shape = current_shape
            if current_shape != shape or fields["shape"] != shape:
                raise ConversionError("packed source values need matching logical shapes")
            layouts.append(fields)
        if not layouts or any(value != layouts[0] for value in layouts):
            raise ConversionError("packed source values need the same register-to-coordinate mapping")
        bases = [(name, layouts[0][field]) for name, field in (
            ("register", "reg_bases"), ("lane", "lane_bases"), ("warp", "warp_bases"), ("block", "block_bases"))]
        names = [f"dim{i}" for i in range(len(shape))]
        forward = linear_layout.LinearLayout.from_bases(bases, names, shape)
        if not forward.is_invertible():
            raise ConversionError("packed source grouping currently requires bijective value coverage; replicated layouts need an observation contract")
        pack = op.get_int_attr("pack")
        if pack is None or pack < 2 or (1 << len(bases[0][1])) % pack:
            raise ConversionError("source pack must divide its actual per-thread register value count")
        inverse = forward.invert()
        maps.append({"id": op.get_str_attr("t2m.map_key"), "pack": pack, "shape": shape,
            "forward": {"bases": forward.bases, "out_dims": forward.out_dims},
            "inverse": {"bases": inverse.bases, "out_dims": inverse.out_dims},
            "source_value_types": [str(value.get_type()) for value in values]})
    if sorted(row["id"] for row in maps) != sorted(expected):
        raise ConversionError("source layout conversion changed the map identity roster")
    text = module.str_nodebug()
    Path(record_path).write_text(json.dumps({"schema": "t2m.source-map-layout.v1", "maps": maps,
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "source_options": options,
        "triton_version": PINNED_TRITON, "pipeline": ["convert-triton-to-tritongpu"],
        "native_library": native, "dependency_source_closure": "not-established",
        "ttgir": text, "ttgir_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "device_codegen": "not-invoked"}, indent=2) + "\n")


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


def capture_ttir(kernel, signature, constexprs, *, debug=None, source_target=None):
    """Capture general pre-layout TTIR; apply semantic ABI checks downstream.

    Unlike the reference frontend's `_wrap`, this preserves regions, arbitrary
    ranks and source types. ``debug`` selects Triton's source diagnostic option;
    None retains the backend default. No backend compilation or launch is run.
    ``source_target`` optionally selects the nominal (backend, arch, warp_size)
    AST frontend descriptor. This permits source dialect/type choices supported
    by another official frontend; it supplies no executable target qualification.
    """
    if debug is not None and type(debug) is not bool:
        raise ValueError("capture debug must be a Boolean source option")
    _, context = _context()
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, make_backend
    if set(signature) != set(kernel.arg_names):
        raise ConversionError("capture signature must describe every source parameter")
    if source_target is None:
        source_target = ("cuda", 80, 32)
    if (not isinstance(source_target, tuple) or len(source_target) != 3
            or not isinstance(source_target[0], str) or not source_target[0]
            or type(source_target[1]) not in (str, int) or type(source_target[2]) is not int
            or source_target[2] <= 0):
        raise ValueError("source target requires a (backend, arch, positive warp_size) descriptor")
    target = GPUTarget(*source_target)
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
