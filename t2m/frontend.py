"""The only module importing Triton internals. Native parsing, never regex IR parsing."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import struct

from .diagnostics import ConversionError

PINNED_TRITON = "3.7.1"


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


def value_id(value):
    return value.id()

