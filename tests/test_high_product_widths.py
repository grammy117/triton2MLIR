"""All declared TT_Int high-product widths preserve actual unsigned words."""

import itertools
import json
import os
from io import StringIO

import numpy as np
import pytest
from xdsl.dialects import arith, builtin
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir, inspect_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import (
    TensorABI,
    TensorBuffer,
    convert_shared,
    opcode,
    parse_shared,
)
from t2m.upstream import lower_with_shared

TOOLS = all(os.environ.get(k) for k in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing Triton/shared/native tools required")
WIDTHS = (1, 4, 8, 16, 32, 64)


def maximum(width, scalar):
    return 65536 if width == 8 and not scalar else 256 if width == 4 and not scalar else 31


def capture(root, width, scalar, profile="tensor"):
    from t2m.examples.integer_product_kernels import (
        high_product_values,
        scalar_high_product,
    )

    carrier = 64 if width == 64 else 32
    dtype = f"i{carrier}"
    kernel = scalar_high_product if scalar else high_product_values
    if scalar:
        buffers = (TensorBuffer("out", ("n",), "write", dtype),)
        scalars = (RuntimeScalar("a", dtype, 1), RuntimeScalar("b", dtype, 2),
                   RuntimeScalar("n", "i32", 3, bounds=(0, maximum(width, scalar))))
    else:
        buffers = (TensorBuffer("x", ("n",), "read", dtype), TensorBuffer("y", ("n",), "read", dtype),
                   TensorBuffer("out", ("n",), "write", dtype))
        scalars = (RuntimeScalar("n", "i32", 3, bounds=(0, maximum(width, scalar))),)
    source = root / "wide-source.ttir"
    source.write_text(capture_ttir(kernel, {
        **{buffer.name: "*" + buffer.dtype for buffer in buffers},
        **{value.name: value.dtype for value in scalars}, "B": "constexpr",
    }, {"B": 8}))
    lower_with_shared(source, root / "wide-source", optimizer=os.environ["TRITON_SHARED_OPT"], stage="source")
    module = parse_shared((root / "wide-source/lowered.mlir").read_text())
    # Python tl.umulhi exposes 32/64 only. Construct a new, native-verified
    # TTIR kernel for each legal narrower word, retaining real captured memory
    # and launch structure. The original wide kernel is not an equivalence claim.
    if width != carrier:
        for operation in list(module.walk()):
            if opcode(operation) != "tt.mulhiui":
                continue
            actual = operation.results[0].type
            narrow = (builtin.TensorType(builtin.IntegerType(width), actual.get_shape())
                      if isinstance(actual, builtin.TensorType) else builtin.IntegerType(width))
            parts = [arith.TruncIOp(value, narrow) for value in operation.operands]
            product = builtin.UnregisteredOp.with_name("tt.mulhiui").create(
                operands=[part.result for part in parts], result_types=[narrow])
            extended = arith.ExtUIOp(product.results[0], actual)
            Rewriter.replace_op(operation, [*parts, product, extended], [extended.result])
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    source = root / "source.ttir"
    source.write_text(stream.getvalue())
    inventory = inspect_ttir(source, root / "inspection")
    high = [row for row in inventory["operations"] if row["opcode"] == "tt.mulhiui"]
    source_type = f"i{width}" if scalar else f"tensor<8xi{width}>"
    assert high and all(row["output_types"] == [source_type] for row in high)
    abi = TensorABI(kernel.__name__, buffers, grid=(GridExtent("n", 8),), runtime_scalars=scalars)
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile, address_policy="source-words" if profile == "guarded-buffer" else "structured")


def cases(width, scalar):
    carrier = 64 if width == 64 else 32
    signed, unsigned = getattr(np, f"int{carrier}"), getattr(np, f"uint{carrier}")
    mask = (1 << width) - 1
    samples = [0, 1, mask, max(0, mask-1), 1 << (width-1), (1 << (width-1))-1, 0x12345678 & mask]
    result = []
    for number, n in enumerate((0, 1, 13, maximum(width, scalar))):
        if scalar:
            a = np.asarray(samples[(number+2) % len(samples)], dtype=unsigned).view(signed)
            b = np.asarray(samples[(number+3) % len(samples)], dtype=unsigned).view(signed)
            high = ((int(a) & mask)*(int(b) & mask)) >> width
            expected = np.full(n, high, dtype=unsigned).view(signed)
            arrays = [a, b, np.asarray(n, np.int32)]
        else:
            if width in (4, 8) and n == maximum(width, scalar):
                pairs = list(itertools.product(range(1 << width), repeat=2))
                a, b = (np.asarray(values, dtype=unsigned).view(signed).copy() for values in zip(*pairs, strict=True))
            else:
                a = np.resize(np.asarray(samples, dtype=unsigned), n).view(signed).copy()
                b = np.resize(np.roll(np.asarray(samples, dtype=unsigned), 3), n).view(signed).copy()
            expected = np.asarray([((int(x) & mask)*(int(y) & mask)) >> width
                                   for x, y in zip(a, b, strict=True)], dtype=unsigned).view(signed)
            arrays = [a, b, np.asarray(n, np.int32)]
        result.append((arrays, [expected]))
    return result


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("scalar", (False, True))
@pytest.mark.parametrize("profile", ("tensor", "buffer", "guarded-buffer"))
def test_every_legal_word_width_executes_in_each_memory_profile(tmp_path, width, scalar, profile):
    from t2m.native import execute_native

    result = capture(tmp_path, width, scalar, profile)
    proof = next(row for row in result.trace["transformations"] if row["kind"] == "ttir-unsigned-high-product")
    assert proof["word_bits"] == width
    assert "mului_extended" in result.mlir_text
    for number, (arrays, expected) in enumerate(cases(width, scalar)):
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native-{number}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        assert (actual.shape, actual.dtype, actual.tobytes()) == (
            expected[0].shape, expected[0].dtype, expected[0].tobytes())


@pytest.mark.parametrize("width", (2, 7))
def test_integer_width_outside_the_original_source_definition_refuses(tmp_path, width):
    with pytest.raises(ConversionError, match="original|source|Triton"):
        capture(tmp_path, width, False)
    assert not (tmp_path / "translation/model.mlir").exists()


@pytest.fixture(scope="module", params=[(w, s) for w in WIDTHS for s in (False, True)])
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(k) for k in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin compiler required")
    width, scalar = request.param
    root = tmp_path_factory.mktemp(f"high-i{width}-{scalar}")
    result = capture(root, width, scalar)
    arrays, expected = cases(width, scalar)[2]
    write_shared_bundle(result, root / "capture", arrays, expected, work=root / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    return root, width, scalar


def test_one_merlin_library_retains_all_source_words_and_runtime_shapes(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    root, width, scalar = compiled
    library = _record(root / "merlin/host/model.so")
    for number, (arrays, expected) in enumerate(cases(width, scalar)):
        inputs, reference = root / f"inputs-{number}.npz", root / f"reference-{number}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=expected[0])
        invoke_kernel(root / "merlin", inputs, reference, root / f"invoke-{number}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(root / f"invoke-{number}")
    assert _record(root / "merlin/host/model.so") == library


def test_serial_materialization_preserves_complete_high_words(compiled):
    from test_spike_model import TOOLS as SERIAL_TOOLS
    from test_spike_model import Q

    if not SERIAL_TOOLS:
        pytest.skip("existing cross compiler and Spike required")
    root, width, scalar = compiled
    objects = []
    for number in (0, 2):
        arrays, expected = cases(width, scalar)[number]
        inputs, reference = root / f"target-inputs-{number}.npz", root / f"target-reference-{number}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=expected[0])
        target = root / f"target-{number}"
        report = Q.qualify(root / "capture", target, merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"], inputs=inputs, reference=reference, comparison="words")
        assert report["evaluation"]["result_count"] == 1
        assert Q.verify(target, merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"]) == report
        build = json.loads((target / "build.json").read_text())["build"]
        objects.append(build["host_artifacts"]["object"]["sha256"])
    assert len(set(objects)) == 1
