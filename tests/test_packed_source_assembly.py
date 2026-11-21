"""Complete packed source programs retain word results through Merlin."""

import json
import os
from importlib.metadata import entry_points

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.examples.packed_assembly_reference import program, reference
from t2m.frontend import capture_ttir, inspect_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
from t2m.source_assembly import GROUP

TOOLS = all(os.environ.get(name) for name in (
    "TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT",
    "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
PROVIDER = bool(entry_points(group=GROUP, name="ptx-integer-v1"))
pytestmark = pytest.mark.skipif(not (TOOLS and PROVIDER), reason="existing source/native/Merlin tools required")


def capture(root, width, pack, profile="tensor", mode="independent"):
    from t2m.examples.packed_assembly_kernels import packed_assembly_state

    root.mkdir(parents=True)
    assembly, constraints = program(width, pack, mode)
    source = root / "source.ttir"
    source.write_text(capture_ttir(packed_assembly_state, {
        name: f"*i{width}" if i < 4 else "i32" if i < 6 else "constexpr"
        for i, name in enumerate(packed_assembly_state.arg_names)
    }, {"ASM": assembly, "CONSTRAINTS": constraints, "PACK": pack,
        "B": 8, "WORDS64": width == 64}))
    inspect_ttir(source, root / "original-native")
    abi = TensorABI("packed_assembly_state", tuple(
        TensorBuffer(name, ("n",), "read" if i < 2 else "write", f"i{width}")
        for i, name in enumerate(("x", "y", "out_a", "out_b"))),
        grid=(GridExtent("n", 8),), runtime_scalars=(
            RuntimeScalar("n", "i32", 4, bounds=(0, 67)),
            RuntimeScalar("steps", "i32", 5, bounds=(-1, 3))))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile, address_policy="source-words" if profile == "guarded-buffer" else "structured",
        assembly_provider="ptx-integer-v1")


def arguments(width, n=13, steps=2):
    signed, unsigned = getattr(np, f"int{width}"), getattr(np, f"uint{width}")
    words = [0, 1, (1 << width) - 1, 1 << (width - 1), (1 << (width - 1)) - 1,
             (0xFEDCBA9876543210 if width == 64 else 0xFEDCBA98)]
    a = np.resize(np.asarray(words, dtype=unsigned), n).view(signed).copy()
    b = np.resize(np.roll(np.asarray(words, dtype=unsigned), 2), n).view(signed).copy()
    left, right = reference(a, b, steps)
    return [a, b, np.asarray(n, np.int32), np.asarray(steps, np.int32)], [left, right]


CASES = [(width, pack, profile) for width in (32, 64) for pack in (2, 4)
         for profile in ("tensor", "buffer", "guarded-buffer")]


@pytest.fixture(scope="module", params=CASES)
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    width, pack, profile = request.param
    root = tmp_path_factory.mktemp(f"packed-asm-i{width}-{pack}-{profile}")
    result = capture(root / "source", width, pack, profile)
    report, = [row for row in result.trace["transformations"] if row["kind"] == "source-assembly-implementation"]
    assert report["packed_elements"] == pack
    assert report["packing_proof"]["grouping"] == "independent-of-register-layout"
    assert report["provider_input_types"] == [f"i{width}"] * (2 * pack)
    assert report["provider_output_types"] == [f"i{width}"] * (2 * pack)
    assert not any(row["opcode"] == "tt.elementwise_inline_asm" for row in result.trace["emitted_operations"])
    intent = json.loads(next(iter(result.module.ops)).attributes["prov.source_assembly_contracts"].data)
    assert intent[0]["packed_elements"] == pack and intent[0]["packing_proof"] == report["packing_proof"]
    assert "scf.for" in result.mlir_text
    if profile == "tensor":
        assert "linalg.generic" in result.mlir_text and "memref." not in result.mlir_text
    arrays, expected = arguments(width)
    write_shared_bundle(result, root / "capture", arrays, expected, work=root / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    return root, width


def test_all_original_results_survive_reuse_with_changed_words_shapes_and_controls(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    root, width = compiled
    library = _record(root / "merlin/host/model.so")
    for index, (n, steps) in enumerate(((0, 3), (1, 0), (31, 3), (13, -1))):
        arrays, expected = arguments(width, n, steps)
        inputs, reference = root / f"inputs-{index}.npz", root / f"reference-{index}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        invoke_kernel(root / "merlin", inputs, reference, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(root / f"invoke-{index}")
    assert _record(root / "merlin/host/model.so") == library


def test_serial_images_preserve_all_words_and_the_original_computation_object(compiled):
    from test_spike_model import TOOLS as SERIAL_TOOLS
    from test_spike_model import Q

    if not SERIAL_TOOLS:
        pytest.skip("existing cross compiler and Spike required")
    root, width = compiled
    objects = []
    for index, (n, steps) in enumerate(((0, 3), (13, 2))):
        arrays, expected = arguments(width, n, steps)
        inputs, reference = root / f"target-inputs-{index}.npz", root / f"target-reference-{index}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        target = root / f"target-{index}"
        report = Q.qualify(root / "capture", target, merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"], inputs=inputs, reference=reference, comparison="words")
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(target, merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"]) == report
        objects.append(json.loads((target / "build.json").read_text())["build"]["host_artifacts"]["object"]["sha256"])
    assert len(set(objects)) == 1


@pytest.mark.parametrize("mode", ("coupled", "asymmetric", "hardware"))
def test_required_grouping_and_source_hardware_observations_refuse_before_rewriting(tmp_path, mode):
    with pytest.raises(ConversionError, match="grouping|register|hardware") as error:
        capture(tmp_path / "source", 32, 2, mode=mode)
    assert error.value.opcode == "tt.elementwise_inline_asm" and error.value.source_id
    assert not (tmp_path / "source/translation/model.mlir").exists()


def test_original_source_rejects_a_pack_larger_than_its_logical_tile(tmp_path):
    with pytest.raises(RuntimeError, match="parsing"):
        capture(tmp_path / "source", 32, 16)
    assert not (tmp_path / "source/translation").exists()
