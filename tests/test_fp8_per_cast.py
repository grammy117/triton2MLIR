"""Different source casts retain independent executable numeric contracts."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.fp8_casts import POLICIES, PROFILE, SOURCE_POLICY, expand_fp8_casts
from t2m.ir import text
from t2m.shared_tensor import _source_ids, parse_shared
from t2m.tensor_storage import assert_ieee_words
from t2m.examples.fp8_cast_reference import inputs, reference

from test_fp8_casts import CAST_NATIVE, MERLIN


def numeric_source(target="f8E5M2", mode="rtne", n=8):
    casts = []
    for i, policy in enumerate(POLICIES):
        casts.append(f'%r{i}="tt.fp_to_fp"(%x) {{rounding={int(mode=="rtne")}:i32,{SOURCE_POLICY}="{policy}",'
            f'prov.family="quantize",prov.op="cast",prov.region_id="cast{i}"}}:(tensor<{n}xf32>)->tensor<{n}x{target}>')
    return f'module {{func.func @cast(%x:tensor<{n}xf32>)->(tensor<{n}x{target}>,tensor<{n}x{target}>,tensor<{n}x{target}>) {{' + '\n'.join(casts) + f'\nreturn %r0,%r1,%r2:tensor<{n}x{target}>,tensor<{n}x{target}>,tensor<{n}x{target}>}}}}'


def test_profile_expands_each_source_policy_without_using_provenance():
    module = parse_shared(numeric_source().replace('prov.op="cast"', 'prov.op="cast",prov.fp8_overflow="guess"'))
    reports = expand_fp8_casts(module, _source_ids, overflow="per-cast")
    module.verify()
    assert [r["overflow_policy"] for r in reports] == list(POLICIES)
    assert all(r["source_policy_profile"] == PROFILE for r in reports)
    generics = [op for op in module.walk() if op.name == "linalg.generic"]
    assert [op.attributes["prov.fp8_overflow"].data for op in generics] == list(POLICIES)
    assert [op.attributes["prov.region_id"].data for op in generics] == [f"cast{i}" for i in range(3)]
    assert all(SOURCE_POLICY not in op.attributes for op in module.walk())
    assert expand_fp8_casts(module, _source_ids, overflow="per-cast") == []


@pytest.mark.parametrize("target", ("f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ"))
@pytest.mark.parametrize("mode", ("rtne", "rtz"))
@pytest.mark.skipif(not CAST_NATIVE, reason="native verification of shaped arithmetic required")
def test_standard_tensor_carrier_keeps_the_complete_recipe(target, mode):
    from t2m.verify import verify_mlir
    module = parse_shared(numeric_source(target, mode))
    reports = expand_fp8_casts(module, _source_ids, overflow="per-cast", tensor_carrier="arith")
    # xDSL's Cmpf result constraint is scalar-only; native MLIR defines the
    # shaped result. Keep the native verifier authoritative at this seam.
    assert verify_mlir(text(module), python=os.environ["T2M_NATIVE_PYTHON"])["native_mlir_verified"]
    assert [row["overflow_policy"] for row in reports] == list(POLICIES)
    assert all(row["tensor_carrier"] == "standard-elementwise-arith" for row in reports)
    assert all(op.name != "linalg.generic" for op in module.walk())
    parsed = parse_shared(text(module))
    assert len(list(parsed.walk())) == len(list(module.walk()))


@pytest.mark.parametrize("change", ("missing", "unknown", "wrong-type", "provenance", "unselected", "global",
    "identity", "standard", "other-target", "mixed-invalid"))
def test_source_numeric_choices_require_complete_explicit_selection_before_edits(change):
    source = numeric_source()
    if change == "missing": source = source.replace(f'{SOURCE_POLICY}="satfinite",', '')
    elif change == "unknown": source = source.replace(f'{SOURCE_POLICY}="satfinite"', f'{SOURCE_POLICY}="guess"')
    elif change == "wrong-type": source = source.replace(f'{SOURCE_POLICY}="satfinite"', f'{SOURCE_POLICY}=1:i32')
    elif change == "provenance": source = source.replace(SOURCE_POLICY, "prov.fp8_overflow")
    elif change == "identity": source = source.replace("xf32", "xf8E5M2")
    elif change == "standard": source = source.replace('"tt.fp_to_fp"', '"arith.truncf"').replace('rounding=1:i32,', '')
    elif change == "other-target": source = source.replace("xf8E5M2", "xf16")
    elif change == "mixed-invalid": source = source.replace('rounding=1:i32', 'rounding=9:i32', 1)
    module = parse_shared(source)
    before = text(module)
    with pytest.raises(ConversionError):
        expand_fp8_casts(module, _source_ids, overflow=None if change == "unselected" else "ieee" if change == "global" else "per-cast")
    assert text(module) == before


@pytest.mark.skipif(not CAST_NATIVE, reason="existing native FP8 compiler required")
@pytest.mark.parametrize("target", ("f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ"))
@pytest.mark.parametrize("mode", ("rtne", "rtz"))
def test_mixed_policy_casts_execute_against_independent_rational_palettes(tmp_path, target, mode):
    from t2m.native import execute_native
    values = inputs("f32", target)
    module = parse_shared(numeric_source(target, mode, values.size))
    expand_fp8_casts(module, _source_ids, overflow="per-cast")
    actual, _ = execute_native(text(module), [values], tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    for value, policy in zip(actual, POLICIES, strict=True):
        assert_ieee_words(value, reference(values, "f32", target, mode, policy), target)


@pytest.mark.skipif(not os.environ.get("TRITON_SHARED_OPT"), reason="existing shared optimizer required")
@pytest.mark.parametrize("selection,policy", [(None, "ieee"), ("ieee", "ieee"), ("per-cast", "unknown")])
def test_source_contract_is_checked_before_an_unused_helper_can_disappear(tmp_path, selection, policy):
    from t2m.examples.mixed_fp8 import source
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    program = source().replace(f'{SOURCE_POLICY}="ieee"', f'{SOURCE_POLICY}="{policy}"')
    # An independent valid entry keeps the entire quantization helper unused.
    program = program.replace("tt.func @mixed_fp8", "tt.func private @mixed_fp8")
    program = program.rsplit('}', 1)[0] + '''tt.func @entry(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>) {
      %zero=arith.constant 0:i32
      %value=tt.load %x:!tt.ptr<f32>
      tt.store %out,%value:!tt.ptr<f32>
      tt.return
    }}'''
    path = tmp_path / "source.ttir"
    path.write_text(program)
    spec = TensorABI("entry", (TensorBuffer("x", (1,)), TensorBuffer("out", (1,), "write")))
    with pytest.raises(ConversionError) as caught:
        convert_shared(path, spec, tmp_path / "conversion", optimizer=os.environ["TRITON_SHARED_OPT"], fp8_overflow=selection)
    assert caught.value.opcode == "tt.fp_to_fp" and caught.value.source_id.startswith("ttir.op.")
    assert not (tmp_path / "conversion/inline").exists()


@pytest.mark.skipif(not MERLIN, reason="existing shared/native/Merlin tools required")
@pytest.mark.parametrize("target,mode,profile", [("f8E5M2", "rtne", "tensor"),
    ("f8E4M3FN", "rtz", "buffer"), ("f8E4M3FNUZ", "rtne", "guarded-buffer"),
    ("f8E4M3FN", "rtne", "tensor"), ("f8E4M3FNUZ", "rtz", "buffer"), ("f8E5M2", "rtz", "guarded-buffer"),
    ("f8E5M2FNUZ", "rtne", "tensor"), ("f8E5M2FNUZ", "rtz", "guarded-buffer")])
def test_actual_helper_and_rounding_recurrence_reach_merlin(tmp_path, target, mode, profile):
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from t2m.examples.mixed_fp8 import source, reference as recurrence
    from test_mixed_effects import qualify
    path = tmp_path / "source.ttir"
    path.write_text(source(target, mode))
    abi = TensorABI("mixed_fp8", (TensorBuffer("x", ("n",)),
        *(TensorBuffer(name, ("n",), "write") for name in ("out0", "out1", "out2", "raw"))),
        runtime_scalars=(RuntimeScalar("n", "i32", 5, bounds=(0, 1024)), RuntimeScalar("steps", "i32", 6, bounds=(0, 4)),
            RuntimeScalar("delta", "f32", 7)), grid=(GridExtent("n", 32),))
    result = convert_shared(path, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile, address_policy="source-words" if profile == "guarded-buffer" else "structured",
        fp8_overflow="per-cast")
    records = [row for row in result.trace["transformations"] if row.get("source_policy_profile") == PROFILE]
    assert [row["overflow_policy"] for row in records] == list(POLICIES)
    assert any(row["opcode"] == "tt.call" for row in result.trace["source_operations"])
    assert '"scf.for"' in result.mlir_text and '"scf.forall"' in result.mlir_text
    words = np.array([0, 0x80000000, 0x3f808000, 0x43e01000, 0xc3f00000, 0x47800000,
        0x7f800000, 0xff800000, 0x7fa00001, 0x00000001, 0x80000001], np.uint32)
    values = np.resize(words, 67).view(np.float32)
    references = recurrence(values, target, mode, 3, .125)
    expected = {"schema": "t2m.effect-reference.v2", "termination": "return", "stdout": "", "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}] * 3 + [{"kind": "words"}]}}
    capture, qualified = qualify(result, tmp_path, [values, np.array(67, np.int32), np.array(3, np.int32),
        np.array(.125, np.float32)], expected, references)
    for location in (capture / "native/outputs.npz", qualified / "evaluation/outputs.npz"):
        with np.load(location, allow_pickle=False) as actual:
            for index, gold in enumerate(references):
                assert_ieee_words(actual[f"out{index}"], gold, "f32")
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    library = _record(qualified / "host/model.so")
    for case, (n, steps, delta) in enumerate(((0, 4, .25), (1, 0, -.5), (37, 2, .0625))):
        caller = tmp_path / f"caller-{case}"
        caller.mkdir()
        current = np.resize(words, n).view(np.float32)
        np.savez(caller / "inputs.npz", in0=current, in1=np.array(n, np.int32), in2=np.array(steps, np.int32),
            in3=np.array(delta, np.float32))
        (caller / "reference.json").write_text(json.dumps(expected))
        np.savez(caller / "reference.npz", **{f"out{i}": v for i,v in enumerate(recurrence(current, target, mode, steps, delta))})
        out = tmp_path / f"replay-{case}"
        invocation = invoke_kernel(qualified, caller / "inputs.npz", caller / "reference.json", out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert invocation["native_compilation"] == "reused-qualified-library"
        assert _record(qualified / "host/model.so") == library
    from test_spike_model import TOOLS as TARGET_TOOLS
    if TARGET_TOOLS:
        from scripts import qualify_spike_effect as E
        options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
        report = E.qualify(capture, tmp_path / "target", **options)
        assert report["evaluation"]["result_count"] == 4 and E.verify(tmp_path / "target", **options) == report
    if target == "f8E5M2FNUZ" and mode == "rtne":
        process = subprocess.run([sys.executable, "-m", "t2m", "shared-tensor", str(path), "--abi", str(capture / "spec.json"),
            "--out", str(tmp_path / "cli"), "--fp8-overflow", "per-cast", "--inputs", str(capture / "inputs.npz"),
            "--effect-reference", str(capture / "reference.json"), "--python", os.environ["T2M_NATIVE_PYTHON"]],
            capture_output=True, timeout=60, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
        assert process.returncode == 0, process.stderr.decode()
        from t2m.bundle import verify_bundle
        verify_bundle(tmp_path / "cli")
