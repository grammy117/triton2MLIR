"""Volatile observations survive normalization, unused values and source writes."""

import json
import os
import subprocess
from copy import deepcopy

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared, parse_shared
from t2m.volatile_memory import validate_payload
from t2m.word_events import print_ir

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
REQUIRED = pytest.mark.skipif(not (OPT and NATIVE), reason="existing source/native tools required")


@pytest.mark.skipif(not NATIVE, reason="existing native tool required")
@pytest.mark.parametrize("dtype", ["i64", "f32", "f64"])
def test_real_strided_descriptor_offsets_retain_exact_loaded_words(tmp_path, dtype):
    from t2m.native import execute_native
    from t2m.volatile_memory import lower_load

    module = parse_shared(f'''module {{func.func @forward(%x:tensor<16x{dtype}>)->tensor<2x{dtype}> {{
      %base=bufferization.to_buffer %x read_only:tensor<16x{dtype}> to memref<16x{dtype}>
      %view=memref.subview %base[2][5][2]:memref<16x{dtype}> to memref<5x{dtype},strided<[2],offset:2>>
      %i=arith.constant 1:index
      %j=arith.constant 3:index
      %a=memref.load %view[%i]:memref<5x{dtype},strided<[2],offset:2>>
      %b=memref.load %view[%j]:memref<5x{dtype},strided<[2],offset:2>>
      %result=tensor.from_elements %a,%b:tensor<2x{dtype}>
      return %result:tensor<2x{dtype}>
    }} }}''')
    for op in list(module.walk()):
        if op.name == "memref.load":
            lower_load(op)
    module.verify()
    if dtype == "i64":
        values = np.arange(16, dtype=np.int64) + (1 << 55)
    else:
        word_dtype = np.uint32 if dtype == "f32" else np.uint64
        float_dtype = np.float32 if dtype == "f32" else np.float64
        words = np.arange(16, dtype=word_dtype)
        words[4] = 0x7FA12345 if dtype == "f32" else 0x7FF0123456789ABC
        words[8] = 0x80000000 if dtype == "f32" else 0x8000000000000000
        values = words.view(float_dtype)
    (actual,), _ = execute_native(print_ir(module), [values], tmp_path / "native", python=NATIVE)
    expected = values[[4, 8]]
    assert (actual.dtype, actual.shape, actual.tobytes()) == (expected.dtype, expected.shape, expected.tobytes())


def implementation(root, dtype="i64", *, effect="volatile"):
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.volatile_kernels import volatile_updates
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar

    source = root / "source.ttir"
    source_type = "fp" + dtype[1:] if dtype.startswith("f") else dtype
    source.write_text(capture_ttir(volatile_updates,
        {**{name: "*" + source_type for name in ("x", "state", "old", "seen")},
         "n": "i32", "steps": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("volatile_updates", (TensorBuffer("x", ("n",), dtype=dtype),
        TensorBuffer("state", ("n",), "readwrite", dtype), TensorBuffer("old", ("n",), "write", dtype),
        TensorBuffer("seen", ("n",), "write", dtype)), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)), RuntimeScalar("steps", "i32", 5, bounds=(0, 3))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile=effect)


def arguments(dtype, n, steps, shift=0):
    dt = getattr(np, {"i32": "int32", "i64": "int64", "f32": "float32", "f64": "float64"}[dtype])
    x = np.arange(n, dtype=dt) - 4 + shift
    initial = np.arange(n, dtype=dt) - shift
    state, old, seen = initial.copy(), np.full(n, -7, dtype=dt), np.full(n, -7, dtype=dt)
    for _ in range(steps):
        old = state.copy()
        state = state + x
        seen = state.copy()
    return [x, initial, np.asarray(n, np.int32), np.asarray(steps, np.int32)], [state, old, seen]


@REQUIRED
@pytest.mark.parametrize("dtype", ["i32", "i64", "f32", "f64"])
def test_source_volatile_reads_keep_unused_values_runtime_updates_and_tails(tmp_path, dtype):
    from t2m.native import execute_native

    result = implementation(tmp_path, dtype)
    expected = result.trace["memory"]["volatile_adaptations"]
    assert len(expected) == 4
    assert sum("volatile_read" in row for row in result.trace["source_operations"]) == 4
    validate_payload(result.module, expected)
    assert "func.call" not in result.mlir_text and "__t2m_volatile_identity" not in result.mlir_text
    for i, (n, steps) in enumerate([(0, 3), (1, 0), (13, 2), (31, 3)]):
        arrays, wanted = arguments(dtype, n, steps, i)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native-{i}", python=NATIVE,
            shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, wanted, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
        # Native LLVM printing is stable here; exact leaf count additionally
        # checks that the unused original observation survives legalization.
        lowered = (tmp_path / f"native-{i}/lowered-llvm.mlir").read_text()
        assert lowered.count("llvm.load volatile ") == 4


@REQUIRED
def test_default_profile_cannot_erase_a_source_volatile_request(tmp_path):
    with pytest.raises(ConversionError, match="volatile observations"):
        implementation(tmp_path, effect="ordinary")


@pytest.fixture(scope="module", params=["i64", "f32"])
def selected(tmp_path_factory, request):
    if not (OPT and NATIVE and MERLIN):
        pytest.skip("existing source/native/Merlin tools required")
    root = tmp_path_factory.mktemp("volatile-" + request.param)
    return root, request.param, implementation(root, request.param)


def test_volatile_kernel_reaches_reusable_merlin_host_execution(selected):
    from test_integer_products import replay_library

    root, dtype, result = selected
    replay_library(root, result, [arguments(dtype, n, steps, i)
        for i, (n, steps) in enumerate([(0, 3), (1, 0), (13, 2), (31, 3)])])


def consumer_check(root, result, *, model=None, artifacts=None, trace=None):
    request = root / "consumer-request.json"
    request.write_text(json.dumps({"model": model or result.mlir_text,
        "artifacts": artifacts or result.auxiliary_artifacts, "trace": trace or result.trace}))
    script = r"""import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/"src"))
from merlin.compiler.frontend.intake.kernel_compile_inputs import _checked_memory_members, _checked_word_intent
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
data=json.loads(Path(sys.argv[2]).read_text())
members={k:v.encode() for k,v in data["artifacts"].items()}
try:
    checked=_checked_memory_members(members.__getitem__,data["trace"])
    module=parse_mlir_text(data["model"])
    module.verify()
    _checked_word_intent(next(iter(module.ops)),checked)
except ValueError as error:
    print(json.dumps({"accepted":False,"error":str(error)}))
else:
    print(json.dumps({"accepted":True}))
"""
    done = subprocess.run([os.environ["T2M_MERLIN_PYTHON"], "-c", script,
        os.environ["T2M_MERLIN_ROOT"], str(request)], capture_output=True, text=True, timeout=30,
        env=dict(os.environ, TMPDIR=str(root), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_changed_volatile_flags_descriptor_recipes_and_missing_observations_refuse_both_readers(selected):
    from xdsl.dialects import builtin

    root, _, result = selected
    assert consumer_check(root, result)["accepted"]
    outcomes = []
    for damage in ("volatile-flag", "byte-scale", "stride", "unused-observation", "extra-load"):
        module = result.module.clone()
        phases = [op for op in module.walk() if "prov.volatile_read" in op.attributes]
        phase = phases[1]
        ops = list(phase.regions[0].block.ops)
        load = next(op for op in ops if op.name == "llvm.load")
        if damage == "volatile-flag":
            load.properties.pop("volatile_")
        elif damage == "byte-scale":
            size = next(op for op in ops if op.name == "arith.constant")
            size.properties["value"] = builtin.IntegerAttr(1, builtin.IndexType())
        elif damage == "stride":
            multiply = next(op for op in ops if op.name == "arith.muli")
            multiply.operands = [multiply.operands[0], multiply.operands[0]]
        elif damage == "unused-observation":
            from xdsl.dialects import arith
            type_ = phase.results[0].type
            constant = arith.ConstantOp(builtin.FloatAttr(0.0, type_)) if isinstance(type_, builtin.AnyFloat) else arith.ConstantOp.from_int_and_width(0, type_)
            constant.attributes.update(phase.attributes)
            phase.parent_block().insert_op_before(constant, phase)
            phase.results[0].replace_all_uses_with(constant.results[0])
            phase.parent_block().erase_op(phase)
        else:
            phase.regions[0].block.insert_op_before(load.clone(), load)
        with pytest.raises(ValueError, match="volatile") as producer:
            validate_payload(module, result.trace["memory"]["volatile_adaptations"])
        consumer = consumer_check(root, result, model=print_ir(module))
        assert not consumer["accepted"] and "volatile" in consumer["error"]
        outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": consumer["error"]})
    (root / "payload-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_cleanup_pipeline_and_effect_profile_cannot_be_downgraded(selected):
    from xdsl.dialects import builtin
    from t2m.memory_contract import validate_memory_evidence
    from t2m.source_capsule import record

    root, _, result = selected
    outcomes = []
    for damage in ("cleanup", "memory-cleanup", "profile", "missing-observation", "identity-body"):
        trace, artifacts = deepcopy(result.trace), deepcopy(result.auxiliary_artifacts)
        stage = json.loads(artifacts["word-stage.json"])
        admission = json.loads(artifacts["memory-admission.json"])
        if damage == "cleanup":
            stage["compute"]["native_stage"]["command"].insert(1, "--canonicalize")
        elif damage == "memory-cleanup":
            continuation = json.loads(artifacts["memory-stage.json"])
            continuation["command"].insert(1, "--remove-dead-values")
            artifacts["memory-stage.json"] = json.dumps(continuation)
        elif damage == "profile":
            trace["memory"]["effect_profile"] = "ordinary"
        elif damage == "missing-observation":
            admission["volatile_adaptations"].pop(1)
            trace["memory"]["volatile_adaptations"] = admission["volatile_adaptations"]
        else:
            from xdsl.dialects import arith
            module = parse_shared(stage["compute"]["input"])
            helper = list(module.ops)[1]
            block = helper.body.block
            # A retained marker cannot replace the real identity body.
            constant = arith.ConstantOp.from_int_and_width(0, builtin.i32)
            block.insert_op_before(constant, block.last_op)
            stage["compute"]["input"] = print_ir(module)
            stage["compute"]["native_stage"]["artifacts"]["source.ttir"] = record(stage["compute"]["input"])
        artifacts["word-stage.json"] = json.dumps(stage)
        artifacts["memory-admission.json"] = json.dumps(admission)
        trace["memory"]["word_stage_sha256"] = record(artifacts["word-stage.json"])["sha256"]
        trace["memory"]["admission_sha256"] = record(artifacts["memory-admission.json"])["sha256"]
        with pytest.raises(ValueError) as producer:
            validate_memory_evidence(trace, artifacts)
        consumer = consumer_check(root, result, artifacts=artifacts, trace=trace)
        assert not consumer["accepted"]
        outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": consumer["error"]})
    (root / "stage-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_actual_serial_target_retains_the_selected_volatile_kernel(selected):
    import sys
    from pathlib import Path
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, dtype, _ = selected
    # The saved capture was qualified with the final 31-element, 3-step case.
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        report = Q.qualify(root / "capture", root / "target", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 3
        assert report["status"] == "passed"
    finally:
        sys.path.remove(scripts)
