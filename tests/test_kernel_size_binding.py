"""A real kernel-produced count remains live through dependent allocations."""

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding, bind_model
from t2m.model_bundle import _arguments
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

REQUIRED = all(
    os.environ.get(name)
    for name in (
        "T2M_NATIVE_PYTHON",
        "T2M_MODEL2MLIR_PYTHON",
        "TRITON_SHARED_OPT",
        "T2M_MERLIN_ROOT",
    )
)
pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing Triton/shared/model2MLIR/Merlin tools required"
)


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    root = tmp_path_factory.mktemp("kernel-count-parent")
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_kernel_size_model.py"
    )
    done = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--out",
            str(root / "capture"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=dict(
            os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)
        ),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return root / "capture"


def implementations(root, profile):
    from test_data_size_model import implementation

    from t2m.examples.kernel_size_kernels import positive_count

    root.mkdir(parents=True)
    source = root / "count.ttir"
    source.write_text(
        capture_ttir(
            positive_count,
            {"x": "*i32", "count": "*i64", "capacity": "i64", "B": "constexpr"},
            {"B": 32},
        )
    )
    spec = TensorABI(
        "positive_count",
        (
            TensorBuffer("x", ("capacity",), "read", "i32"),
            TensorBuffer("count", (1,), "write", "i64"),
        ),
        runtime_scalars=(RuntimeScalar("capacity", "i64", 2, bounds=(0, 31)),),
    )
    count = convert_shared(
        source,
        spec,
        root / "count",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    prefix = implementation(root, profile)
    return [
        KernelBinding(
            "t2m_demo::positive_count", count, runtime_operands={"capacity": "capacity"}
        ),
        KernelBinding(
            "t2m_demo::kernel_size_prefix",
            prefix,
            runtime_operands={"n": "n", "capacity": "capacity"},
        ),
    ]


@pytest.fixture(scope="module", params=["tensor", "buffer", "guarded-buffer"])
def selected(tmp_path_factory, parent, request):
    root = tmp_path_factory.mktemp("kernel-count-" + request.param)
    bindings = implementations(root / "translation", request.param)
    source = (parent / "model.mlir").read_text()
    bound = bind_model(source, bindings, result_mode="execution-owned")
    return root, parent, source, bindings, bound


def test_internal_count_binds_without_a_public_input_size_program(selected):
    _root, _parent, source, bindings, bound = selected
    assert bound.trace["schema"] == "t2m.model-kernel-binding.v4"
    assert bound.trace["binding_profile"] == "t2m.execution-checked-calls.v1"
    assert "runtime_shape_contract" not in bound.trace
    assert "model_shape_profile" not in bound.trace
    assert len(bound.trace["calls"]) == 3
    assert len(list(bound.module.ops)) == 1
    assert not any(op.name == "func.call" for op in bound.module.walk())
    with pytest.raises(ConversionError, match="actual parent SSA/type fact"):
        bind_model(source, bindings)


def test_actual_count_and_pack_execute_once_and_preserve_all_original_results(
    selected, monkeypatch
):
    root, parent, _source, _bindings, bound = selected
    monkeypatch.syspath_prepend(str(Path(os.environ["T2M_MERLIN_ROOT"]) / "src"))
    from merlin.compiler.lower.options.owned_results import compile_owned_results

    # Compilation receives source only; neither arguments nor references
    # exist in its build directory or participate in result allocation.
    program = compile_owned_results(
        bound.mlir_text, workdir=root / "compiled", timeout=45
    )
    arrays, _ = _arguments(
        parent,
        bound.module,
        root / "arguments",
        os.environ["T2M_NATIVE_PYTHON"],
        compiler_support_root=os.environ["T2M_MERLIN_ROOT"],
        dynamic_inputs=True,
    )
    snapshots = [array.tobytes() for array in arrays]
    actual, observation = program.execute(arrays)
    assert [array.tobytes() for array in arrays] == snapshots
    assert observation["source_calls"] == 1
    with np.load(parent / "goldens.npz", allow_pickle=False) as saved:
        assert len(actual) == len(saved.files) == 7
        for index, result in enumerate(actual):
            expected = saved[f"out{index}"]
            assert (result.dtype, result.shape, result.tobytes()) == (
                expected.dtype,
                expected.shape,
                expected.tobytes(),
            )
    library = hashlib.sha256(program.library.read_bytes()).hexdigest()
    for capacity, weight in ((2, -100), (3, -1), (31, 0), (13, 100), (31, -29)):
        x = np.arange(capacity, dtype=np.int32)
        parameter = np.asarray(weight, np.int32)
        original = x + parameter
        count = np.asarray([np.count_nonzero(original > 0)], np.int64)
        first = original[: count[0]].copy()
        expected = (
            first,
            first,
            original,
            x.copy(),
            count,
            parameter.copy(),
            first[::2],
        )
        snapshots = (parameter.tobytes(), x.tobytes())
        outputs, observation = program.execute((parameter, x))
        assert (parameter.tobytes(), x.tobytes()) == snapshots
        assert observation["source_calls"] == 1
        for out, reference in zip(outputs, expected, strict=True):
            assert (out.dtype, out.shape, out.tobytes()) == (
                reference.dtype,
                reference.shape,
                reference.tobytes(),
            )
        assert hashlib.sha256(program.library.read_bytes()).hexdigest() == library


def test_new_selection_rejects_unknown_modes(parent):
    with pytest.raises(ValueError, match="unknown explicitly selected"):
        bind_model((parent / "model.mlir").read_text(), [], result_mode="guess")


@pytest.mark.parametrize("failure", ["missing", "ambiguous", "identity", "operands"])
def test_executed_binding_retains_original_call_contract_refusals(selected, failure):
    _root, _parent, source, bindings, _bound = selected
    if failure == "missing":
        bindings = bindings[:1]
    elif failure == "ambiguous":
        bindings = bindings * 2
    elif failure == "identity":
        source = source.replace(
            'prov.kernel = "t2m_demo::positive_count"',
            'prov.kernel = "other::positive_count"',
            1,
        )
    else:
        source = source.replace(
            'prov.kernel_operands = ["x", "capacity"]',
            'prov.kernel_operands = ["x", "wrong"]',
            1,
        )
    with pytest.raises(ConversionError):
        bind_model(source, bindings, result_mode="execution-owned")


def consumer_inputs(selected, monkeypatch):
    _root, _parent, source, bindings, bound = selected
    monkeypatch.syspath_prepend(str(Path(os.environ["T2M_MERLIN_ROOT"]) / "src"))
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    from t2m.executed_model_binding import serialize

    trace = json.loads(json.dumps(bound.trace))
    interfaces = {}
    for binding in bindings:
        implementation = binding.implementation
        digest = implementation.trace["mlir_sha256"]
        interfaces[digest] = (
            trace["kernels"][digest]["spec"],
            serialize(implementation.module),
        )
    return parse_mlir_text(source), parse_mlir_text(bound.mlir_text), trace, interfaces


def test_independent_consumer_reconstructs_every_original_ssa_edge(
    selected, monkeypatch
):
    inputs = consumer_inputs(selected, monkeypatch)
    from merlin.compiler.frontend.intake.execution_model_correspondence import checked_execution_model

    report = checked_execution_model(*inputs)
    assert report["calls"] == 3
    assert report["schema"] == "merlin.executed-model-correspondence.v1"


@pytest.mark.parametrize(
    "damage",
    ["guard", "body", "mapping", "order", "profile", "program", "source", "location"],
)
def test_rehashed_records_cannot_replace_independent_source_body_guard_checks(
    selected, monkeypatch, damage
):
    parent, bound, trace, interfaces = consumer_inputs(selected, monkeypatch)
    from merlin.compiler.frontend.intake.execution_model_correspondence import checked_execution_model
    from xdsl.dialects import arith, builtin

    trace = deepcopy(trace)
    if damage == "guard":
        guard = next(
            op
            for op in bound.walk()
            if op.name == "cf.assert"
            and op.attributes.get("prov.family") == builtin.StringAttr("binding")
        )
        guard.parent.erase_op(guard)
    elif damage == "body":
        old = next(
            op
            for op in bound.walk()
            if op.name == "arith.addi" and "prov.kernel_callsite" not in op.attributes
        )
        changed = arith.SubiOp(*old.operands)
        changed.attributes.update(old.attributes)
        changed.location = old.location
        old.parent.insert_op_before(changed, old)
        old.results[0].replace_all_uses_with(changed.result)
        old.parent.erase_op(old)
    elif damage == "mapping":
        trace["calls"][1]["runtime_operand_arguments"]["n"] = "capacity"
    elif damage == "order":
        trace["calls"][0], trace["calls"][1] = trace["calls"][1], trace["calls"][0]
    elif damage == "profile":
        trace["schema"] = "t2m.model-kernel-binding.v1"
    elif damage == "program":
        trace["runtime_shape_contract"] = {"outputs": []}
    elif damage == "source":
        trace["calls"][0]["source_sha256"] = "0" * 64
    else:
        operation = next(
            op for op in bound.walk() if "prov.kernel_callsite" in op.attributes
        )
        operation.location = builtin.NameLoc(
            builtin.StringAttr("altered source location"), builtin.UnknownLoc()
        )
    bound.verify()
    from t2m.executed_model_binding import serialize

    trace["mlir_sha256"] = hashlib.sha256(serialize(bound).encode()).hexdigest()
    with pytest.raises(ValueError):
        checked_execution_model(parent, bound, trace, interfaces)
