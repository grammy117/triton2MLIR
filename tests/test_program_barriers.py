"""Program rendezvous retain complete memory phases and cross-lane values."""

import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.program_barriers import contract, expand
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import (
    TensorABI,
    TensorBuffer,
    _preflight,
    _source_ids,
    opcode,
    parse_shared,
)


def source(mask=31, *, attributes="", body="", wrapper=""):
    return parse_shared(f'''"builtin.module"() ({{
      "tt.func"() <{{sym_name="f", function_type=(!tt.ptr<i64>)->(), sym_visibility="public"}}> ({{
      ^bb0(%p:!tt.ptr<i64>):
        {body}
        "ttg.barrier"() <{{addrSpace={mask}:i32}}> {attributes} : ()->()
        "tt.return"() : ()->()
      }}) : ()->()
    }}) {wrapper} : ()->()''')


@pytest.mark.parametrize("mask", [0, 1, 6, 31])
def test_exact_barrier_scope_is_retained_with_an_executable_phase(mask):
    module = source(mask, body='%x="arith.constant"() <{value=2:i64}> : ()->i64')
    inventory = _preflight(module, source_entry="f")
    barrier = next(row for row in inventory if row["opcode"] == "ttg.barrier")
    reports = expand(module, _source_ids)
    phase = next(op for op in module.walk() if op.name == "scf.execute_region")
    assert phase.attributes["no_inline"] is not None
    assert reports[0]["contract"] == barrier["program_barrier"]
    assert reports[0]["contract"]["address_space_mask"] == mask
    assert any(op.name == "arith.constant" for op in phase.walk())
    assert not any(opcode(op) == "ttg.barrier" for op in module.walk())


@pytest.mark.parametrize("mask", [-1, 32])
def test_unknown_address_space_masks_are_located_refusals(mask):
    op = next(op for op in source(mask).walk() if opcode(op) == "ttg.barrier")
    with pytest.raises(ConversionError) as error:
        contract(op, "source.bad")
    assert error.value.source_id == "source.bad"


def test_multi_cta_participation_does_not_acquire_single_program_semantics():
    module = source(wrapper='{ "ttg.num-ctas"=2:i32 }')
    _preflight(module, source_entry="f")
    with pytest.raises(ConversionError, match="cluster contract"):
        expand(module, _source_ids)


def test_native_no_inline_property_survives_structural_transport():
    module = parse_shared('''module {func.func @f() {
        "scf.execute_region"() <{no_inline}> ({"scf.yield"():()->()}) : ()->()
        func.return
    }}''')
    module.verify()
    phase = next(op for op in module.walk() if op.name == "scf.execute_region")
    assert "no_inline" in phase.attributes and "no_inline" not in phase.properties


def test_a_barrier_in_a_scalar_combiner_has_no_whole_program_participation():
    module = parse_shared('''module {func.func @f(%x:i64) -> i64 {
        %r = "linalg.generic"(%x) ({^bb0(%v:i64):
          "ttg.barrier"() <{addrSpace=31:i32}> : ()->()
          "linalg.yield"(%v):(i64)->()
        }) : (i64)->i64
        func.return %r:i64
    }}''')
    op = next(op for op in module.walk() if opcode(op) == "ttg.barrier")
    with pytest.raises(ConversionError, match="whole-program structured control"):
        contract(op, "source.combiner")


def test_source_only_audit_rejects_other_physical_operations_before_dead_code_elimination():
    module = source(body='"ttg.async_wait"() : ()->()')
    with pytest.raises(ConversionError) as error:
        _preflight(module, source_entry="f")
    assert error.value.opcode == "ttg.async_wait" and error.value.source_id is not None


def test_successive_source_barriers_keep_separate_ordered_phases():
    module = source(body='''%x="arith.constant"() <{value=2:i64}> : ()->i64
        "ttg.barrier"() <{addrSpace=6:i32}> : ()->()
        %y="arith.addi"(%x,%x):(i64,i64)->i64''')
    _preflight(module, source_entry="f")
    reports = expand(module, _source_ids)
    phases = [op for op in module.walk() if op.name == "scf.execute_region"]
    assert len(phases) == len(reports) == 2 and phases[0].parent_block() is phases[1].parent_block()
    assert phases[1].regions[0].block.first_op.operands[0] is phases[0].results[0]


OPT = os.environ.get("TRITON_SHARED_OPT")
NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
TOOLS = pytest.mark.skipif(not (OPT and NATIVE), reason="existing source/native tools required")


def capture(root, profile, *, cross=False):
    from t2m.examples.barrier_kernels import exchange_tile
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import convert_shared

    source = root / "source.ttir"
    source.write_text(capture_ttir(exchange_tile,
        {"state": "*i64", "out": "*i64", "n": "i32", "steps": "i32", "B": "constexpr", "CROSS": "constexpr"},
        {"B": 4, "CROSS": cross}))
    abi = TensorABI("exchange_tile", (TensorBuffer("state", ("n", 4), "readwrite", "i64"),
        TensorBuffer("out", ("n", 4), "write", "i64")), grid=(GridExtent("n", 4),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 19)),
                         RuntimeScalar("steps", "i32", 3, bounds=(0, 4))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile=profile)


def arguments(n, steps):
    initial = np.arange(n * 4, dtype=np.int64).reshape(n, 4) + (1 << 55)
    state = initial + steps * (steps + 1) // 2
    out = np.full((n, 4), -7, dtype=np.int64)
    if steps:
        for origin in range(0, n, 4):
            tile = np.full((4, 4), -3, dtype=np.int64)
            size = min(4, n - origin)
            tile[:size] = state[origin:origin + size]
            out[origin:origin + size] = tile.T[:size]
    return [initial, np.asarray(n, np.int32), np.asarray(steps, np.int32)], [state, out]


@TOOLS
@pytest.mark.parametrize("profile", ["tensor", "guarded-buffer"])
def test_real_loop_exchange_retains_phases_and_native_values(tmp_path, profile):
    from t2m.native import execute_native

    result = capture(tmp_path, profile)
    source = [row for row in result.trace["source_operations"] if row["opcode"] == "ttg.barrier"]
    reports = [row for row in result.trace["transformations"] if row["kind"] == "synchronous-program-memory-phase"]
    assert len(source) == len(reports) == 2
    assert "scf.execute_region" in result.mlir_text
    assert "ttg." not in result.mlir_text and "gpu." not in result.mlir_text
    assert "scf.forall" in result.mlir_text
    for i, (n, steps) in enumerate([(0, 0), (1, 1), (4, 2), (7, 4), (19, 0)]):
        arrays, expected = arguments(n, steps)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native-{i}",
                                    python=NATIVE, shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


@TOOLS
@pytest.mark.skipif(not all(os.environ.get(k) for k in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")),
                    reason="existing Merlin toolchain required")
@pytest.mark.parametrize("profile", ["tensor", "guarded-buffer"])
def test_saved_merlin_exchange_library_retains_runtime_phases(tmp_path, profile):
    from test_integer_products import replay_library

    result = capture(tmp_path, profile)
    replay_library(tmp_path, result, [arguments(n, steps) for n, steps in [(7, 2), (0, 4), (1, 0), (4, 4), (19, 3)]])


@TOOLS
def test_program_barrier_cannot_authorize_a_cross_program_readwrite_dependency(tmp_path):
    with pytest.raises(ConversionError, match="owned tile|bounds"):
        capture(tmp_path, "tensor", cross=True)


@TOOLS
def test_guarded_program_barrier_keeps_the_actual_cross_program_conflict_abort(tmp_path):
    import signal

    from t2m.native import observe_native

    result = capture(tmp_path, "guarded-buffer", cross=True)
    arrays, _ = arguments(8, 1)
    work = tmp_path / "conflict"
    outcome = observe_native(result.mlir_text, arrays, work, python=NATIVE,
                             shape_abi=result.spec.runtime_shapes())
    assert outcome["returncode"] == -signal.SIGABRT and not outcome["timed_out"]
    diagnostic = (work / "stdout.bin").read_bytes() + (work / "stderr.bin").read_bytes()
    assert b"Triton memory conflicts across logical programs" in diagnostic
    assert not (work / "outputs.npz").exists()


@TOOLS
@pytest.mark.skipif(not all(os.environ.get(k) for k in ("MERLIN_SPIKE", "MERLIN_RISCV_GCC", "T2M_MERLIN_ROOT")),
                    reason="existing serial target tools required")
@pytest.mark.parametrize("profile", ["tensor", "guarded-buffer"])
def test_actual_serial_target_executes_cross_lane_phases(tmp_path, profile):
    import importlib.util
    import sys
    from pathlib import Path

    from t2m.shared_bundle import write_shared_bundle

    result = capture(tmp_path, profile)
    arrays, expected = arguments(7, 3)
    bundle = tmp_path / "capture"
    write_shared_bundle(result, bundle, arrays, expected, work=tmp_path / "native", python=NATIVE)
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("barrier_target_qualification", scripts / "qualify_spike_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.qualify(bundle, tmp_path / "spike", merlin_root=os.environ["T2M_MERLIN_ROOT"],
                                python=os.environ["T2M_MERLIN_PYTHON"], comparison="words")
        assert report["status"] == "passed"
    finally:
        sys.path.remove(str(scripts))
