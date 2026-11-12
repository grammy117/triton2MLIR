"""Original CFGs cross official structuralization and the ordinary tensor ABI."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT = os.environ.get("TRITON_SHARED_OPT")
NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(
    os.environ.get(k)
    for k in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
)
pytestmark = pytest.mark.skipif(
    not (OPT and NATIVE), reason="existing shared/native tools required"
)


def capture(root, profile="guarded-buffer", width=64, declared_width=None):
    from t2m.examples.cfg_kernels import early_return_words
    from t2m.frontend import capture_ttir

    source = root / "source.ttir"
    source.write_text(
        capture_ttir(
            early_return_words,
            {
                "x": f"*i{width}",
                "state": f"*i{width}",
                "n": "i32",
                "steps": "i32",
                "mode": "i32",
                "B": "constexpr",
            },
            {"B": 8},
        )
    )
    storage = declared_width or width
    abi = TensorABI(
        "early_return_words",
        (
            TensorBuffer("x", ("n",), dtype=f"i{storage}"),
            TensorBuffer("state", ("n",), "readwrite", f"i{storage}"),
        ),
        grid=(GridExtent("n", 8),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 2, bounds=(0, 31)),
            RuntimeScalar("steps", "i32", 3, bounds=(0, 3)),
            RuntimeScalar("mode", "i32", 4, bounds=(0, 1)),
        ),
    )
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )


def arguments(width=64, n=19, steps=3, mode=0):
    dtype = np.dtype(f"int{width}")
    pattern = np.asarray(
        [
            0,
            -1,
            1,
            -(1 << (width - 1)),
            (1 << (width - 1)) - 1,
            (1 << 53) + 1 if width == 64 else (1 << 24) + 1,
        ],
        dtype=dtype,
    )
    x, state = np.resize(pattern, n).copy(), np.resize(pattern[::-1], n).copy()
    expected = state.copy()
    unsigned = np.dtype(f"uint{width}")
    mask = (1 << width) - 1
    for lane in range(n):
        if (lane // 8 + mode) % 2 == 0:
            continue
        word = int(expected.view(unsigned)[lane])
        for i in range(steps):
            word ^= (int(x.view(unsigned)[lane]) + i + 1) & mask
        expected.view(unsigned)[lane] = word
    return [
        x,
        state,
        *(np.asarray(value, np.int32) for value in (n, steps, mode)),
    ], expected


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_original_early_return_keeps_source_edges_and_state(tmp_path, profile):
    from t2m.native import execute_native

    result = capture(tmp_path, profile)
    assert (
        result.trace["control_flow_normalization"]["schema"]
        == "t2m.control-flow-normalization.v1"
    )
    assert any(
        op["opcode"] == "cf.cond_br" and op["successors"]
        for op in result.trace["source_operations"]
    )
    assert not any(
        op["opcode"] in ("cf.br", "cf.cond_br")
        for op in result.trace["normalized_operations"]
    )
    for index, options in enumerate(
        (
            {},
            {"n": 0},
            {"n": 1, "mode": 1, "steps": 0},
            {"n": 31, "mode": 1, "steps": 2},
        )
    ):
        inputs, expected = arguments(**options)
        actual, _ = execute_native(
            result.mlir_text,
            inputs,
            tmp_path / f"native-{index}",
            python=NATIVE,
            shape_abi=result.spec.runtime_shapes(),
        )
        assert (
            actual[0].dtype == expected.dtype
            and actual[0].tobytes() == expected.tobytes()
        )


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_cfg_backedges_and_phi_values_preserve_zero_trip_numeric_state(
    tmp_path, profile
):
    from t2m.native import execute_native

    source = tmp_path / "source.ttir"
    source.write_text("""module { tt.func public @phi(%x:!tt.ptr<i64>,%state:!tt.ptr<i64>,%steps:i32) {
      %zero=arith.constant 0:i32
      %one=arith.constant 1:i32
      %seed=tt.load %state:!tt.ptr<i64>
      %data=tt.load %x:!tt.ptr<i64>
      cf.br ^loop(%zero,%seed:i32,i64)
    ^loop(%i:i32,%value:i64):
      %continue=arith.cmpi slt,%i,%steps:i32
      cf.cond_br %continue,^body,^done(%value:i64)
    ^body:
      %next=arith.xori %value,%data:i64
      %increment=arith.addi %i,%one:i32
      cf.br ^loop(%increment,%next:i32,i64)
    ^done(%result:i64):
      tt.store %state,%result:!tt.ptr<i64>
      tt.return
    }}""")
    abi = TensorABI(
        "phi",
        (
            TensorBuffer("x", (1,), dtype="i64"),
            TensorBuffer("state", (1,), "readwrite", "i64"),
        ),
        runtime_scalars=(RuntimeScalar("steps", "i32", 2, bounds=(0, 3)),),
    )
    result = convert_shared(
        source,
        abi,
        tmp_path / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    for steps in (0, 1, 3):
        x, state = np.asarray([2**53 + 1], np.int64), np.asarray([-(2**63)], np.int64)
        actual, _ = execute_native(
            result.mlir_text,
            [x, state, np.asarray(steps, np.int32)],
            tmp_path / f"native-{steps}",
            python=NATIVE,
            shape_abi=result.spec.runtime_shapes(),
        )
        assert actual[0].tobytes() == (state ^ x if steps % 2 else state).tobytes()
    if MERLIN:
        import json

        from test_mixed_effects import expected, qualify

        from t2m.invocation import invoke_kernel, verify_invocation

        arrays = [x, state, np.asarray(3, np.int32)]
        policy = expected("", [{"kind": "words"}])
        bundle, compiled = qualify(result, tmp_path, arrays, policy, [state ^ x])
        for steps in (0, 1):
            caller = tmp_path / f"caller-{steps}"
            caller.mkdir()
            np.savez(
                caller / "inputs.npz", in0=x, in1=state, in2=np.asarray(steps, np.int32)
            )
            np.savez(caller / "reference.npz", out0=state ^ x if steps % 2 else state)
            (caller / "reference.json").write_text(json.dumps(policy))
            invoke_kernel(
                compiled,
                caller / "inputs.npz",
                caller / "reference.json",
                tmp_path / f"replay-{steps}",
                merlin_root=os.environ["T2M_MERLIN_ROOT"],
                python=os.environ["T2M_MERLIN_PYTHON"],
            )
            verify_invocation(tmp_path / f"replay-{steps}")
        from test_spike_model import TOOLS

        if TOOLS:
            from scripts import qualify_spike_effect as target

            options = {
                "merlin_root": os.environ["T2M_MERLIN_ROOT"],
                "python": os.environ["T2M_MERLIN_PYTHON"],
            }
            receipt = target.qualify(bundle, tmp_path / "target", **options)
            assert target.verify(tmp_path / "target", **options) == receipt


@pytest.mark.skipif(not MERLIN, reason="existing Merlin tools required")
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
@pytest.mark.parametrize("width", [32, 64])
def test_early_return_packages_retain_cfg_evidence_through_merlin_and_target(
    tmp_path, profile, width
):
    import json

    from test_mixed_effects import expected, qualify

    from t2m.bundle import _record
    from t2m.cfg_intake import FILES
    from t2m.invocation import invoke_kernel, verify_invocation

    result = capture(tmp_path, profile, width)
    arrays, reference = arguments(width)
    policy = expected("", [{"kind": "words"}])
    bundle, compiled = qualify(result, tmp_path, arrays, policy, [reference])
    for name in FILES:
        assert (bundle / name).read_bytes() == (
            compiled / "compiler/source" / name
        ).read_bytes()
    library = _record(compiled / "host/model.so")
    for index, options in enumerate(
        (
            {"n": 0},
            {"n": 1, "mode": 1, "steps": 0},
            {"n": 31, "mode": 1, "steps": 2},
            {"n": 13, "mode": 1, "steps": 3},
        )
    ):
        arrays, reference = arguments(width, **options)
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)}
        )
        np.savez(caller / "reference.npz", out0=reference)
        (caller / "reference.json").write_text(json.dumps(policy))
        invoke_kernel(
            compiled,
            caller / "inputs.npz",
            caller / "reference.json",
            tmp_path / f"replay-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(tmp_path / f"replay-{index}")
        assert _record(compiled / "host/model.so") == library
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_effect as target

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        receipt = target.qualify(bundle, tmp_path / "target", **options)
        assert target.verify(tmp_path / "target", **options) == receipt


def test_original_cfg_audit_does_not_admit_an_incorrect_public_signature(tmp_path):
    from t2m.diagnostics import ConversionError

    with pytest.raises(ConversionError, match="pointer type differs"):
        capture(tmp_path, declared_width=32)


@pytest.mark.parametrize("provider", [None, "ptx-integer-v1"])
def test_unreachable_original_hardware_observations_refuse_before_cfg_rewrites(
    tmp_path, monkeypatch, provider
):
    import t2m.cfg_intake
    from t2m.diagnostics import ConversionError

    source = tmp_path / "source.ttir"
    source.write_text("""module {
      tt.func public @cfg(%out:!tt.ptr<i64>,%enabled:i1) {
        cf.cond_br %enabled,^yes,^done
      ^yes:
        %value=arith.constant 42:i64
        tt.store %out,%value:!tt.ptr<i64>
        cf.br ^done
      ^done: tt.return
      }
      tt.func private @hidden() -> i64 {
        %clock="tt.elementwise_inline_asm"() <{asm_string="mov.u64 $0, %clock64;",constraints="=l",pure=true,packed_element=1:i32}>:()->i64
        tt.return %clock:i64
      }
    }""")
    abi = TensorABI(
        "cfg",
        (TensorBuffer("out", (1,), "readwrite", "i64"),),
        runtime_scalars=(RuntimeScalar("enabled", "i1", 1),),
    )

    def no_normalization(*args, **kwargs):
        raise AssertionError(
            "original source must be audited before CFG canonicalization"
        )

    monkeypatch.setattr(t2m.cfg_intake, "normalize", no_normalization)
    with pytest.raises(
        ConversionError, match="assembly|clock|unknown|profile"
    ) as error:
        convert_shared(
            source,
            abi,
            tmp_path / "translation",
            optimizer=OPT,
            assembly_provider=provider,
            memory_profile="guarded-buffer",
            address_policy="source-words",
        )
    assert error.value.opcode == "tt.elementwise_inline_asm" and error.value.source_id
