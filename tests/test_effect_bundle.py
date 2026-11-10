"""Real void/failing kernels publish observed effects instead of fake goldens."""

import json
import os
from pathlib import Path
import sys
import shutil

import numpy as np
import pytest

from t2m.bundle import verify_bundle
from t2m.effect_bundle import write_effect_bundle
from t2m.effect_reference import compare_outcome
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI,convert_shared
from t2m.bundle import _record

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = pytest.mark.skipif(not (OPT and NATIVE),reason="existing native toolchain required")


def reference(fail=False,message="constant source assertion"):
    return {"schema":"t2m.effect-reference.v1","termination":"assertion_failure" if fail else "return",
        "stdout":"Assertion `"+message+"` failed\n" if fail else "","stderr":""}


def convert(root,profile,*,fail=False,scalar=False):
    from t2m.frontend import capture_ttir
    from t2m.examples.effect_kernels import constant_check,check_scalar
    kernel = check_scalar if scalar else constant_check
    signature,constants = ({"value":"i32","steps":"i32"},{}) if scalar else ({"FAIL":"constexpr"},{"FAIL":fail})
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel,signature,constants,debug=True))
    scalars = (RuntimeScalar("value","i32",0,bounds=(-9,9)),RuntimeScalar("steps","i32",1,bounds=(0,4))) if scalar else ()
    abi = TensorABI(kernel.__name__,(),runtime_scalars=scalars,effect_only=True,grid=(3,))
    return convert_shared(source,abi,root / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")


@REQUIRED
@pytest.mark.parametrize("profile", ["tensor","buffer","guarded-buffer"])
@pytest.mark.parametrize("fail", [False,True])
def test_actual_zero_argument_entry_returns_or_aborts_with_no_tensor_result(tmp_path,profile,fail):
    result = convert(tmp_path,profile,fail=fail)
    out = tmp_path / "capture"
    write_effect_bundle(result,out,[],reference(fail),work=tmp_path / "native",python=NATIVE)
    receipt = verify_bundle(out)
    assert receipt["materialized_abi"] == {"complete":True,"inputs":0,"outputs":0}
    assert not (out / "golden.npy").exists()
    report = json.loads((out / "execution.json").read_text())
    assert report["reference_comparison"] == "passed" and report["termination"] == reference(fail)["termination"]
    assert (out / "native/outputs.npz").exists() == (not fail)


@REQUIRED
@pytest.mark.parametrize("value,steps,fail", [(-1,0,False),(3,3,False),(-1,3,True)])
def test_effect_expectation_is_independent_of_source_loop_execution(tmp_path,value,steps,fail):
    result = convert(tmp_path,"tensor",scalar=True)
    write_effect_bundle(result,tmp_path / "capture",[np.asarray(value,np.int32),np.asarray(steps,np.int32)],
        reference(fail,"effect scalar must be nonnegative"),work=tmp_path / "native",python=NATIVE)


@REQUIRED
def test_wrong_effect_reference_never_publishes_a_capture_receipt(tmp_path):
    result = convert(tmp_path,"tensor",fail=True)
    with pytest.raises(ValueError,match="differs from its independent reference"):
        write_effect_bundle(result,tmp_path / "capture",[],reference(),work=tmp_path / "native",python=NATIVE)
    assert not (tmp_path / "capture/capture_receipt.json").exists()


def test_setup_exit_cannot_be_reclassified_as_a_source_assertion(tmp_path):
    from t2m.process_outcome import observe
    observe([sys.executable,"-c","raise SystemExit(2)"],tmp_path,environment=dict(os.environ),timeout=5)
    with pytest.raises(ValueError,match="before its actual kernel invocation"):
        compare_outcome(tmp_path,reference(True),backend="native-jit",program={},arguments={},implementation={},result_count=0)


@REQUIRED
def test_initial_failure_retains_real_declared_tensor_results_without_fabricating_values(tmp_path):
    from t2m.examples.effect_kernels import fail_before_store
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorBuffer
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(fail_before_store,{"out":"*i32","B":"constexpr"},{"B":8},debug=True))
    abi = TensorABI("fail_before_store",(TensorBuffer("out",(8,),"write","i32"),))
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT)
    out = tmp_path / "capture"
    write_effect_bundle(result,out,[],reference(True,"failure precedes real output"),work=tmp_path / "native",python=NATIVE)
    receipt = verify_bundle(out)
    assert receipt["materialized_abi"]["outputs"] == 1
    report = json.loads((out / "execution.json").read_text())
    assert report["declared_result_count"] == 1 and report["result_count"] == 0 and report["outputs_published"] is False


@pytest.fixture(scope="module")
def failing_capture(tmp_path_factory):
    if not (OPT and NATIVE):
        pytest.skip("existing native toolchain required")
    root = tmp_path_factory.mktemp("effect-integrity")
    result = convert(root,"tensor",fail=True)
    write_effect_bundle(result,root / "capture",[],reference(True),work=root / "native",python=NATIVE)
    return root / "capture"


@pytest.mark.parametrize("change", ["reference","diagnostic","signal","timeout","phase","published-result"])
def test_rehashed_effect_records_still_need_a_valid_source_outcome(tmp_path,failing_capture,change):
    out = tmp_path / "capture"
    shutil.copytree(failing_capture,out)
    if change in ("reference","diagnostic"):
        ref = reference() if change == "reference" else reference(True,"not a source assertion")
        (out / "reference.json").write_text(json.dumps(ref))
    elif change in ("signal","timeout"):
        outcome = json.loads((out / "native/outcome.json").read_text())
        outcome["returncode" if change == "signal" else "timed_out"] = -11 if change == "signal" else True
        (out / "native/outcome.json").write_text(json.dumps(outcome))
    elif change == "phase":
        phase = json.loads((out / "native/invocation-started.json").read_text())
        phase["arguments"]["sha256"] = "0"*64
        (out / "native/invocation-started.json").write_text(json.dumps(phase))
    else:
        np.savez(out / "native/outputs.npz")
    execution = json.loads((out / "execution.json").read_text())
    execution.update(reference=_record(out / "reference.json"),observation=_record(out / "native/outcome.json"))
    (out / "execution.json").write_text(json.dumps(execution))
    receipt = json.loads((out / "capture_receipt.json").read_text())
    receipt["artifacts"] = {name:_record(out / name) for name in receipt["artifacts"]}
    (out / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_bundle(out)


@REQUIRED
@pytest.mark.parametrize("flag", [False,True])
def test_original_standard_cf_assertion_uses_its_actual_plain_message(tmp_path,flag):
    source = tmp_path / "source.ttir"
    source.write_text('''module {tt.func public @plain_check(%flag:i1) {
      cf.assert %flag, "plain source diagnostic"
      tt.return
    }}''')
    abi = TensorABI("plain_check",(),runtime_scalars=(RuntimeScalar("flag","i1",0),),effect_only=True)
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT)
    ref = {"schema":"t2m.effect-reference.v1","termination":"return" if flag else "assertion_failure",
        "stdout":"" if flag else "plain source diagnostic\n","stderr":""}
    write_effect_bundle(result,tmp_path / "capture",[np.asarray(flag)],ref,work=tmp_path / "native",python=NATIVE)
