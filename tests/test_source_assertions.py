"""Real TTIR assertions retain data, failure effects and control placement."""

import os

import numpy as np
import pytest

from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(k) for k in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
REQUIRED = pytest.mark.skipif(not (OPT and NATIVE), reason="existing native toolchain required")


@pytest.mark.parametrize("dtype,message,results", [("i32",'"literal"',"()"),("i1","7:i32","()"),("i1",'"literal"',"i1")])
def test_unqualified_assertion_schemas_are_located_refusals(dtype,message,results):
    from t2m.diagnostics import ConversionError
    from t2m.shared_tensor import parse_shared, opcode
    from t2m.source_assertions import contract
    module = parse_shared(f'''builtin.module {{func.func @f(%c:{dtype}) {{
      "tt.assert"(%c) <{{message={message}}}> : ({dtype})->{results}
      func.return
    }} }}''')
    op = next(op for op in module.walk() if opcode(op) == "tt.assert")
    with pytest.raises(ConversionError) as error:
        contract(op,"ttir.op.bad")
    assert error.value.source_id == "ttir.op.bad" and error.value.opcode == "tt.assert"


def test_source_debug_option_rejects_untyped_values_before_importing_triton():
    from t2m.frontend import capture_ttir
    with pytest.raises(ValueError,match="Boolean source option"):
        capture_ttir(None,{}, {}, debug="true")


def test_embedded_nul_keeps_the_complete_source_diagnostic():
    from xdsl.dialects.builtin import StringAttr
    from t2m.shared_tensor import parse_shared, opcode
    from t2m.source_assertions import contract
    module = parse_shared('''builtin.module {func.func @f(%c:i1) {
      "tt.assert"(%c) <{message="literal"}> : (i1)->()
      func.return
    }}''')
    op = next(op for op in module.walk() if opcode(op) == "tt.assert")
    op.properties["message"] = StringAttr("before\x00after")
    record = contract(op,"ttir.op.nul")
    assert record["diagnostic_text"] == "Assertion `before\x00after` failed\n"
    assert record["source_ids"] == ["ttir.op.nul"]


@pytest.mark.parametrize("opname,key", [("tt.assert","message"),("cf.assert","msg")])
def test_native_escaped_utf8_is_a_lossless_literal_string(opname,key):
    from t2m.shared_tensor import parse_shared,opcode
    from t2m.source_assertions import contract
    source = 'module {func.func @entry(%c:i1) {"'+opname+'"' + r'(%c) <{'+key+r'="before\00\CE\BB\00after"}> : (i1)->() func.return}}'
    module = parse_shared(source)
    record = contract(next(op for op in module.walk() if opcode(op)==opname),"source.utf8")
    assert record["message"] == "before\x00λ\x00after"


@REQUIRED
def test_checked_copy_demo_records_real_source_effects(tmp_path,capsys):
    import json
    from t2m.cli import main
    from t2m.bundle import verify_bundle
    out = tmp_path / "demo"
    assert main(["shared-demo","--kernel","checked-copy","--out",str(out),"--optimizer",OPT,"--python",NATIVE,
                 "--memory-profile","buffer"]) == 0
    assert json.loads(capsys.readouterr().out)["native_numerical_check"] == "passed"
    verify_bundle(out)
    compatibility = json.loads((out / "compatibility.json").read_text())
    assert any("source assertions" in difference for difference in compatibility["differences"])
    with np.load(out / "inputs.npz",allow_pickle=False) as inputs:
        expected = inputs["in1"].copy()
        expected[:13] = inputs["in0"][:13] + 2
    np.testing.assert_array_equal(np.load(out / "golden.npy",allow_pickle=False),expected)


def capture(root, profile, mode=2, rank=1):
    from t2m.examples.assertion_kernels import checked_copy
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    source.write_text(capture_ttir(checked_copy, {"x":"*i32", "out":"*i32", "n":"i32", "steps":"i32", "limit":"i32",
        "B":"constexpr", "MODE":"constexpr", "RANK":"constexpr"}, {"B":8,"MODE":mode,"RANK":rank}, debug=True))
    abi = TensorABI("checked_copy", (TensorBuffer("x",(17,),dtype="i32"), TensorBuffer("out",(17,),"readwrite","i32")), grid=(3,),
        runtime_scalars=(RuntimeScalar("n","i32",2,bounds=(-4,17)), RuntimeScalar("steps","i32",3,bounds=(0,4)),
                         RuntimeScalar("limit","i32",4,bounds=(-100,100))))
    return convert_shared(source,abi,root / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")


def domain(n, steps, *, bad=False):
    x = np.arange(17,dtype=np.int32) % 9 - 4
    if bad:
        x[7] = 99
    initial = np.full((17,),-77,np.int32)
    expected = initial.copy()
    if steps:
        expected[:max(n,0)] = x[:max(n,0)] + ((steps - 1) // 2) * 2
    return [x,initial,np.asarray(n,np.int32),np.asarray(steps,np.int32),np.asarray(5,np.int32)], [expected]


@REQUIRED
@pytest.mark.parametrize("profile", ["tensor","buffer","guarded-buffer"])
@pytest.mark.parametrize("mode,rank", [(0,1),(1,2),(2,1),(3,2)])
def test_native_scalar_and_tensor_assertions_retain_original_loop_branch_masks_and_fills(tmp_path,profile,mode,rank):
    from t2m.native import execute_native
    result = capture(tmp_path,profile,mode,rank)
    assertions = [t for t in result.trace["transformations"] if t["kind"] == "source-fail-stop-assertions"]
    assert len(assertions) == 1
    user_checks = [a for a in assertions[0]["assertions"] if a["message"].startswith("source ")]
    assert len(user_checks) == (2 if mode in (2,3) else 1)
    assert "cf.assert" in result.mlir_text and "prov.source_assertions" in result.mlir_text
    source_ids = {s for a in user_checks for s in a["source_ids"]}
    assert source_ids <= {s for e in result.trace["emitted_operations"] if e["opcode"] == "cf.assert" for s in e["source_ids"]}
    for i,(n,steps) in enumerate([(-2,0),(0,3),(1,1),(13,4),(17,2)]):
        arrays,expected = domain(n,steps)
        outputs,_ = execute_native(result.mlir_text,arrays,tmp_path / f"native-{i}",python=NATIVE)
        np.testing.assert_array_equal(outputs[0],expected[0])
    if mode == 3:
        arrays,expected = domain(-2,3)
        arrays[-1] = np.asarray(-1,np.int32)
        outputs,_ = execute_native(result.mlir_text,arrays,tmp_path / "branch-skipped",python=NATIVE)
        np.testing.assert_array_equal(outputs[0],expected[0])
    failures = ([(-1,False,"source count must be nonnegative")] if mode != 1 else []) + ([(13,True,"source tile exceeds limit")] if mode != 0 else [])
    for i,(n,bad,message) in enumerate(failures):
        arrays,_ = domain(n,1,bad=bad)
        with pytest.raises(ValueError,match=message):
            execute_native(result.mlir_text,arrays,tmp_path / f"failed-{i}",python=NATIVE)


@REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin toolchain required")
@pytest.mark.parametrize("profile", ["tensor","buffer","guarded-buffer"])
def test_actual_merlin_library_keeps_source_assertions_across_invocations(tmp_path,profile):
    from test_integer_products import replay_library
    from t2m.invocation import invoke_kernel
    result = capture(tmp_path,profile,mode=3,rank=2)
    domains = [domain(n,steps) for n,steps in [(-2,0),(0,3),(1,1),(13,4),(17,2)]]
    arrays,expected = domain(-2,3)
    arrays[-1] = np.asarray(-1,np.int32)
    domains.append((arrays,expected))
    replay_library(tmp_path,result,domains)
    for i,(n,bad,message) in enumerate([(-1,False,"source count must be nonnegative"),(13,True,"source tile exceeds limit")]):
        arrays,expected = domain(n,1,bad=bad)
        inputs,reference = tmp_path / f"bad-inputs-{i}.npz", tmp_path / f"bad-reference-{i}.npz"
        np.savez(inputs,**{f"in{j}":array for j,array in enumerate(arrays)})
        np.savez(reference,out0=expected[0])
        out = tmp_path / f"rejected-{i}"
        with pytest.raises(ValueError,match=message):
            invoke_kernel(tmp_path / "merlin",inputs,reference,out,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        assert not (out / "invocation.json").exists()
