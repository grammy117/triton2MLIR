"""One real kernel call returns every tensor and its exact diagnostic stream."""

import json
import os
import shutil

import numpy as np
import pytest

from t2m.bundle import verify_bundle,_record
from t2m.effect_bundle import write_effect_bundle
from t2m.examples.mixed_effect_reference import update_reference
from t2m.frontend import capture_ttir
from t2m.invocation import invoke_kernel,verify_invocation
from t2m.kernel_consumer import qualify_kernel,verify_kernel_qualification
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI,TensorBuffer,convert_shared
from t2m.tensor_storage import DTYPES

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(key) for key in
    ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing native/Merlin environments required")


def expected(stdout,comparisons,*,fail=False):
    reference = {"schema":"t2m.effect-reference.v1" if fail else "t2m.effect-reference.v2",
        "termination":"assertion_failure" if fail else "return","stdout":stdout,"stderr":""}
    if not fail:
        reference["results"] = {"archive":"reference.npz","comparisons":comparisons}
    return reference


def convert_update(root,profile):
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.mixed_effect_kernels import update_and_print
    source = root / "source.ttir"
    source.write_text(capture_ttir(update_and_print,{"x":"*i64","state":"*i64","out":"*i64","n":"i32","steps":"i32",
        "allow":"i1","B":"constexpr"},{"B":8},debug=True))
    abi = TensorABI(update_and_print.__name__,(TensorBuffer("x",("n",),dtype="i64"),
        TensorBuffer("state",("n",),"readwrite","i64"),TensorBuffer("out",("n",),"write","i64")),grid=(GridExtent("n",8),),
        runtime_scalars=(RuntimeScalar("n","i32",3,bounds=(0,17)),RuntimeScalar("steps","i32",4,bounds=(0,3)),
                        RuntimeScalar("allow","i1",5)))
    return convert_shared(source,abi,root / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")


def qualify(result,root,inputs,reference,results):
    capture,qualified = root / "capture",root / "qualified"
    write_effect_bundle(result,capture,inputs,reference,references=results,work=root / "native",python=NATIVE)
    verify_bundle(capture)
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    verify_kernel_qualification(qualified)
    for base,backend in [(capture / "native","native"),(qualified / "evaluation","Merlin")]:
        assert (base / "stdout.bin").read_bytes() == reference["stdout"].encode("utf-8"),backend
        with np.load(base / "outputs.npz",allow_pickle=False) as actual:
            assert actual.files == [f"out{i}" for i in range(len(results))]
            for index,value in enumerate(results):
                if reference["results"]["comparisons"][index]["kind"] == "words":
                    assert actual[f"out{index}"].tobytes() == value.tobytes()
    assert json.loads((qualified / "evaluation/evaluation.json").read_text())["invocations"] == 1
    assert "reference.npz" in json.loads((qualified / "compiler/compile-inputs.json").read_text())["excluded_evaluator_members"]
    return capture,qualified


@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
def test_full_tuple_and_streams_reuse_one_library_across_runtime_shapes_and_failure(tmp_path,profile):
    result = convert_update(tmp_path,profile)
    if profile != "guarded-buffer":
        assert any(record["kind"] == "numeric-control-without-pointer-state" for record in result.trace["transformations"])
    assert "scf.for" in result.mlir_text
    n,steps = 13,3
    x = np.arange(n,dtype=np.int64)+2**53+1
    state = np.arange(n,dtype=np.int64)*3-7
    reference,results = update_reference(x,state,n,steps,True)
    arrays = [x,state,np.asarray(n,np.int32),np.asarray(steps,np.int32),np.asarray(True)]
    capture,qualified = qualify(result,tmp_path,arrays,reference,results)
    report = json.loads((qualified / "host/execution.json").read_text())
    assert report["result_count"] == report["declared_result_count"] == 2
    assert report["result_comparison"] == "passed"
    signature = json.loads((qualified / "compiler/signature.json").read_text())
    assert signature["outputs"] == [{"shape":["n"],"dtype":"i64"},{"shape":["n"],"dtype":"i64"}]
    library = _record(qualified / "host/model.so")
    for index,(n,steps,allow) in enumerate([(0,3,False),(1,0,True),(17,1,True),(13,1,False)]):
        x = np.arange(n,dtype=np.int64)+2**53+1
        state = np.arange(n,dtype=np.int64)*3-7
        reference,results = update_reference(x,state,n,steps,allow)
        caller = tmp_path / f"caller-{index}";caller.mkdir()
        arguments,ref = caller / "inputs.npz",caller / "reference.json"
        np.savez(arguments,in0=x,in1=state,in2=np.asarray(n,np.int32),in3=np.asarray(steps,np.int32),in4=np.asarray(allow))
        ref.write_text(json.dumps(reference))
        if reference["termination"] == "return":
            np.savez(caller / "reference.npz",out0=results[0],out1=results[1])
        out = tmp_path / f"invoke-{index}"
        record = invoke_kernel(qualified,arguments,ref,out,merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert record["native_compilation"] == "reused-qualified-library"
        assert record["result_count"] == (0 if n and not allow else 2)
        assert _record(out / "source/host/model.so") == library
        assert (out / "evaluation/stdout.bin").read_bytes() == reference["stdout"].encode("utf-8")
        assert (out / "evaluation/outputs.npz").exists() == (reference["termination"] == "return")
    assert _record(qualified / "host/model.so") == library


@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
@pytest.mark.parametrize("dtype",["i1","i8","i16","i32","i64","f16","bf16","f32","f64"])
def test_every_existing_tensor_carrier_keeps_printed_and_returned_words(tmp_path,profile,dtype):
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.mixed_effect_kernels import copy_and_print
    from test_source_prints import FLOAT_WORDS
    signature = {"f16":"fp16","f32":"fp32","f64":"fp64"}.get(dtype,dtype)
    if dtype in FLOAT_WORDS:
        bits = 16 if dtype in ("f16","bf16") else int(dtype[1:])
        words = np.asarray(FLOAT_WORDS[dtype][:9],dtype=f"uint{bits}")
        values = words if dtype == "bf16" else words.view(DTYPES[dtype])
    elif dtype == "i1":
        bits,values = 1,np.asarray([False,True,True,False,False,True,False,True,True])
        words = values.astype(np.uint8)
    else:
        bits = int(dtype[1:])
        values = np.asarray([-(1 << (bits-1)),-1,0,(1 << (bits-1))-1,1,7,-3,2,0],dtype=DTYPES[dtype])
        words = values.view(f"uint{bits}")
    n = len(values)
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(copy_and_print,{"x":"*"+signature,"out":"*"+signature,"n":"i32","B":"constexpr"},{"B":8}))
    abi = TensorABI(copy_and_print.__name__,(TensorBuffer("x",("n",),dtype=dtype),TensorBuffer("out",("n",),"write",dtype)),
        grid=(GridExtent("n",8),),runtime_scalars=(RuntimeScalar("n","i32",2,bounds=(0,17)),))
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    lines = []
    for pid in range((n+7)//8):
        for lane in range(8):
            index = pid*8+lane
            word = int(words[index]) if index < n else 0
            lines.append(f"pid ({pid}, 0, 0) idx ({lane}) copy: 0x{word:0{max(1,bits//4)}x}\n")
    qualify(result,tmp_path,[values,np.asarray(n,np.int32)],expected("".join(lines),[{"kind":"words"}]),[values.copy()])


@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
def test_exact_and_approximate_result_policies_compose_with_exact_streams(tmp_path,profile):
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.mixed_effect_kernels import copy_scale_and_print
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(copy_scale_and_print,{"x":"*fp32","out":"*fp32","scaled":"*fp32","n":"i32","B":"constexpr"},{"B":8}))
    abi = TensorABI(copy_scale_and_print.__name__,(TensorBuffer("x",("n",)),TensorBuffer("out",("n",),"write"),
        TensorBuffer("scaled",("n",),"write")),grid=(GridExtent("n",8),),runtime_scalars=(RuntimeScalar("n","i32",3,bounds=(0,17)),))
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    values = (np.arange(13,dtype=np.float32)-5)/np.float32(8)
    scaled = values*np.float32(2)
    lines = [f"pid ({pid}, 0, 0) idx ({lane}) scaled: {float(scaled[pid*8+lane]) if pid*8+lane < len(values) else 0:.6f}\n"
        for pid in range(2) for lane in range(8)]
    ref = expected("".join(lines),[{"kind":"words"},{"kind":"values","rtol":0,"atol":1e-6}])
    approximate = np.nextafter(scaled,np.float32(np.inf))
    capture,qualified = qualify(result,tmp_path,[values,np.asarray(13,np.int32)],ref,[values,approximate])
    from t2m.package_coverage import package_coverage
    report = package_coverage([capture],qualifications=[qualified])
    assert report["packages"][0]["reference_kind"] == "observable-effects-and-tensors"
    assert report["kernel_cases"][0]["reference_kind"] == "observable-effects-and-tensors"
    assert report["packages"][0]["merlin_numeric_policy"] == {"results":ref["results"]["comparisons"]}
    assert report["packages"][0]["native_numeric_policy"] == {"results":ref["results"]["comparisons"]}


@pytest.fixture(scope="module")
def mixed_tuple(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed-tuple")
    result = convert_update(root,"guarded-buffer")
    n,steps = 13,3
    x,state = np.arange(n,dtype=np.int64)+2**53+1,np.arange(n,dtype=np.int64)*3-7
    ref,results = update_reference(x,state,n,steps,True)
    inputs = [x,state,np.asarray(n,np.int32),np.asarray(steps,np.int32),np.asarray(True)]
    capture,qualified = qualify(result,root,inputs,ref,results)
    invoked = root / "invoked"
    invoke_kernel(qualified,capture / "inputs.npz",capture / "reference.json",invoked,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
    return capture,qualified,invoked


def update_json(path,edit):
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def change_observation(root,backend,change):
    directory = root / ("native" if backend == "native" else "evaluation")
    completion = directory / ("execution.json" if backend == "native" else "evaluation.json")
    if change == "streams":
        (directory / "stdout.bin").write_bytes(b"wrong observed stream\n")
        update_json(directory / "outcome.json",lambda value:value.update(stdout=_record(directory / "stdout.bin")))
    elif change == "completion":
        update_json(completion,lambda value:value.update(invocations=2))
    else:
        with np.load(directory / "outputs.npz",allow_pickle=False) as archive:
            values = {name:archive[name].copy() for name in archive.files}
        if change == "low-bit": values["out0"][0] ^= 1
        elif change == "missing-slot": del values["out1"]
        elif change == "extra-slot": values["out2"] = values["out1"]
        elif change == "swapped": values["out0"],values["out1"] = values["out1"],values["out0"]
        elif change == "shape": values["out1"] = values["out1"][:-1]
        elif change == "dtype": values["out1"] = values["out1"].astype(np.float64)
        else: raise AssertionError(change)
        np.savez(directory / "outputs.npz",**values)
        update_json(completion,lambda value:value.update(outputs=_record(directory / "outputs.npz")))


@pytest.mark.parametrize("backend",["native","merlin","reused"])
@pytest.mark.parametrize("change",["low-bit","missing-slot","extra-slot","swapped","shape","dtype","streams","completion"])
def test_rehashed_observation_changes_cannot_hide_missing_values_or_effects(mixed_tuple,tmp_path,backend,change):
    source = mixed_tuple[["native","merlin","reused"].index(backend)]
    target = tmp_path / "changed"
    shutil.copytree(source,target)
    change_observation(target,backend,change)
    filename = {"native":"capture_receipt.json","merlin":"qualification.json","reused":"invocation.json"}[backend]
    def refresh(value):
        value["artifacts"] = {name:_record(target / name) for name in value["artifacts"]}
    update_json(target / filename,refresh)
    verifier = {"native":verify_bundle,"merlin":verify_kernel_qualification,"reused":verify_invocation}[backend]
    with pytest.raises(ValueError,match="reference|roster|shape/ABI|one-call|readonly host call"):
        verifier(target)


@pytest.mark.parametrize("change",["missing","count","shape","dtype","integer-tolerance"])
def test_bad_caller_tensor_references_refuse_before_the_qualified_library_runs(mixed_tuple,tmp_path,monkeypatch,change):
    import t2m.effect_consumer as consumer
    capture,qualified,_ = mixed_tuple
    ref = json.loads((capture / "reference.json").read_text())
    with np.load(capture / "reference.npz",allow_pickle=False) as archive:
        values = {name:archive[name].copy() for name in archive.files}
    if change == "count": del values["out1"]
    elif change == "shape": values["out1"] = values["out1"][:-1]
    elif change == "dtype": values["out1"] = values["out1"].astype(np.float64)
    elif change == "integer-tolerance": ref["results"]["comparisons"][1] = {"kind":"values","rtol":0,"atol":1}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(ref))
    if change != "missing": np.savez(tmp_path / "reference.npz",**values)
    def unexpected(*args,**kwargs):
        raise AssertionError("invalid reference reached the actual library")
    monkeypatch.setattr(consumer,"observe_merlin_host",unexpected)
    out = tmp_path / "invoked"
    with pytest.raises(ValueError,match="bounded regular|complete out0|typed runtime shape/ABI|exact values"):
        invoke_kernel(qualified,capture / "inputs.npz",path,out,merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"])
    assert not out.exists()


def test_shared_tensor_cli_loads_the_complete_independent_result_archive(mixed_tuple,tmp_path,capsys):
    from t2m.cli import main
    capture,_,_ = mixed_tuple
    out = tmp_path / "cli-capture"
    assert main(["shared-tensor",str(capture / "source.ttir"),"--abi",str(capture / "spec.json"),
        "--inputs",str(capture / "inputs.npz"),"--effect-reference",str(capture / "reference.json"),
        "--out",str(out),"--optimizer",OPT,"--python",NATIVE,
        "--memory-profile","guarded-buffer","--address-policy","source-words"]) == 0
    assert json.loads(capsys.readouterr().out)["checked_bundle"] is True
    verify_bundle(out)


@pytest.mark.parametrize("change",["tensor-value","stream"])
def test_wrong_independent_reference_cannot_publish_a_successful_invocation(mixed_tuple,tmp_path,change):
    capture,qualified,_ = mixed_tuple
    ref = json.loads((capture / "reference.json").read_text())
    with np.load(capture / "reference.npz",allow_pickle=False) as archive:
        values = {name:archive[name].copy() for name in archive.files}
    if change == "tensor-value": values["out1"][0] ^= 1
    else: ref["stdout"] += "unexpected event\n"
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(ref))
    np.savez(tmp_path / "reference.npz",**values)
    out = tmp_path / "invoked"
    with pytest.raises(ValueError,match="independent reference|word reference"):
        invoke_kernel(qualified,capture / "inputs.npz",path,out,merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"])
    assert not (out / "invocation.json").exists()
    assert json.loads((out / "evaluation/evaluation.json").read_text())["invocations"] == 1
