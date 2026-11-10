"""Combined result/effect references retain typed values and exact words."""

import io
import zipfile

import numpy as np
import pytest

from t2m.effect_reference import validate_reference
from t2m.effect_results import load_archive,validate_policy,validate_results,compare_results
from t2m.shared_tensor import TensorABI,TensorBuffer
from t2m.tensor_storage import DTYPES,FLOAT_FORMATS


def reference(comparisons=None):
    return {"schema":"t2m.effect-reference.v2","termination":"return","stdout":"event\n","stderr":"",
        "results":{"archive":"reference.npz","comparisons":comparisons or [{"kind":"words"}]}}


@pytest.mark.parametrize("change",["failure","absent","empty","path","unknown","missing-tolerance","negative","nan","infinity","boolean","huge","extra"])
def test_reference_policies_require_complete_explicit_versions(change):
    ref = reference()
    if change == "failure": ref["termination"] = "assertion_failure"
    elif change == "absent": del ref["results"]
    elif change == "empty": ref["results"]["comparisons"] = []
    elif change == "path": ref["results"]["archive"] = "../reference.npz"
    elif change == "unknown": ref["results"]["comparisons"] = [{"kind":"automatic"}]
    elif change == "missing-tolerance": ref["results"]["comparisons"] = [{"kind":"values"}]
    elif change == "extra": ref["results"]["comparisons"] = [{"kind":"words","rtol":0}]
    else:
        value = {"negative":-1,"nan":float("nan"),"infinity":float("inf"),"boolean":True,"huge":10**400}[change]
        ref["results"]["comparisons"] = [{"kind":"values","rtol":value,"atol":0}]
    with pytest.raises(ValueError):
        validate_reference(ref)


def test_versions_are_explicit_and_old_void_profiles_remain_valid():
    assert validate_reference(reference())["schema"].endswith(".v2")
    v1 = {"schema":"t2m.effect-reference.v1","termination":"return","stdout":"","stderr":""}
    assert validate_reference(v1) == v1
    with pytest.raises(ValueError):
        validate_reference(dict(v1,results=reference()["results"]))


@pytest.mark.parametrize("dtype",list(FLOAT_FORMATS))
@pytest.mark.parametrize("change",["nan-payload","zero-sign","finite-bit","infinity-sign","nan-class"])
def test_ieee_policy_preserves_all_words_except_nan_payload_choice(tmp_path,dtype,change):
    exponent,fraction = FLOAT_FORMATS[dtype]
    width = 1+exponent+fraction
    exp_mask = ((1 << exponent)-1) << fraction
    sign = 1 << (width-1)
    words = np.asarray([sign,0,exp_mask|1,exp_mask|(1 << (fraction-1)),exp_mask,sign|exp_mask,1,exp_mask-1],dtype=f"uint{width}")
    expected_words,actual_words = words.copy(),words.copy()
    if change == "nan-payload": actual_words[2:4] ^= sign|3
    elif change == "zero-sign": actual_words[0] ^= sign
    elif change == "finite-bit": actual_words[-1] ^= 1
    elif change == "infinity-sign": actual_words[4] ^= sign
    elif change == "nan-class": actual_words[2] = exp_mask
    expected_array,actual_array = (value if dtype == "bf16" else value.view(DTYPES[dtype]) for value in (expected_words,actual_words))
    spec = TensorABI("f",(TensorBuffer("out",(len(words),),"write",dtype),))
    ref = reference([{"kind":"ieee"}])
    validate_reference(ref)
    golden,actual = tmp_path / "reference.npz",tmp_path / "outputs.npz"
    np.savez(golden,out0=expected_array);np.savez(actual,out0=actual_array)
    if change == "nan-payload":
        assert compare_results(actual,golden,ref,spec,[]) == 1
    else:
        with pytest.raises(ValueError,match="numeric value reference"):
            compare_results(actual,golden,ref,spec,[])


def test_ieee_policy_cannot_relax_integer_results_or_accept_extra_options():
    spec = TensorABI("f",(TensorBuffer("out",(1,),"write","i64"),))
    with pytest.raises(ValueError,match="declared floating"):
        validate_results(spec,[],[np.asarray([2**53+1],np.int64)],[{"kind":"ieee"}])
    with pytest.raises(ValueError):
        validate_reference(reference([{"kind":"ieee","rtol":0}]))


@pytest.mark.parametrize("change",["missing","extra","duplicate","object","truncated","oversized-header"])
def test_archive_checks_complete_roster_and_sizes_before_allocating(tmp_path,change):
    path = tmp_path / "reference.npz"
    if change in ("missing","extra","object"):
        fields = {"out0":np.asarray([1],np.int64)}
        if change == "missing": fields = {}
        if change == "extra": fields["out1"] = fields["out0"]
        if change == "object": fields["out0"] = np.asarray([object()],object)
        np.savez(path,**fields)
    else:
        stream = io.BytesIO()
        if change == "oversized-header":
            np.lib.format.write_array_header_1_0(stream,{"descr":"<i8","fortran_order":False,"shape":(2**40,)})
            raw = stream.getvalue()
        else:
            np.save(stream,np.asarray([1],np.int64))
            raw = stream.getvalue()
            if change == "truncated": raw = raw[:-1]
        with zipfile.ZipFile(path,"w") as archive:
            archive.writestr("out0.npy",raw)
            if change == "duplicate":
                with pytest.warns(UserWarning,match="Duplicate"):
                    archive.writestr("out0.npy",raw)
    with pytest.raises(ValueError):
        load_archive(path,1)


def test_word_and_value_policies_are_distinct_and_preserve_i64_low_bits(tmp_path):
    spec = TensorABI("f",(TensorBuffer("out",(3,),"write","f32"),TensorBuffer("wide",(1,),"write","i64")))
    words = np.asarray([0x80000000,0x7f800001,0xffc12345],np.uint32).view(np.float32)
    expected = [words,np.asarray([2**53+1],np.int64)]
    ref = reference([{"kind":"words"},{"kind":"values","rtol":0,"atol":0}])
    golden,actual = tmp_path / "reference.npz",tmp_path / "outputs.npz"
    np.savez(golden,out0=expected[0],out1=expected[1])
    np.savez(actual,out0=expected[0],out1=expected[1])
    assert compare_results(actual,golden,ref,spec,[]) == 2
    changed = words.view(np.uint32).copy()
    changed[1] ^= 0x400000
    np.savez(actual,out0=changed.view(np.float32),out1=expected[1])
    with pytest.raises(ValueError,match="word reference"):
        compare_results(actual,golden,ref,spec,[])
    np.savez(actual,out0=expected[0],out1=expected[1]+1)
    with pytest.raises(ValueError,match="value reference"):
        compare_results(actual,golden,ref,spec,[])


@pytest.mark.parametrize("change",["shape","dtype","count","policies","int-tolerance","boolean"])
def test_references_must_match_the_complete_typed_output_abi(change):
    dtype = "i1" if change == "boolean" else "i64"
    spec = TensorABI("f",(TensorBuffer("out",(2,),"write",dtype),))
    arrays = [np.asarray([1,0],np.bool_) if dtype == "i1" else np.asarray([1,2],np.int64)]
    policies = [{"kind":"words"}]
    if change == "shape": arrays[0] = np.asarray([1],np.int64)
    elif change == "dtype": arrays[0] = arrays[0].astype(np.float64)
    elif change == "count": arrays = []
    elif change == "policies": policies = []
    elif change == "int-tolerance": policies = [{"kind":"values","rtol":1,"atol":0}]
    else: arrays[0].view(np.uint8)[0] = 3
    with pytest.raises(ValueError):
        validate_results(spec,[],arrays,policies)


def test_numeric_float_policy_reuses_logical_bfloat_comparison(tmp_path):
    spec = TensorABI("f",(TensorBuffer("out",(2,),"write","bf16"),))
    ref = reference([{"kind":"values","rtol":0,"atol":0.01}])
    validate_policy(ref["results"])
    expected = np.asarray([0x3f80,0x4000],np.uint16)
    np.savez(tmp_path / "reference.npz",out0=expected)
    np.savez(tmp_path / "outputs.npz",out0=np.asarray([0x3f81,0x4000],np.uint16))
    assert compare_results(tmp_path / "outputs.npz",tmp_path / "reference.npz",ref,spec,[]) == 1
