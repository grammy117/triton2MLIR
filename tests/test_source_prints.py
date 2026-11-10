"""Logical print data/effects survive actual native and Merlin execution."""

import itertools
import json
import os
import struct

import numpy as np
import pytest

from t2m.bundle import verify_bundle
from t2m.effect_bundle import write_effect_bundle
from t2m.frontend import capture_ttir
from t2m.invocation import invoke_kernel,verify_invocation
from t2m.kernel_consumer import qualify_kernel,verify_kernel_qualification
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI,TensorBuffer,convert_shared

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(key) for key in
    ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
NATIVE_TEST = pytest.mark.skipif(not REQUIRED,reason="existing native/Merlin toolchains required")


@pytest.mark.parametrize("change",["pointer","roster","prefix","hex","result","dynamic"])
def test_unqualified_print_schemas_are_located_refusals(change):
    from t2m.diagnostics import ConversionError
    from t2m.shared_tensor import parse_shared,opcode
    from t2m.source_prints import contract
    from xdsl.dialects import builtin
    type_ = {"pointer":"!tt.ptr<i32>","dynamic":"tensor<?xi32>"}.get(change,"i32")
    source = 'module {func.func @f(%x:'+type_+') {"tt.print"(%x) <{prefix="value",hex=false,isSigned=array<i32:1>}>:('+type_+')->() func.return}}'
    module = parse_shared(source)
    op = next(op for op in module.walk() if opcode(op) == "tt.print")
    if change == "roster":
        op.properties["isSigned"] = builtin.DenseArrayBase.from_list(builtin.i32,[])
    elif change == "prefix":
        op.properties["prefix"] = builtin.IntegerAttr(7,builtin.i32)
    elif change == "hex":
        op.properties["hex"] = builtin.IntegerAttr(0,builtin.i32)
    elif change == "result":
        op = type(op).create(operands=list(op.operands),result_types=[builtin.i32],properties=dict(op.properties))
    with pytest.raises(ConversionError) as error:
        contract(op,"ttir.print.bad")
    assert error.value.source_id == "ttir.print.bad" and error.value.opcode == "tt.print"


def expected(stdout,fail=False):
    return {"schema":"t2m.effect-reference.v1","termination":"assertion_failure" if fail else "return",
        "stdout":stdout,"stderr":""}


def qualify(result,root,arrays,reference):
    capture,qualified = root / "capture",root / "qualified"
    write_effect_bundle(result,capture,arrays,reference,work=root / "native",python=NATIVE)
    verify_bundle(capture)
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    verify_kernel_qualification(qualified)
    assert (capture / "native/stdout.bin").read_bytes() == reference["stdout"].encode("utf-8")
    assert (qualified / "evaluation/stdout.bin").read_bytes() == reference["stdout"].encode("utf-8")
    assert "lower_scalar_diagnostics" in json.loads((qualified / "host/compilation.json").read_text())["features"]
    assert any(op["opcode"] == "tt.print" and "source_print" in op for op in
        result.trace.get("normalized_operations",result.trace["source_operations"]))
    return qualified


def integer_reference(values,n,steps,allow,hex_,width=32):
    lines = []
    for pid in range((n+width-1)//width):
        for step in range(steps):
            if step % 2:
                continue
            for operand in range(4):
                for row,col in itertools.product(range(2),range(width//2)):
                    offset = pid*width+row*(width//2)+col
                    value = int(values[offset]) if offset < n else -7
                    bits = [64,64,8,1][operand]
                    if operand == 1:
                        value %= 2**64
                    elif operand == 2:
                        value %= 256
                    elif operand == 3:
                        value = int(value != 0)
                    printed = f"0x{value % (1 << bits):0{max(1,bits//4)}x}" if hex_ else str(value)
                    lines.append(f"pid ({pid}, 0, 0) idx ({row}, {col:2}) words: (operand {operand}) {printed}\n")
            lines.extend([f"pid ({pid}, 0, 0) idx () control: (operand 0) {steps}\n",
                          f"pid ({pid}, 0, 0) idx () control: (operand 1) {int(allow)}\n",f"pid ({pid}, 0, 0) event\n"])
        if not allow:
            lines.append("Assertion `logical print completed` failed\n")
            break
    return expected("".join(lines),n > 0 and not allow)


@NATIVE_TEST
@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
@pytest.mark.parametrize("hex_",[False,True])
def test_integer_words_coordinates_control_and_one_reused_library(tmp_path,profile,hex_):
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.print_kernels import print_integer_words
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(print_integer_words,{"x":"*i64","n":"i32","steps":"i32","allow":"i1","B":"constexpr","HEX":"constexpr"},
        {"B":32,"HEX":hex_},debug=True))
    abi = TensorABI(print_integer_words.__name__,(TensorBuffer("x",("n",),dtype="i64"),),
        grid=(GridExtent("n",32),),runtime_scalars=(RuntimeScalar("n","i32",1,bounds=(0,65)),
            RuntimeScalar("steps","i32",2,bounds=(0,3)),RuntimeScalar("allow","i1",3)),effect_only=True)
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    base = np.asarray([-(2**63),-1,0,2**63-1,-128,255,127,2**32+1],np.int64)
    values = np.resize(base,33)
    arrays = [values,np.asarray(33,np.int32),np.asarray(1,np.int32),np.asarray(True)]
    qualified = qualify(result,tmp_path,arrays,integer_reference(values,33,1,True,hex_))
    original = (qualified / "host/model.so").read_bytes()
    for index,(n,steps,allow) in enumerate([(0,3,False),(1,0,True),(17,3,True),(17,1,False)]):
        values = np.resize(base,n)
        arguments,reference = tmp_path / f"arguments-{index}.npz",tmp_path / f"reference-{index}.json"
        np.savez(arguments,in0=values,in1=np.asarray(n,np.int32),in2=np.asarray(steps,np.int32),in3=np.asarray(allow))
        reference.write_text(json.dumps(integer_reference(values,n,steps,allow,hex_)))
        invocation = tmp_path / f"invoke-{index}"
        invoke_kernel(qualified,arguments,reference,invocation,merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(invocation)
        assert (invocation / "evaluation/stdout.bin").read_bytes() == json.loads(reference.read_text())["stdout"].encode("utf-8")
    assert (qualified / "host/model.so").read_bytes() == original


FLOAT_WORDS = {
    "f16":[0,0x8000,1,0x7bff,0x7c00,0xfc00,0x7e23,0xfe11,0x7c01,0x3c00,0xbc00,0x3555],
    "bf16":[0,0x8000,1,0x7f7f,0x7f80,0xff80,0x7fc3,0xffd1,0x7f81,0x3f80,0xbf80,0x3eab],
    "f32":[0,0x80000000,1,0x7f7fffff,0x7f800000,0xff800000,0x7fc01234,0xffc06543,0x7f800001,0x3f800000,0xbf800000,0x3eaaaaab],
    "f64":[0,0x8000000000000000,1,0x7fefffffffffffff,0x7ff0000000000000,0xfff0000000000000,0x7ff8000012345678,
           0xfff8000065432100,0x7ff0000000000001,0x3ff0000000000000,0xbff0000000000000,0x3fd5555555555555],
}


@NATIVE_TEST
@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
@pytest.mark.parametrize("dtype",["f16","bf16","f32","f64"])
@pytest.mark.parametrize("hex_",[False,True])
def test_float_words_fixed_decimal_and_multiaxis_logical_programs(tmp_path,profile,dtype,hex_):
    from t2m.examples.print_kernels import print_float_tiles
    bits = {"f16":16,"bf16":16,"f32":32,"f64":64}[dtype]
    words = np.asarray(FLOAT_WORDS[dtype],dtype=f"uint{bits}")
    values = words if dtype == "bf16" else words.view({"f16":np.float16,"f32":np.float32,"f64":np.float64}[dtype])
    source = tmp_path / "source.ttir"
    signature = {"f16":"fp16","bf16":"bf16","f32":"fp32","f64":"fp64"}[dtype]
    source.write_text(capture_ttir(print_float_tiles,{"x":"*"+signature,"n":"i32","B":"constexpr","HEX":"constexpr"},
        {"B":16,"HEX":hex_}))
    abi = TensorABI(print_float_tiles.__name__,(TensorBuffer("x",("n",),dtype=dtype),),grid=(2,2),
        runtime_scalars=(RuntimeScalar("n","i32",1,bounds=(0,16)),),effect_only=True)
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    lines = []
    for px,py in itertools.product(range(2),range(2)):
        for index in range(16):
            word = int(words[index]) if index < len(words) else 0
            if hex_:
                printed = f"0x{word:0{bits//4}x}"
            else:
                if dtype == "bf16":
                    value = struct.unpack("<f",(word << 16).to_bytes(4,"little"))[0]
                else:
                    value = struct.unpack("<"+{16:"e",32:"f",64:"d"}[bits],word.to_bytes(bits//8,"little"))[0]
                if np.isnan(value):
                    printed = "-nan" if word >> (bits-1) else "nan"
                else:
                    printed = format(value,".6f")
            lines.append(f"pid ({px}, {py}, 0) idx ({index//8}, {index%8}) values: {printed}\n")
    qualify(result,tmp_path,[values,np.asarray(len(words),np.int32)],expected("".join(lines)))


@NATIVE_TEST
@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
def test_saved_literal_prefix_is_data_with_nul_unicode_and_percent(tmp_path,profile):
    source = tmp_path / "source.ttir"
    source.write_text(r'''module {tt.func public @literal() {
      tt.print " before\00\CE\BB %s %d" {hex=false,isSigned=array<i32>}
      tt.return
    }}''')
    abi = TensorABI("literal",(),effect_only=True,grid=(2,))
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,memory_profile=profile)
    qualify(result,tmp_path,[],expected("".join(f"pid ({pid}, 0, 0) before\x00λ %s %d\n" for pid in range(2))))


@NATIVE_TEST
def test_source_word_formatting_and_grid_do_not_depend_on_provenance(tmp_path):
    from t2m.ir import text
    from t2m.native import observe_native
    source = tmp_path / "source.ttir"
    source.write_text(r'''module {tt.func public @raw(%value:i64) {
      tt.print " word: " {hex=true,isSigned=array<i32:0>} : %value : i64
      tt.return
    }}''')
    abi = TensorABI("raw",(),grid=(2,),runtime_scalars=(RuntimeScalar("value","i64",0),),effect_only=True)
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT)
    stripped = result.module.clone()
    for op in stripped.walk():
        for key in list(op.attributes):
            if key.startswith("prov."):
                del op.attributes[key]
    native = tmp_path / "stripped"
    observe_native(text(stripped),[np.asarray(-1,np.int64)],native,python=NATIVE)
    assert json.loads((native / "outcome.json").read_text())["returncode"] == 0
    assert (native / "stdout.bin").read_bytes() == b"".join(
        f"pid ({pid}, 0, 0) idx () word: 0xffffffffffffffff\n".encode() for pid in range(2))


@NATIVE_TEST
@pytest.mark.parametrize("name",["printf","putchar","abort"])
def test_source_entry_spelling_cannot_shadow_the_native_runtime(tmp_path,name):
    source = tmp_path / "source.ttir"
    source.write_text('module {tt.func public @'+name+r'''(%value:i32) {
      tt.print " value: " {hex=false,isSigned=array<i32:1>} : %value : i32
      tt.return
    }}''')
    abi = TensorABI(name,(),runtime_scalars=(RuntimeScalar("value","i32",0),),effect_only=True)
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT)
    qualify(result,tmp_path,[np.asarray(-71,np.int32)],expected("pid (0, 0, 0) idx () value: -71\n"))


@NATIVE_TEST
@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
@pytest.mark.parametrize("hex_",[False,True])
def test_narrow_integer_signedness_and_wrapping(tmp_path,profile,hex_):
    from t2m.examples.print_kernels import print_narrow_integer_words
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(print_narrow_integer_words,{"x":"*i64","B":"constexpr","HEX":"constexpr"},
        {"B":8,"HEX":hex_}))
    abi = TensorABI(print_narrow_integer_words.__name__,(TensorBuffer("x",(8,),dtype="i64"),),effect_only=True)
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    values = np.asarray([-(2**63),-1,0,127,128,65535,2**31+2,2**63-1],np.int64)
    lines = []
    for operand,(bits,signed) in enumerate([(8,True),(16,True),(32,True),(16,False),(32,False)]):
        for index,value in enumerate(values):
            word = int(value) % (1 << bits)
            number = word - (1 << bits) if signed and word >= 1 << (bits-1) else word
            printed = f"0x{word:0{bits//4}x}" if hex_ else str(number)
            lines.append(f"pid (0, 0, 0) idx ({index}) narrow: (operand {operand}) {printed}\n")
    qualify(result,tmp_path,[values],expected("".join(lines)))


@NATIVE_TEST
def test_prints_compose_with_real_integer_cas_and_preserve_declared_results(tmp_path):
    from t2m.examples.print_kernels import print_atomic_old
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(print_atomic_old,{"out":"*i32","allow":"i1"},{},debug=True))
    abi = TensorABI(print_atomic_old.__name__,(TensorBuffer("out",(1,),"readwrite","i32"),),
        runtime_scalars=(RuntimeScalar("allow","i1",1),))
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,memory_profile="guarded-buffer",
        address_policy="source-words",effect_profile="typed-atomics")
    reference = expected("pid (0, 0, 0) idx () old: 1\nAssertion `atomic print completed` failed\n",True)
    qualified = qualify(result,tmp_path,[np.asarray([1],np.int32),np.asarray(False)],reference)
    signature = json.loads((qualified / "compiler/signature.json").read_text())
    assert signature["outputs"] == [{"shape":[1],"dtype":"i32"}]
    features = json.loads((qualified / "host/compilation.json").read_text())["features"]
    assert "lower_integer_compare_exchange" in features
    assert not (qualified / "evaluation/outputs.npz").exists()


@NATIVE_TEST
@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
def test_empty_static_print_operand_has_zero_logical_events(tmp_path,profile):
    source = tmp_path / "source.ttir"
    source.write_text('''module {tt.func public @empty() {
      %v=arith.constant dense<> : tensor<0xi32>
      tt.print " empty: " {hex=false,isSigned=array<i32:1>} : %v : tensor<0xi32>
      tt.return
    }}''')
    result = convert_shared(source,TensorABI("empty",(),effect_only=True,grid=(3,)),tmp_path / "translation",
        optimizer=OPT,memory_profile=profile)
    qualify(result,tmp_path,[],expected(""))
