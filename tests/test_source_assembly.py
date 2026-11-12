"""Source implementation selection and its independent portable payload gates."""

import json
import os
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from xdsl.dialects import arith, builtin
from xdsl.ir import Block

from t2m.diagnostics import ConversionError
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
from t2m.source_assembly import GROUP, AssemblyExpansion

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
PROVIDER = bool(entry_points(group=GROUP, name="ptx-integer-v1"))
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not OPT, reason="existing pinned optimizer required")


def source(root, *, pure=True):
    path = root / "source.ttir"
    path.write_text('''module {tt.func public @assembly(%x:!tt.ptr<i32>,%out:!tt.ptr<i32>) {
      %v=tt.load %x:!tt.ptr<i32>
      %w="tt.elementwise_inline_asm"(%v) <{asm_string="add.u32 $0, $1, 1;",
        constraints="=r,r", pure=''' + ("true" if pure else "false") + ''', packed_element=1:i32}>:(i32)->i32
      tt.store %out,%w:!tt.ptr<i32>
      tt.return
    }}''')
    abi = TensorABI("assembly", (TensorBuffer("x", (1,), dtype="i32"), TensorBuffer("out", (1,), "write", "i32")))
    return path, abi


def convert(path, abi, work, **options):
    return convert_shared(path, abi, work, optimizer=OPT, memory_profile="guarded-buffer", address_policy="source-words", **options)


def test_default_intake_still_refuses_undeclared_source_semantics(tmp_path):
    path, abi = source(tmp_path)
    with pytest.raises(ConversionError, match="separately qualified") as exc:
        convert(path, abi, tmp_path / "translation")
    assert exc.value.opcode == "tt.elementwise_inline_asm" and exc.value.source_id


@pytest.mark.skipif(not PROVIDER, reason="optional source provider required")
@pytest.mark.parametrize("provider",[None,"ptx-integer-v1"])
def test_unused_source_implementation_refuses_before_composite_dot_normalization(tmp_path,monkeypatch,provider):
    from t2m import shared_tensor
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.dot_precision_kernels import precision_tiled_dot
    from t2m.frontend import capture_ttir
    from t2m.ir import text
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import opcode,parse_shared
    from t2m.upstream import lower_with_shared
    from xdsl.ir import Region

    path = tmp_path / "source.ttir"
    path.write_text(capture_ttir(precision_tiled_dot,
        {name:"*fp32" if i < 4 else "i32" if i < 7 else "constexpr" for i,name in enumerate(precision_tiled_dot.arg_names)},
        {"BM":16,"BN":16,"BK":32,"PRECISION":"bf16x3"}))
    lower_with_shared(path,tmp_path / "generic",optimizer=OPT,stage="source")
    module = parse_shared((tmp_path / "generic/lowered.mlir").read_text())
    block = Block(arg_types=[builtin.i32])
    asm = builtin.UnregisteredOp.with_name("tt.elementwise_inline_asm").create(
        operands=[block.args[0]],result_types=[builtin.i32],properties={
            "asm_string":builtin.StringAttr("mov.u32 $0, %clock;"),"constraints":builtin.StringAttr("=r,r"),
            "pure":builtin.IntegerAttr.from_bool(True),"packed_element":builtin.IntegerAttr(1,builtin.i32)})
    block.add_ops([asm,builtin.UnregisteredOp.with_name("tt.return").create(operands=list(asm.results))])
    helper = builtin.UnregisteredOp.with_name("tt.func").create(properties={
        "sym_name":builtin.StringAttr("unused_hardware_observation"),"sym_visibility":builtin.StringAttr("private"),
        "function_type":builtin.FunctionType.from_lists([builtin.i32],[builtin.i32])},regions=[Region(block)])
    module.body.block.add_op(helper)
    assert any(opcode(op) == "tt.dot" for op in module.walk())
    path.write_text(text(module))
    abi = TensorABI("precision_tiled_dot",(TensorBuffer("x",("m","k")),TensorBuffer("y",("k","n")),
        TensorBuffer("initial",("m","n")),TensorBuffer("out",("m","n"),"write")),
        grid=(GridExtent("m",16),GridExtent("n",16)),runtime_scalars=tuple(RuntimeScalar(name,"i32",i,bounds=(0,65))
            for i,name in enumerate(("m","n","k"),4)))
    monkeypatch.setattr(shared_tensor,"expand_tensor_pointers",lambda *_a,**_kw:pytest.fail("source implementation reached normalization before decoding"))
    with pytest.raises(ConversionError,match="qualified|hardware|instruction|operand|register"):
        convert(path,abi,tmp_path / "translation",assembly_provider=provider)


def test_missing_provider_cannot_become_an_opaque_success(tmp_path):
    path, abi = source(tmp_path)
    with pytest.raises(ValueError, match="exactly one installed"):
        convert(path, abi, tmp_path / "translation", assembly_provider="missing-provider")


def test_changed_installed_files_cannot_relabel_a_cached_provider(tmp_path, monkeypatch):
    from t2m import source_assembly

    implementation = tmp_path / "provider.py"
    implementation.write_text("# initial implementation\n")

    class Provider:
        api_version = 1

        def lower(self, request, arguments):
            return AssemblyExpansion([], arguments, {})

    distribution = SimpleNamespace(files=[Path("provider.py")], metadata={"Name": "test-provider"}, version="1",
                                   locate_file=lambda member: tmp_path / member)
    point = SimpleNamespace(dist=distribution, value="provider:Provider", load=lambda: Provider)
    monkeypatch.setattr(source_assembly, "entry_points", lambda **kwargs: [point])
    monkeypatch.setattr(source_assembly, "_LOADED", {})
    _, identity = source_assembly.load_provider("test")
    assert identity["files"]["provider.py"]["bytes"] == implementation.stat().st_size
    implementation.write_text("# changed implementation\n")
    with pytest.raises(ValueError, match="changed after import"):
        source_assembly.load_provider("test")


@pytest.mark.parametrize("kind", ["llvm", "unknown-arith", "free-value", "wrong-type", "parented"])
def test_provider_cannot_change_the_closed_standard_value_and_effect_boundary(tmp_path, monkeypatch, kind):
    from t2m import source_assembly

    class InvalidProvider:
        def lower(self, request, arguments):
            if kind in ("llvm", "unknown-arith"):
                name = "llvm.inline_asm" if kind == "llvm" else "arith.unqualified_effect"
                op = builtin.UnregisteredOp.with_name(name).create(operands=arguments, result_types=[builtin.i32])
            elif kind == "free-value":
                op = arith.AddiOp(arguments[0], Block(arg_types=[builtin.i32]).args[0])
            elif kind == "wrong-type":
                op = arith.ConstantOp.from_int_and_width(0, builtin.i64)
            else:
                op = arith.AddiOp(arguments[0], arguments[0])
                Block([op])
            return AssemblyExpansion([op], list(op.results), {})

    monkeypatch.setattr(source_assembly, "load_provider", lambda name: (InvalidProvider(), {}))
    path, abi = source(tmp_path)
    with pytest.raises(ConversionError, match="source assembly provider") as exc:
        convert(path, abi, tmp_path / "translation", assembly_provider="invalid")
    assert exc.value.opcode == "tt.elementwise_inline_asm" and exc.value.source_id


@pytest.mark.skipif(not PROVIDER, reason="optional source-semantics wheel required")
def test_purity_annotation_cannot_supply_an_effect_implementation(tmp_path):
    path, abi = source(tmp_path, pure=False)
    with pytest.raises(ConversionError, match="explicitly pure"):
        convert(path, abi, tmp_path / "translation", assembly_provider="ptx-integer-v1")


@pytest.mark.skipif(not (PROVIDER and NATIVE and MERLIN), reason="optional provider and native/Merlin tools required")
def test_public_cli_emits_and_qualifies_the_actual_scalar_program(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    from t2m.kernel_consumer import verify_kernel_qualification

    path, abi = source(tmp_path)
    (tmp_path / "abi.json").write_text(json.dumps(abi.as_dict()))
    np.savez(tmp_path / "inputs.npz", in0=np.asarray([(1 << 31) - 1], np.int32))
    np.savez(tmp_path / "reference.npz", out0=np.asarray([-(1 << 31)], np.int32))
    assert main(["shared-tensor", str(path), "--abi", str(tmp_path / "abi.json"), "--out", str(tmp_path / "capture"),
        "--optimizer", OPT, "--memory-profile", "guarded-buffer", "--address-policy", "source-words",
        "--assembly-provider", "ptx-integer-v1", "--inputs", str(tmp_path / "inputs.npz"),
        "--reference", str(tmp_path / "reference.npz"), "--python", NATIVE]) == 0
    verify_bundle(tmp_path / "capture")
    assert main(["qualify-merlin", str(tmp_path / "capture"), "--out", str(tmp_path / "merlin"),
        "--merlin-root", os.environ["T2M_MERLIN_ROOT"], "--merlin-python", os.environ["T2M_MERLIN_PYTHON"],
        "--merlin-clang", os.environ["T2M_MERLIN_CLANG"], "--python", NATIVE]) == 0
    verify_kernel_qualification(tmp_path / "merlin")


@pytest.mark.skipif(not (PROVIDER and NATIVE and MERLIN), reason="optional provider and native/Merlin tools required")
def test_maintained_optional_selection_keeps_both_qualified_and_refused_assembly(tmp_path):
    from t2m.examples.product_corpus import prepare
    from t2m.product_corpus import qualify_product_corpus, verify_product_corpus

    manifest = prepare(tmp_path / "inputs", assembly_provider="ptx-integer-v1")
    report = qualify_product_corpus(manifest, tmp_path / "report", optimizer=OPT, python=NATIVE,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    assert report["selected_cases"] == 14 and report["complete_selection_measured"]
    cases = {case["id"]: case for case in report["cases"]}
    assert cases["source-assembly-qualified"]["merlin_host"]
    assert cases["source-assembly-qualified"]["merlin_reference_examples"] == 4
    assert {name for name, row in cases.items() if row["status"] == "refused"} == {"source-assembly", "packed-map"}
    assert report["ttir"]["tt.elementwise_inline_asm"]["merlin_cases"] == ["source-assembly-qualified"]
    verify_product_corpus(tmp_path / "report")
