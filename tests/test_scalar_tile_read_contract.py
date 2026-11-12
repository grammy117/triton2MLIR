"""A scalar projection never changes the original lookup's control position."""

import os

import numpy as np
import pytest
from xdsl.dialects import builtin, tensor

from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.scalar_tile_reads import expand_scalar_tile_reads
from t2m.shared_tensor import _source_ids, opcode, parse_shared


def module_for(*, conditional=False, shared=False):
    lookup = """%indices="tt.splat"(%index):(i32)->tensor<1xi32> loc("ttir.op.1")
      %selected="tt.gather"(%table,%indices) {axis=0:i32}:(tensor<4xi64>,tensor<1xi32>)->tensor<1xi64> loc("ttir.op.2")"""
    return parse_shared(f"""module {{func.func @forward(%table:tensor<4xi64>,%index:i32,%choose:i1)->tensor<1xi64> {{
      {lookup if not conditional else ""}
      %value=scf.if %choose -> (i64) {{
        {lookup if conditional else ""}
        %scalar="tt.unsplat"(%selected):(tensor<1xi64>)->i64 loc("ttir.op.3")
        scf.yield %scalar:i64
      }} else {{
        %zero=arith.constant 0:i64
        scf.yield %zero:i64
      }}
      {('%extra="tt.unsplat"(%selected):(tensor<1xi64>)->i64 loc("ttir.op.4")' if shared else "")}
      %out=tensor.from_elements %{"extra" if shared else "value"}:tensor<1xi64>
      return %out:tensor<1xi64>
    }} }}""")


def test_shared_projection_reads_once_before_the_original_branch():
    module = module_for(shared=True)
    records = expand_scalar_tile_reads(module, _source_ids)
    module.verify()
    function = next(iter(module.ops))
    reads = [op for op in function.walk() if isinstance(op, tensor.ExtractOp)]
    assert len(reads) == 1
    assert reads[0].parent is function.body.block
    assert {"ttir.op.2", "ttir.op.3", "ttir.op.4"} <= set(
        _source_ids(reads[0].location)
    )
    assert len(records) == 2 and all(
        item["read_position"] == "original-gather" for item in records
    )
    assert not any(
        opcode(op) in ("tt.gather", "tt.splat", "tt.unsplat") for op in function.walk()
    )


def test_source_layout_hint_is_retained_without_choosing_physical_placement():
    module = module_for()
    gather = next(op for op in module.walk() if opcode(op) == "tt.gather")
    gather.attributes["efficient_layout"] = builtin.UnitAttr()
    reports = expand_scalar_tile_reads(module, _source_ids)
    module.verify()
    assert reports[0]["source_efficient_layout_hint"] is True
    assert reports[0]["physical_mapping"] == "undecided"
    extracted = next(op for op in module.walk() if isinstance(op, tensor.ExtractOp))
    assert isinstance(
        extracted.attributes["prov.source_efficient_layout"], builtin.UnitAttr
    )


def test_shared_index_splat_is_removed_only_after_both_original_gathers():
    module = module_for(shared=True)
    function = next(iter(module.ops))
    gather = next(op for op in function.walk() if opcode(op) == "tt.gather")
    projections = [op for op in function.walk() if opcode(op) == "tt.unsplat"]
    second = gather.clone()
    function.body.block.insert_op_before(second, projections[-1])
    projections[-1].operands = (second.results[0],)
    expand_scalar_tile_reads(module, _source_ids)
    module.verify()
    assert sum(isinstance(op, tensor.ExtractOp) for op in function.walk()) == 2
    assert not any(
        opcode(op) in ("tt.gather", "tt.splat", "tt.unsplat") for op in function.walk()
    )


def test_logical_pointer_projection_stays_with_the_pointer_adapter():
    module = parse_shared("""module {func.func @pointer(%x:tensor<1x!tt.ptr<i32>>)->!tt.ptr<i32> {
      %out="tt.unsplat"(%x):(tensor<1x!tt.ptr<i32>>)->!tt.ptr<i32>
      return %out:!tt.ptr<i32>
    }}""")
    before = text(module)
    assert expand_scalar_tile_reads(module, _source_ids) == []
    assert text(module) == before


def test_serialized_tags_cannot_recreate_generated_guard_admission():
    from t2m.scans import expand_scans

    module = parse_shared("""module {func.func @forward(%x:tensor<8xi64>,%table:tensor<4xi64>)->tensor<8xi64> {
      %out="tt.scan"(%x) <{axis=0:i32,reverse=false}> ({
      ^bb0(%a:i64,%b:i64):
        %zero=arith.constant 0:i32
        %index="tt.splat"(%zero):(i32)->tensor<1xi32>
        %read="tt.gather"(%table,%index) {axis=0:i32}:(tensor<4xi64>,tensor<1xi32>)->tensor<1xi64>
        %value="tt.unsplat"(%read):(tensor<1xi64>)->i64
        "tt.scan.return"(%value):(i64)->()
      }):(tensor<8xi64>)->tensor<8xi64>
      return %out:tensor<8xi64>
    }}""")
    guards = set()
    expand_scalar_tile_reads(module, _source_ids, generated_guards=guards)
    assert len(guards) == 1
    copied = parse_shared(text(module))
    with pytest.raises(ConversionError, match="pure scalar"):
        expand_scans(copied, _source_ids, generated_guards=guards)
    expand_scans(module, _source_ids, generated_guards=guards)
    module.verify()


@pytest.mark.skipif(
    not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native compiler required"
)
def test_singleton_splat_projects_the_actual_wide_scalar_word(tmp_path):
    from t2m.native import execute_native

    module = parse_shared("""module {func.func @forward(%x:i64)->tensor<1xi64> {
      %one="tt.splat"(%x):(i64)->tensor<1xi64>
      %value="tt.unsplat"(%one):(tensor<1xi64>)->i64
      %out=tensor.from_elements %value:tensor<1xi64>
      return %out:tensor<1xi64>
    }}""")
    expand_scalar_tile_reads(module, _source_ids)
    module.verify()
    value = np.asarray((1 << 62) + 3, np.int64)
    outputs, _ = execute_native(
        text(module),
        [value],
        tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"],
    )
    assert outputs[0].tobytes() == value.tobytes()


@pytest.mark.parametrize("kind", ["axis", "property", "index-width", "splat-property"])
def test_all_original_candidates_validate_before_any_projection_is_replaced(kind):
    module = module_for(shared=True)
    gather = next(op for op in module.walk() if opcode(op) == "tt.gather")
    if kind == "axis":
        gather.properties["axis"] = builtin.IntegerAttr(7, builtin.i32)
    elif kind == "property":
        gather.attributes["unknown_policy"] = builtin.UnitAttr()
    else:
        splat = gather.operands[1].owner
        if kind == "splat-property":
            splat.attributes["unknown_policy"] = builtin.UnitAttr()
        else:
            splat.results[0]._type = builtin.TensorType(builtin.i16, (1,))
    before = text(module)
    with pytest.raises(ConversionError):
        expand_scalar_tile_reads(module, _source_ids)
    assert text(module) == before


@pytest.mark.skipif(
    not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native compiler required"
)
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("choose", [False, True])
@pytest.mark.parametrize("annotations", [False, True])
def test_actual_native_bounds_follow_the_gather_even_when_projection_is_inactive(
    tmp_path, conditional, choose, annotations
):
    from t2m.native import observe_native

    module = module_for(conditional=conditional)
    expand_scalar_tile_reads(module, _source_ids)
    if not annotations:
        for operation in module.walk():
            for key in list(operation.attributes):
                if key.startswith("prov."):
                    del operation.attributes[key]
    module.verify()
    work = tmp_path / "native"
    outcome = observe_native(
        text(module),
        [np.arange(4, dtype=np.int64), np.asarray(99, np.int32), np.asarray(choose)],
        work,
        python=os.environ["T2M_NATIVE_PYTHON"],
    )
    assert (work / "lowered-llvm.mlir").is_file()
    if conditional and not choose:
        assert outcome["returncode"] == 0
        with np.load(work / "outputs.npz", allow_pickle=False) as outputs:
            assert outputs["out0"].tolist() == [0]
    else:
        assert outcome["returncode"] != 0 and not outcome["timed_out"]
        assert b"TTIR tile gather index outside source axis" in (
            (work / "stdout.bin").read_bytes() + (work / "stderr.bin").read_bytes()
        )
