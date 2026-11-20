"""Actual lexical choices supply local Boolean facts, never earlier writes."""

from xdsl.dialects import builtin

from t2m.shared_tensor import opcode, parse_shared
from t2m.tile_write_proof import analyze_writes


def source():
    return parse_shared('''builtin.module {
      func.func @scope(%ptr: tensor<8x!tt.ptr<i64>>, %data: tensor<8xi64>, %q: i1) {
        %gate = "tt.splat"(%q) : (i1) -> tensor<8xi1>
        "tt.store"(%ptr, %data, %gate) : (tensor<8x!tt.ptr<i64>>, tensor<8xi64>, tensor<8xi1>) -> ()
        %outside = "tt.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<8x!tt.ptr<i64>>) -> tensor<8xi64>
        scf.if %q {
          %inside = "tt.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<8x!tt.ptr<i64>>) -> tensor<8xi64>
          scf.yield
        } else { scf.yield }
        func.return
      }
    }''')


def test_selected_successor_proves_its_actual_guarded_prefix_read():
    module = source()
    store = next(op for op in module.walk() if opcode(op) == "tt.store")
    reads = [op for op in module.walk() if opcode(op) == "tt.load"]
    proof = analyze_writes([store, reads[1]], reads[1].results[0].type)
    assert proof is not None and proof.matches(None)
    assert not proof.matches(None, global_scope=True)


def test_later_edge_and_provenance_cannot_initialize_the_original_prefix_read():
    module = source()
    store = next(op for op in module.walk() if opcode(op) == "tt.store")
    read = next(op for op in module.walk() if opcode(op) == "tt.load")
    read.attributes["prov.initialized"] = builtin.StringAttr("claimed")
    assert analyze_writes([store, read], read.results[0].type) is None
