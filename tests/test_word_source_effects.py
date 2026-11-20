"""Actual typed primitives cannot inherit permissions from a byte-only stage."""

from copy import deepcopy
import json
import os

import pytest

from t2m.word_source_effects import validate_source_effects

ATOMIC = "private-invocation-typed-atomics-v2"
VOLATILE = "private-invocation-volatile-loads-v1"


def fixture(*, atomic=True, volatile=True, dtype="i64"):
    floating = dtype in ("f32", "f64")
    bits = int(dtype[1:])
    value = "1.0" if floating else "1"
    body = f'%x="arith.constant"() <{{value={value}:{dtype}}}>:()->{dtype}'
    rows = []
    if atomic:
        body += f'''%old="tt.atomic_rmw"(%p,%x) <{{atomic_rmw_op={5 if floating else 4}:i32,sem=4:i32,scope=1:i32}}>
                    :(!tt.ptr<{dtype}>,{dtype})->{dtype} loc("ttir.op.atomic")'''
        info = {"operation": "fadd" if floating else "add", "standard_kind": 0 if floating else 1, "word_bits": bits,
            "source_order": "acq_rel", "source_scope": "gpu", "logical_scope": "all-logical-programs-in-private-invocation",
            "standard_order": "acq_rel", "external_participants": "excluded-by-private-buffer-ABI"}
        if floating:
            info.update(value_dtype=dtype, numeric_contract="ieee-f32-rne-add" if dtype == "f32" else "ieee-f64-rne-add-quiet-nans")
        rows.append({"id": "ttir.op.atomic", "opcode": "tt.atomic_rmw", "atomic": info})
    body += f'''%read="tt.load"(%p) <{{isVolatile={str(volatile).lower()}}}>
                :(!tt.ptr<{dtype}>)->{dtype} loc("ttir.op.read")'''
    row = {"id": "ttir.op.read", "opcode": "tt.load"}
    if volatile:
        row["volatile_read"] = {"schema": "t2m.volatile-read.v1", "source_id": "ttir.op.read", "result_type": dtype,
            "scope": VOLATILE, "observation": "one-actual-volatile-read-per-active-logical-lane",
            "external_participants": "excluded-by-private-buffer-ABI", "physical_mapping": "undecided"}
    rows.append(row)
    source = f'''"builtin.module"() ({{
      "tt.func"() <{{sym_name="f",function_type=(!tt.ptr<{dtype}>)->()}}> ({{^bb0(%p:!tt.ptr<{dtype}>):
        {body}
        "tt.return"():()->()
      }}):()->()
    }}):()->()'''
    memory = {"effect_profile": ATOMIC if atomic else VOLATILE if volatile else "ordinary"}
    if atomic and volatile:
        memory["volatile_profile"] = VOLATILE
    if dtype == "f32" and atomic:
        memory["f32_atomic_add"] = "ieee"
    stage = {"schema": "t2m.source-word-stages.v4" if atomic and volatile else
             "t2m.source-word-stages.v3" if volatile else "t2m.source-word-stages.v2"}
    return source, stage, memory, rows


@pytest.mark.parametrize("atomic,volatile", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_selected_primitives_remain_admitted(atomic, volatile):
    validate_source_effects(*fixture(atomic=atomic, volatile=volatile))


@pytest.mark.parametrize("schema", ["t2m.triton-shared-stage.v1", "t2m.source-word-stages.v2"])
def test_legacy_byte_custody_cannot_admit_an_actual_volatile_flag(schema):
    source, stage, memory, rows = fixture(atomic=False)
    stage["schema"] = schema
    with pytest.raises(ValueError, match="actual volatile"):
        validate_source_effects(source, stage, memory, rows)


@pytest.mark.parametrize("atomic,volatile", [(True, False), (False, True), (True, True)])
def test_an_ordinary_profile_cannot_hide_actual_effects_by_deleting_their_declarations(atomic, volatile):
    source, stage, _, rows = fixture(atomic=atomic, volatile=volatile)
    rows = [{"id": row["id"], "opcode": "tt.load"} for row in rows]
    with pytest.raises(ValueError, match="actual atomic|actual volatile"):
        validate_source_effects(source, stage, {"effect_profile": "ordinary"}, rows)


@pytest.mark.parametrize("damage", ["kind", "order", "scope", "missing", "flag", "lost-read", "dtype", "numeric"])
def test_actual_source_fields_and_declarations_must_agree(damage):
    source, stage, memory, rows = fixture(dtype="f32")
    if damage == "kind":
        source = source.replace("atomic_rmw_op=5:i32", "atomic_rmw_op=10:i32")
    elif damage == "order":
        source = source.replace("sem=4:i32", "sem=1:i32")
    elif damage == "scope":
        source = source.replace("scope=1:i32", "scope=2:i32")
    elif damage == "missing":
        rows[0].pop("atomic")
    elif damage == "flag":
        source = source.replace("isVolatile=true", "isVolatile=false")
    elif damage == "lost-read":
        rows[-1].pop("volatile_read")
    elif damage == "dtype":
        rows[-1]["volatile_read"]["result_type"] = "i32"
    else:
        memory["f32_atomic_add"] = "ftz"
    with pytest.raises(ValueError, match="atomic|volatile"):
        validate_source_effects(source, stage, memory, rows)


@pytest.mark.parametrize("schema", ["t2m.triton-shared-stage.v1", "t2m.source-word-stages.v2"])
def test_existing_ieee_f64_source_labels_remain_compatible_without_mutating_evidence(schema):
    source, stage, memory, rows = fixture(volatile=False, dtype="f64")
    stage["schema"] = schema
    rows[0]["atomic"]["numeric_contract"] = "ieee-f64-rne-add"
    before = deepcopy(rows)
    validate_source_effects(source, stage, memory, rows)
    assert rows == before


def test_unreachable_original_declarations_do_not_invent_actual_source_operations():
    source, stage, memory, rows = fixture()
    rows.append({**deepcopy(rows[0]), "id": "ttir.op.unreachable"})
    validate_source_effects(source, stage, memory, rows)


@pytest.mark.parametrize("name", ["ttg.barrier", "tt.elementwise_inline_asm", "llvm.fence"])
def test_unexpanded_physical_operations_cannot_hide_in_an_older_stage(name):
    source, stage, memory, rows = fixture(atomic=False, volatile=False)
    source = source.replace('"tt.return"()', f'"{name}"():()->() "tt.return"()')
    with pytest.raises(ValueError, match="unexpanded physical/effect"):
        validate_source_effects(source, stage, memory, rows)


def test_malformed_typed_source_refuses_admission():
    _, stage, memory, rows = fixture()
    with pytest.raises(ValueError, match="parseable typed"):
        validate_source_effects("this is not IR", stage, memory, rows)


TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


@pytest.fixture(scope="module", params=["legacy", "isolated"])
def admitted(tmp_path_factory, request):
    if not TOOLS:
        pytest.skip("existing source/native/Merlin tools required")
    from t2m.memory_contract import validate_memory_evidence
    from test_volatile_memory import consumer_check

    root = tmp_path_factory.mktemp("word-source-" + request.param)
    if request.param == "legacy":
        from test_shared_tensor import source_case
        from t2m.shared_tensor import convert_shared
        source, abi = source_case("matrix_add", root)
        result = convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
                                memory_profile="guarded-buffer", address_policy="source-words")
        expected = "t2m.triton-shared-stage.v1"
    else:
        from test_word_events import capture
        result = capture(root)
        expected = "t2m.source-word-stages.v2"
    stage = json.loads(result.auxiliary_artifacts["word-stage.json"])
    assert stage["schema"] == expected
    validate_memory_evidence(result.trace, result.auxiliary_artifacts)
    assert consumer_check(root, result)["accepted"]
    return root, result


def test_rehashed_actual_source_effects_refuse_both_complete_memory_gates(admitted):
    from test_volatile_memory import consumer_check
    from t2m.memory_contract import validate_memory_evidence
    from t2m.shared_tensor import _source_ids, opcode, parse_shared
    from t2m.word_events import print_ir, templates
    from t2m.word_evidence import record
    from xdsl.dialects import builtin

    root, result = admitted
    outcomes = []
    for effect in ("volatile", "atomic", "physical"):
        artifacts, trace = dict(result.auxiliary_artifacts), deepcopy(result.trace)
        module = parse_shared(artifacts["word-input.mlir"])
        load = next(op for op in module.walk() if opcode(op) == "tt.load")
        if effect == "volatile":
            load.properties["isVolatile"] = builtin.IntegerAttr(True, builtin.i1)
        else:
            if effect == "atomic":
                element = load.results[0].type.element_type
                floating = isinstance(element, builtin.AnyFloat)
                operation = builtin.UnregisteredOp.with_name("tt.atomic_rmw").create(
                    operands=[load.operands[0], load.results[0]], result_types=[load.results[0].type],
                    properties={"atomic_rmw_op": builtin.IntegerAttr(5 if floating else 4, builtin.i32),
                                "sem": builtin.IntegerAttr(4, builtin.i32), "scope": builtin.IntegerAttr(1, builtin.i32)})
            else:
                operation = builtin.UnregisteredOp.with_name("ttg.barrier").create(
                    properties={"addrSpace": builtin.IntegerAttr(31, builtin.i32)})
            load.parent_block().insert_op_after(operation, load)
        source = print_ir(module)
        artifacts["word-input.mlir"] = source
        stage = json.loads(artifacts["word-stage.json"])
        if stage["schema"] == "t2m.triton-shared-stage.v1":
            stage["artifacts"]["source.ttir"] = record(source)
        else:
            stage["input"] = record(source)
            if effect == "volatile":
                isolated, bindings, _ = templates(module, _source_ids)
                assert bindings == stage["events"]["bindings"]
                stage["events"]["input"] = print_ir(isolated)
                stage["events"]["native_stage"]["artifacts"]["source.ttir"] = record(stage["events"]["input"])
        artifacts["word-stage.json"] = json.dumps(stage)
        admission = json.loads(artifacts["memory-admission.json"])
        admission["word_input_sha256"] = record(source)["sha256"]
        artifacts["memory-admission.json"] = json.dumps(admission)
        trace["memory"]["word_stage_sha256"] = record(artifacts["word-stage.json"])["sha256"]
        trace["memory"]["admission_sha256"] = record(artifacts["memory-admission.json"])["sha256"]
        with pytest.raises(ValueError, match="actual volatile|actual atomic|unexpanded physical/effect") as producer:
            validate_memory_evidence(trace, artifacts)
        consumer = consumer_check(root, result, artifacts=artifacts, trace=trace)
        assert not consumer["accepted"] and any(word in consumer["error"] for word in ("actual", "unexpanded"))
        outcomes.append({"effect": effect, "producer": str(producer.value), "consumer": consumer["error"]})
    (root / "refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")
