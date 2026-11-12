"""Source implementation contracts expand before allocation wrapper inlining."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest
from test_descriptor_state import REQUIRED
from test_source_assembly import PROVIDER, source


@pytest.mark.skipif(not REQUIRED, reason="existing native/shared tools required")
def test_default_allocation_audits_unused_implementations_before_normalizing(
    tmp_path, monkeypatch
):
    from t2m import frontend
    from t2m.allocation_bindings import PointerArgument, convert_allocations
    from t2m.diagnostics import ConversionError

    path, abi = source(tmp_path)
    path.write_text(
        """module {
      tt.func public @assembly(%x:!tt.ptr<i32>,%out:!tt.ptr<i32>) {
        %v=tt.load %x:!tt.ptr<i32>
        tt.store %out,%v:!tt.ptr<i32>
        tt.return
      }
      tt.func private @unused(%value:i32)->i32 {
        %observation="tt.elementwise_inline_asm"(%value) <{asm_string="mov.u32 $0, %clock;",
          constraints="=r,r",pure=true,packed_element=1:i32}>:(i32)->i32
        tt.return %observation:i32
      }
    }"""
    )
    monkeypatch.setattr(
        frontend,
        "expand_tensor_pointers",
        lambda *_args, **_kwargs: pytest.fail(
            "unselected source semantics reached normalization"
        ),
    )
    with pytest.raises(ConversionError, match="separately qualified") as error:
        convert_allocations(
            path,
            replace(abi, entry="allocation_assembly"),
            (PointerArgument("x"), PointerArgument("out")),
            tmp_path / "binding",
            source_entry="assembly",
            optimizer=os.environ["TRITON_SHARED_OPT"],
            memory_profile="guarded-buffer",
            address_policy="source-words",
        )
    assert error.value.source_id and error.value.opcode == "tt.elementwise_inline_asm"
    assert not (tmp_path / "binding/intake").exists()


@pytest.mark.skipif(
    not (REQUIRED and PROVIDER),
    reason="existing native/Merlin and optional source provider required",
)
def test_selected_scalar_assembly_is_executable_before_allocation_binding(tmp_path):
    from test_mixed_effects import qualify

    from t2m.allocation_bindings import PointerArgument, convert_allocations

    original, abi = source(tmp_path)
    result = convert_allocations(
        original,
        replace(abi, entry="allocation_assembly"),
        (PointerArgument("x"), PointerArgument("out")),
        tmp_path / "binding",
        source_entry="assembly",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer",
        address_policy="source-words",
        assembly_provider="ptx-integer-v1",
    )
    report = json.loads(result.auxiliary_artifacts["allocation-protection.json"])
    assert len(report["assembly_conversions"]) == 1 and not report["fp8_conversions"]
    capsule = result.trace["source_argument_binding"]["source_capsule"]
    assert (
        capsule["schema"] == "t2m.source-allocation-capsule.v2"
        and capsule["normalized"] is False
    )
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "words"}]},
    }
    qualify(
        result,
        tmp_path,
        [np.asarray([-1], np.int32)],
        policy,
        [np.asarray([0], np.int32)],
    )
