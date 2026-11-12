"""Original nested effects cannot disappear under parent Pure or helper folding."""

import os

import pytest

from t2m.diagnostics import ConversionError
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT = os.environ.get("TRITON_SHARED_OPT")
pytestmark = pytest.mark.skipif(not OPT, reason="existing shared verifier required")


@pytest.mark.parametrize("kind", ["reduce", "scan"])
@pytest.mark.parametrize("count", [1, 8])
@pytest.mark.parametrize("helper", [False, True])
@pytest.mark.parametrize("effect", ["load", "print"])
def test_original_effectful_combiners_refuse_before_inlining_even_when_unused(
    tmp_path, kind, count, helper, effect
):
    type_ = f"tensor<{count}xi64>"
    returned_type = "i64" if kind == "reduce" else type_
    attributes = "axis=0:i32" + (",reverse=false" if kind == "scan" else "")
    instruction = (
        "%unused=tt.load %out:!tt.ptr<i64>"
        if effect == "load"
        else '"tt.print"(%a) <{prefix="observed",hex=false,isSigned=array<i32:1>}>:(i64)->()'
    )
    declaration = ""
    body = instruction + "\n%sum=arith.addi %a,%b:i64"
    if helper:
        declaration = f"""tt.func private @helper(%a:i64,%b:i64,%out:!tt.ptr<i64>)->i64 {{
          {body}
          tt.return %sum:i64
        }}"""
        body = "%sum=tt.call @helper(%a,%b,%out):(i64,i64,!tt.ptr<i64>)->i64"
    path = tmp_path / "source.ttir"
    path.write_text(f"""module {{
      tt.func public @kernel(%out:!tt.ptr<i64>) {{
        %input=arith.constant dense<1>:{type_}
        %result="tt.{kind}"(%input) <{{{attributes}}}> ({{
        ^bb0(%a:i64,%b:i64):
          {body}
          "tt.{kind}.return"(%sum):(i64)->()
        }}):({type_})->{returned_type}
        tt.return
      }}
      {declaration}
    }}""")
    work = tmp_path / "conversion"
    with pytest.raises(ConversionError, match="combiner effects") as error:
        convert_shared(
            path,
            TensorABI("kernel", (TensorBuffer("out", (1,), "write", "i64"),)),
            work,
            optimizer=OPT,
        )
    assert error.value.opcode == "tt." + effect and error.value.source_id
    assert not (work / "inline").exists()
