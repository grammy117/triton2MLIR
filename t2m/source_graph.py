"""Retain the exact typed preflight graph across protected native rewrites."""

from copy import deepcopy

from xdsl.dialects.builtin import StringAttr, UnregisteredOp


def inventory(module, records):
    """Use the same SSA enumeration as preflight, with explicit block edges.

    The source has already passed the native source-only stage. This is a
    structural inventory, not an independent source-equivalence certificate.
    """
    values, regions, blocks = {}, {}, {}
    for op in module.walk():
        for region in op.regions:
            regions[region] = f"source.region.{len(regions):04d}"
            for block in region.blocks:
                blocks[block] = f"source.block.{len(blocks):04d}"
                for value in block.args:
                    values[value] = f"ttir.value.{len(values):04d}"
        for value in op.results:
            values[value] = f"ttir.value.{len(values):04d}"
    ops = [op for op in module.walk() if op is not module]
    if len(ops) != len(records):
        raise ValueError(
            "source inventory no longer matches the preflight operation roster"
        )
    result = deepcopy(records)
    for op, record in zip(ops, result, strict=True):
        name = op.op_name.data if isinstance(op, UnregisteredOp) else op.name
        if (
            record["opcode"] != name
            or record["inputs"] != [values[v] for v in op.operands]
            or record["outputs"] != [values[v] for v in op.results]
            or record["input_types"] != [str(v.type) for v in op.operands]
            or record["output_types"] != [str(v.type) for v in op.results]
            or record["properties"]
            != {key: str(value) for key, value in op.properties.items()}
            or record["attributes"]
            != {
                key: str(value)
                for key, value in op.attributes.items()
                if key != "op_name"
            }
        ):
            raise ValueError("typed source inventory disagrees with preflight")
        record.update(
            block=blocks[op.parent_block()], regions=[regions[r] for r in op.regions]
        )
        intent = {
            key: value.data
            for key, value in op.attributes.items()
            if key in ("prov.family", "prov.op", "prov.region_id")
            and isinstance(value, StringAttr)
        }
        if intent:
            record["intent"] = intent
        contract = record.get("dot_numeric")
        if (
            contract
            and contract["input_precision_applies"]
            and contract["input_precision"] != "ieee"
        ):
            mode = contract["input_precision"]
            record["dot_precision"] = {
                "input_precision": mode,
                "max_imprecise_accumulations": contract["max_imprecise_accumulations"],
                "accumulator": "original-source-C"
                if mode == "tf32"
                else "source-C-added-after-official-composite",
                "physical_mapping": "undecided",
            }
            if mode.startswith("tf32"):
                record["dot_precision"]["logical_numeric_profile"] = (
                    "logical-tf32-e8m10-rtz-v1"
                )
    return {
        "operations": result,
        "blocks": [
            {
                "id": name,
                "parent_region": regions[block.parent],
                "arguments": [
                    {"id": values[arg], "type": str(arg.type)} for arg in block.args
                ],
            }
            for block, name in blocks.items()
        ],
        "attributes": "complete-typed-generic-source-fields; source-only-native-verification; no source-equivalence claim",
    }
