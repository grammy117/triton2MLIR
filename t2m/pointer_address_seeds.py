"""Expose established outer pointer roots before integer-address region analysis.

Reuse pointer_differences' exact addptr/bitcast/select and lane-path recipe.
Original narrow increments execute before signed byte widening. Reanalysis must
derive bit facts from the new byte recipe, never retain zero-displacement facts
for a seed which previously included an offset in its actual pointer.
"""

from xdsl.dialects import arith, builtin
from xdsl.ir import SSAValue
from xdsl.utils.exceptions import ParseError

from .diagnostics import ConversionError
from .pointer_addresses import _like
from .pointer_differences import pointer_offsets


def rebase(module, source_ids, *, retained=()):
    from .shared_tensor import opcode

    reports, rewritten = [], {}
    published_nodes = 0
    for op in tuple(module.walk()):
        if op in retained or opcode(op) != "tt.ptr_to_int" or len(op.operands) != 1 or len(op.results) != 1:
            continue
        recipe, published = [], False

        def emit(created, recipe=recipe, published_nodes=published_nodes):
            if len(recipe) >= 32768 or published_nodes + len(recipe) >= 65536:
                created.drop_all_references()
                raise ValueError("pointer seed exceeds its bounded recipe budget")
            recipe.append(created)
            return created.results[0]

        try:
            offsets, consumed = pointer_offsets(emit, source_ids, shape_anchors=True)
            root, displacement, _ = offsets(op.operands[0])
            # A nonuniform shape path is a comparison key, not an actual SSA
            # root. A differently shaped root also needs separate lane transport;
            # keep the existing exact pointer pair in those cases.
            if (not isinstance(root, SSAValue) or root is op.operands[0]
                    or _like(root.type, builtin.i64) != op.results[0].type):
                continue
            if (op.regions or op.successors or any(
                    key not in ("op_name", "op_name__") and not key.startswith("prov.")
                    for key in (*op.properties, *op.attributes))):
                raise ValueError("pointer seed has unqualified source fields or effects")
            identities = sorted(set(source_ids(op.location)) | consumed)
            if len(identities) > 4096:
                raise ValueError("pointer seed exceeds its 4096-source budget")
            cast = builtin.UnregisteredOp.with_name("tt.ptr_to_int").create(
                operands=[root], result_types=[op.results[0].type], properties=dict(op.properties),
                attributes={key: value for key, value in op.attributes.items() if key not in ("op_name", "op_name__")})
            address = arith.AddiOp(emit(cast), displacement)
            emit(address)
            location = builtin.FusedLoc(builtin.ArrayAttr([
                op.location, *(builtin.NameLoc(builtin.StringAttr(identity), builtin.UnknownLoc())
                               for identity in identities)]), builtin.NoneAttr())
            for created in recipe:
                created.location = location
            op.parent.insert_ops_before(recipe, op)
            op.results[0].replace_all_uses_with(address.result)
            op.parent.erase_op(op)
            published = True
            published_nodes += len(recipe)
            rewritten[cast] = op
            reports.append({"kind": "integer-address-seed-rebase", "source_ids": identities,
                            "representation": "actual-same-shaped-pointer-root-plus-i64-byte-recipe",
                            "source_arithmetic": "original-width-before-signed-widening-and-per-addptr-byte-scale",
                            "bit_facts": "recomputed-from-executable-byte-recipe",
                            "physical_address": "unobserved", "public_abi": "unchanged"})
        except (ValueError, ParseError) as error:
            raise ConversionError(str(error), opcode="tt.ptr_to_int",
                                  source_id=next(iter(source_ids(op.location)), None)) from error
        finally:
            if not published:
                for created in reversed(recipe):
                    created.drop_all_references()
    return reports, rewritten
