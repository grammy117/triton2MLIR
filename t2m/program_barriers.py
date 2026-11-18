"""Reify pre-layout CTA rendezvous as complete logical memory phases.

Triton v3.7.1's TTG_BarrierOp orders selected synchronous accesses within one
CTA. Meta/shared 17f0c234 and archived Microsoft 050b93ae do not convert it.
SCF's execute_region supplies the supported sequential region boundary; the
ordinary tensor-version or standard-memory adapters supply actual dependencies.
This is not an implementation of inter-CTA, asynchronous or external effects.
"""

from xdsl.dialects import builtin, scf
from xdsl.ir import Block, Region

from .diagnostics import ConversionError

SCHEMA = "t2m.synchronous-program-barrier.v1"
SPACES = ((1, "local"), (2, "global_read"), (4, "global_write"),
          (8, "tensor_read"), (16, "tensor_write"))


def contract(op, source_id):
    """Check the native-verified, pinned source operation's complete fields."""
    def refuse(message):
        raise ConversionError(message, source_id=source_id, opcode="ttg.barrier")

    space = op.properties.get("addrSpace")
    if (op.operands or op.results or op.regions or op.successors
            or set(op.properties) != {"addrSpace"}
            or not isinstance(space, builtin.IntegerAttr) or space.type != builtin.i32
            or not 0 <= space.value.data <= 31
            or any(not key.startswith(("prov.", "op_name")) for key in op.attributes)):
        refuse("program barrier requires the pinned operand-free address-space mask")
    parent = op.parent_op()
    while parent is not None and parent.name != "builtin.module":
        name = parent.op_name.data if isinstance(parent, builtin.UnregisteredOp) else parent.name
        if name not in {"tt.func", "scf.for", "scf.if", "scf.while"}:
            refuse("barrier participation requires whole-program structured control")
        if any(len(region.blocks) != 1 for region in parent.regions):
            refuse("barrier participation in a source CFG needs a separate proof")
        parent = parent.parent_op()
    return {"schema": SCHEMA, "source_id": source_id, "address_space_mask": space.value.data,
            "address_spaces": [name for bit, name in SPACES if space.value.data & bit],
            "participants": "complete-pre-layout-program; single-CTA",
            "ordering": "complete-synchronous-tile-operations-at-source-control-position",
            "external_participants": "not-admitted", "physical_mapping": "undecided"}


def expand(module, source_ids):
    """Preserve each original phase and its live values in a standard region.

    Native source verification and complete preflight must run first. All
    non-barrier physical operations still refuse there. Existing memory gates
    prove disjoint programs or enforce private-state conflicts at execution.
    A barrier cannot create initialization or inter-program ownership credit.
    """
    from .shared_tensor import opcode

    barriers = [op for op in module.walk() if opcode(op) == "ttg.barrier"]
    if not barriers:
        return []
    first = barriers[0]

    def refuse(message):
        raise ConversionError(message, source_id=next(iter(source_ids(first.location)), None),
                              opcode="ttg.barrier")

    for op in module.walk():
        ctas = op.attributes.get("ttg.num-ctas", op.properties.get("ttg.num-ctas"))
        if ctas is not None and (not isinstance(ctas, builtin.IntegerAttr) or ctas.value.data != 1):
            refuse("multi-CTA barrier participation needs an explicit cluster contract")
        if opcode(op) in {"tt.atomic_rmw", "tt.atomic_cas", "tt.elementwise_inline_asm", "tt.inline_asm"}:
            refuse("barrier composition with atomic or assembly effects needs a separate profile")
        arguments = [arg for region in op.regions for block in region.blocks for arg in block.args]
        for value in (*op.operands, *op.results, *arguments):
            if isinstance(value.type, builtin.TensorType) and not isinstance(value.type.encoding, builtin.NoneAttr):
                refuse("program barriers require pre-layout values without physical encodings")

    contracts = [(op, contract(op, next(iter(source_ids(op.location)), "ttir.barrier"))) for op in barriers]
    reports, previous = [], {}
    # Snapshot before mutations: a phase can itself contain a structured region
    # whose nested source barrier still needs its own original-position phase.
    for barrier, record in contracts:
        ids = source_ids(barrier.location)
        block = barrier.parent_block()
        prefix, active = [], block not in previous
        for op in block.ops:
            if op is barrier:
                break
            if active:
                prefix.append(op)
            elif op is previous[block]:
                active = True
        inside = {nested for op in prefix for nested in op.walk()}
        live = [value for op in prefix for value in op.results
                if any(use.operation not in inside for use in value.uses)]
        body = Block()
        for op in prefix:
            block.detach_op(op)
            body.add_op(op)
        yielded = scf.YieldOp(*live)
        yielded.location = barrier.location
        body.add_op(yielded)
        phase = scf.ExecuteRegionOp([value.type for value in live], Region(body))
        phase.attributes["no_inline"] = builtin.UnitAttr()
        phase.attributes.update({"prov.family": builtin.StringAttr("synchronization"),
                                 "prov.op": builtin.StringAttr("complete_program_phase")})
        phase.location = barrier.location
        block.insert_op_before(phase, barrier)
        def external_use(use, inside=inside, yielded=yielded):
            return use.operation not in inside and use.operation is not yielded

        for old, new in zip(live, phase.results, strict=True):
            old.replace_uses_with_if(new, external_use)
        block.erase_op(barrier)
        previous[block] = phase
        reports.append({"kind": "synchronous-program-memory-phase", "source_ids": ids,
                        "contract": record,
                        "phase_source_ids": list(dict.fromkeys(s for op in prefix for nested in op.walk()
                                                               for s in source_ids(nested.location))),
                        "live_value_types": [str(value.type) for value in live],
                        "representation": "scf.execute_region; full tensor versions or synchronous standard memory",
                        "ownership": "unchanged source bounds/initialization/program-conflict gates"})
    return reports
