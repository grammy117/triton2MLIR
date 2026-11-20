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
ATOMIC_SCHEMA = "t2m.synchronous-program-barrier.v2"
VOLATILE_SCHEMA = "t2m.synchronous-program-barrier.v3"
COMBINED_SCHEMA = "t2m.synchronous-program-barrier.v4"
SPACES = ((1, "local"), (2, "global_read"), (4, "global_write"),
          (8, "tensor_read"), (16, "tensor_write"))
CFG_PROFILE = "whole-program-scalar-cfg-v1"


def cfg_participation(function):
    """Check whole-program scalar edges before official CFG-to-SCF lifting.

    A scalar TTIR branch chooses one edge for the complete logical program.
    This grants no per-lane divergence, inter-program order or progress fact.
    Native source verification and the full effect/encoding audit still apply.
    """
    from .shared_tensor import opcode
    from xdsl.utils.exceptions import VerifyException

    if len(function.regions) != 1 or not function.regions[0].blocks:
        raise ValueError("program barrier CFG requires one nonempty source function region")
    blocks = set(function.regions[0].blocks)
    for block in blocks:
        terminal = block.last_op
        if terminal is None or any(op.successors for op in list(block.ops)[:-1]):
            raise ValueError("program barrier CFG requires complete terminal control edges")
        name = opcode(terminal)
        if name == "tt.return":
            if terminal.successors or terminal.results or terminal.regions:
                raise ValueError("program barrier CFG return has unexpected control fields")
            continue
        if name not in ("cf.br", "cf.cond_br", "cf.switch"):
            raise ValueError("program barrier CFG has an unsupported terminator")
        if not terminal.successors or any(target not in blocks for target in terminal.successors):
            raise ValueError("program barrier CFG edges must stay in the original program region")
        if name != "cf.br" and (not terminal.operands
                or not isinstance(terminal.operands[0].type, (builtin.IntegerType, builtin.IndexType))
                or name == "cf.cond_br" and terminal.operands[0].type != builtin.i1):
            raise ValueError("program barrier CFG conditions require actual scalar integer control")
        try:
            terminal.verify()
            if name == "cf.br":
                edges = [(terminal.successor, terminal.arguments)]
            elif name == "cf.cond_br":
                edges = [(terminal.then_block, terminal.then_arguments), (terminal.else_block, terminal.else_arguments)]
            else:
                edges = [(terminal.default_block, terminal.default_operands),
                         *zip(terminal.case_blocks, terminal.case_operand, strict=True)]
            if any([value.type for value in values] != [arg.type for arg in target.args] for target, values in edges):
                raise ValueError("program barrier CFG successor arguments differ from their original block types")
        except VerifyException as error:
            raise ValueError("program barrier CFG has invalid typed successor arguments") from error
    return CFG_PROFILE


def atomic_options(value):
    """Decode only the existing explicit private-invocation atomic profiles."""
    from .atomic_memory import EFFECT_PROFILE, TYPED_EFFECT_PROFILE

    if (not isinstance(value, dict) or set(value) != {"profile", "f32_atomic_add"}
            or value.get("profile") not in (EFFECT_PROFILE, TYPED_EFFECT_PROFILE)
            or value.get("f32_atomic_add") not in (None, "ieee", "ftz")
            or value.get("f32_atomic_add") is not None and value["profile"] != TYPED_EFFECT_PROFILE):
        raise ValueError("program barrier needs an explicit private atomic effect profile")
    return {"effect_profile": "typed-atomics" if value["profile"] == TYPED_EFFECT_PROFILE else "atomics",
            "f32_atomic_add": value["f32_atomic_add"]}


def atomic_selection(effect_profile, f32_atomic_add):
    from .atomic_memory import EFFECT_PROFILE, TYPED_EFFECT_PROFILE

    if effect_profile not in ("atomics", "typed-atomics", "atomic-volatile"):
        raise ValueError("atomic program phases require the selected private atomic effect profile")
    value = {"profile": TYPED_EFFECT_PROFILE if effect_profile in ("typed-atomics", "atomic-volatile") else EFFECT_PROFILE,
             "f32_atomic_add": f32_atomic_add}
    atomic_options(value)
    return value


def volatile_options(value):
    """Decode only the selected private volatile-read observation contract."""
    from .volatile_memory import EFFECT_PROFILE

    if value != {"profile": EFFECT_PROFILE}:
        raise ValueError("program barrier needs an explicit private volatile effect profile")
    return {"effect_profile": "volatile"}


def volatile_selection(effect_profile):
    from .volatile_memory import EFFECT_PROFILE

    if effect_profile not in ("volatile", "atomic-volatile"):
        raise ValueError("volatile program phases require the selected private volatile effect profile")
    return {"profile": EFFECT_PROFILE}


def contract(op, source_id, *, atomic_effects=None, volatile_effects=None):
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
    source_control = None
    while parent is not None and parent.name != "builtin.module":
        name = parent.op_name.data if isinstance(parent, builtin.UnregisteredOp) else parent.name
        if name not in {"tt.func", "scf.for", "scf.if", "scf.while"}:
            refuse("barrier participation requires whole-program structured control")
        if any(len(region.blocks) != 1 for region in parent.regions):
            if name != "tt.func":
                refuse("barrier participation needs single-block structured regions")
            try:
                source_control = cfg_participation(parent)
            except ValueError as error:
                refuse(str(error))
        parent = parent.parent_op()
    if atomic_effects is not None:
        atomic_options(atomic_effects)
    if volatile_effects is not None:
        volatile_options(volatile_effects)
        if atomic_effects is not None and atomic_options(atomic_effects)["effect_profile"] != "typed-atomics":
            refuse("combined program phases require the selected typed atomic component")
    schema = (COMBINED_SCHEMA if atomic_effects is not None and volatile_effects is not None else
              VOLATILE_SCHEMA if volatile_effects is not None else ATOMIC_SCHEMA if atomic_effects is not None else SCHEMA)
    return {"schema": schema,
            **({"source_control": source_control} if source_control is not None else {}),
            **({"atomic_effects": atomic_effects} if atomic_effects is not None else {}),
            **({"volatile_effects": volatile_effects} if volatile_effects is not None else {}),
            "source_id": source_id, "address_space_mask": space.value.data,
            "address_spaces": [name for bit, name in SPACES if space.value.data & bit],
            "participants": "complete-pre-layout-program; single-CTA",
            "ordering": "complete-synchronous-tile-operations-at-source-control-position",
            "external_participants": "not-admitted", "physical_mapping": "undecided"}


def expand(module, source_ids, *, effect_profile="ordinary", f32_atomic_add=None):
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

    atomics = [op for op in module.walk() if opcode(op) in {"tt.atomic_rmw", "tt.atomic_cas"}]
    volatile_reads = [op for op in module.walk() if opcode(op) == "tt.load"
                      and isinstance(op.properties.get("isVolatile"), builtin.IntegerAttr)
                      and op.properties["isVolatile"].value.data]
    atomic_effects = None
    if atomics:
        try:
            atomic_effects = atomic_selection(effect_profile, f32_atomic_add)
        except ValueError as error:
            refuse(str(error))
        from .atomic_memory import contract as atomic_contract
        for op in atomics:
            atomic_contract(op, source_ids, typed=effect_profile in ("typed-atomics", "atomic-volatile"), f32_atomic_add=f32_atomic_add)
    volatile_effects = None
    if volatile_reads:
        try:
            volatile_effects = volatile_selection(effect_profile)
        except ValueError as error:
            refuse(str(error))
        from .volatile_memory import contract as volatile_contract
        for op in volatile_reads:
            volatile_contract(op, next(iter(source_ids(op.location)), "ttir.volatile"))
    for op in module.walk():
        ctas = op.attributes.get("ttg.num-ctas", op.properties.get("ttg.num-ctas"))
        if ctas is not None and (not isinstance(ctas, builtin.IntegerAttr) or ctas.value.data != 1):
            refuse("multi-CTA barrier participation needs an explicit cluster contract")
        if opcode(op) in {"tt.elementwise_inline_asm", "tt.inline_asm"}:
            refuse("barrier composition with assembly effects needs a separate profile")
        arguments = [arg for region in op.regions for block in region.blocks for arg in block.args]
        for value in (*op.operands, *op.results, *arguments):
            if isinstance(value.type, builtin.TensorType) and not isinstance(value.type.encoding, builtin.NoneAttr):
                refuse("program barriers require pre-layout values without physical encodings")

    contracts = [(op, contract(op, next(iter(source_ids(op.location)), "ttir.barrier"),
                              atomic_effects=atomic_effects, volatile_effects=volatile_effects))
                 for op in barriers]
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
