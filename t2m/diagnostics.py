class ConversionError(ValueError):
    """An unsupported or unproved source program, with its source operation ID."""

    def __init__(self, message, *, source_id=None, opcode=None):
        self.source_id = source_id
        self.opcode = opcode
        prefix = f"{source_id} ({opcode}): " if source_id else ""
        super().__init__(prefix + message)

