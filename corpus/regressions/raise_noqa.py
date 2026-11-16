def assign(chunk):
    if not isinstance(chunk, dict):
        msg = "The input must be a dict."
        raise ValueError(msg)  # noqa: TRY004
    return chunk
