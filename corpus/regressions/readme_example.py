def process(items):
    # Validate user items
    validate_user_items(items)
    for x in items:
        if x:
            if x.ok:
                if x.ready:
                    if x.score > 0:
                        return run(x)
    try:
        commit()
    except Exception:
        pass
