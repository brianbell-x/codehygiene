def run(kwargs, response, request):
    # Get the examples to use.
    examples = get_examples(**kwargs)
    # Add new cookies from the server.
    extract_cookies_to_jar(response.cookies, request)
    # TODO: fix this
    total = count_examples(examples)
    # ... existing code ...
    return examples, total


def load(path):
    try:
        return read(path)
    except Exception:  # noqa: BLE001
        pass
    return None
