def normalize_email(raw_email):
    trimmed = raw_email.strip().lower()
    if "@" not in trimmed:
        raise ValueError("email address has no domain part")
    local_part, domain = trimmed.rsplit("@", 1)
    return f"{local_part}@{domain}"


def split_full_name(full_name):
    parts = full_name.strip().split()
    if not parts:
        raise ValueError("full name is empty")
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def build_contact(record):
    first_name, last_name = split_full_name(record["name"])
    email = normalize_email(record["email"])
    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "source": record.get("source", "import"),
    }


def summarize_contacts(contacts):
    domains = {}
    for contact in contacts:
        domain = contact["email"].rsplit("@", 1)[1]
        domains[domain] = domains.get(domain, 0) + 1
    return sorted(domains.items(), key=lambda pair: pair[1], reverse=True)


