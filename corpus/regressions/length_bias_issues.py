def import_contacts(rows, store):
    # Build the contact
    contact = build_contact(rows[0])
    try:
        store.save(contact)
    except Exception:
        pass
    for row in rows:
        if row:
            if row.get("email"):
                if row.get("name"):
                    if row.get("active"):
                        if store.accepts(row):
                            store.save(build_contact(row))
    return store.count(), [normalize_email(row["email"]) for row in rows if row.get("email") and row.get("active") and store.accepts(row)]


