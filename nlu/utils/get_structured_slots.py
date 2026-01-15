def bio_to_structured_slots(word_slots_tuple):
    slots = {}
    current_key = None
    buffer = []

    for word, tag in word_slots_tuple:
        if tag == "O":
            if current_key:
                slots.setdefault(current_key, []).append(" ".join(buffer))
                buffer = []
                current_key = None
            continue

        prefix, label = tag.split("-", 1)
        if prefix == "B":
            if current_key:
                slots.setdefault(current_key, []).append(" ".join(buffer))
            current_key = label
            buffer = [word]
        elif prefix == "I" and current_key == label:
            buffer.append(word)

    if current_key:
        slots.setdefault(current_key, []).append(" ".join(buffer))

    # flatten singletons
    return {k: v[0] if len(v) == 1 else v for k, v in slots.items()}
