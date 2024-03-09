def batched_indices(lengths, max_tokens: int):
    assert len(lengths) > 0
    indices = make_indices(lengths)
    batches = []
    batch = []
    acc = 0
    max_len = lengths[indices[0]]
    for index in indices:
        acc += 1
        if acc * max_len > max_tokens:
            batches.append(batch)
            batch = [index]
            acc = 1
            max_len = lengths[index]
        else:
            batch.append(index)
    if batch:
        batches.append(batch)
    return batches


def make_indices(lengths):
    xs = list(enumerate(lengths))
    xs.sort(key = lambda x: -x[1])
    xs = [i for i, _ in xs]
    return xs

