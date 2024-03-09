def mask_index_dict(seqs, mask_tok):
    mask_id_dict = {}
    for seq in seqs:
        for mask_id, masked_seq in masked_sequence_iterator(seq, mask_tok):
            mask_id_dict[tuple(masked_seq)] = mask_id
    return mask_id_dict


def masked_sequence_iterator(seq, mask_tok):
    for where_to_mask in range(1, len(seq) - 1):
        copied = seq.copy()
        copied[where_to_mask] = mask_tok
        yield where_to_mask, copied

