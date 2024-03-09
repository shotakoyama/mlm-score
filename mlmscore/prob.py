from tqdm import tqdm
import torch
from .mask import mask_index_dict
from .index import batched_indices


def make_mask_logprob_dict(seqs, mask_tok, pad_tok, max_tokens, model):
    mask_id_dict = mask_index_dict(seqs, mask_tok)
    masked_seqs = [list(x) for x in mask_id_dict.keys()]
    lengths = [len(x) for x in masked_seqs]
    batches = batched_indices(lengths, max_tokens)
    mask_logprob_dict = {}

    for batch in tqdm(batches, bar_format = '{l_bar}{r_bar}'):
        ten, msk = make_batch(batch, lengths, masked_seqs, pad_tok)
        ten = torch.tensor(ten).cuda()
        msk = torch.tensor(msk).cuda()
        with torch.no_grad():
            out = model(ten, attention_mask = msk)[0]
        for logit, index in zip(out, batch):
            tupled_seq = tuple(masked_seqs[index])
            mask_id = mask_id_dict[tupled_seq]
            log_dist = torch.log_softmax(logit[mask_id], dim = -1)
            mask_logprob_dict[tupled_seq] = log_dist.cpu().numpy()
    return mask_logprob_dict


def make_batch(indices, lengths, seqs, pad_tok):
    max_len = lengths[indices[0]]
    ten = [seqs[i] + [pad_tok] * (max_len - lengths[i]) for i in indices]
    msk = [[1] * len(seqs[i]) + [0] * (max_len - len(seqs[i])) for i in indices]
    return ten, msk

