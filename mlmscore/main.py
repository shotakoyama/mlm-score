import sys
import numpy as np
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .prob import make_mask_logprob_dict
from .mask import masked_sequence_iterator


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--arch', default='cl-tohoku/bert-base-japanese-v3')
    parser.add_argument('--max-tokens', type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer, pad_tok, mask_tok = prepare_tokenizer(args.arch)
    model = prepare_model(args.arch)

    texts = [text.strip() for text in sys.stdin]
    seqs = [tokenizer(x)['input_ids'] for x in texts]
    mask_logprob_dict = make_mask_logprob_dict(seqs, mask_tok, pad_tok, args.max_tokens, model)

    for text, seq in zip(texts, seqs):
        probs = calc_score(seq, mask_logprob_dict, mask_tok)
        score = np.mean(probs)
        print(f'{score}\t{text}')


def calc_score(seq, mask_logprob_dict, mask_tok):
    probs = []
    for where_to_mask, copied in masked_sequence_iterator(seq, mask_tok):
        probs.append(mask_logprob_dict[tuple(copied)][seq[where_to_mask]])
    return probs


def prepare_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    pad_tok = tokenizer.vocab[tokenizer.pad_token]
    mask_tok = tokenizer.vocab[tokenizer.mask_token]
    return tokenizer, pad_tok, mask_tok


def prepare_model(name):
    model = AutoModelForMaskedLM.from_pretrained(name)
    model.eval()
    model.cuda()
    return model


