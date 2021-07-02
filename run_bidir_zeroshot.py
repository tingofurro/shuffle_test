from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM

import torch, tqdm, argparse, logging, numpy as np
from utils_dataset import load_shuffle_test_set, test_set_names
from utils_misc import diff_to_tag, printer, shuffle_doc
from collections import Counter


logging.disable(logging.WARNING)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="wsj", choices=test_set_names, help="Can be `wsj`, `legal` or `stories` for now.")
parser.add_argument("--model_card", type=str, default="roberta-base", choices=["roberta-base", "bert-base-uncased"], help="Can be `bert-base-uncased`, `roberta-base`")
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()
print("Using", args.model_card)
print("On", args.dataset, "data")

dataset = load_shuffle_test_set(args.dataset)
sent_split = "line" if args.dataset == "wsj" else "nltk"

if args.model_card.startswith('roberta'):
    model = RobertaForMaskedLM.from_pretrained(args.model_card).cuda()
    tokenizer = RobertaTokenizer.from_pretrained(args.model_card)
else:
    model = BertForMaskedLM.from_pretrained(args.model_card).cuda()
    tokenizer = BertTokenizer.from_pretrained(args.model_card)

mask_id = tokenizer.mask_token_id
vocab_size = tokenizer.vocab_size
cross_ent = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

def create_all_masks(tokens):
    all_masks = []
    for i in range(len(tokens)):
        mask_i = tokens[:i] + [mask_id] + tokens[(i+1):]
        all_masks.append(mask_i)
    return all_masks

def compute_logprob(maskeds, unmaskeds, batch_size):
    N = len(maskeds)

    masked = torch.LongTensor(maskeds).cuda()
    full_unmasked = torch.LongTensor(unmaskeds).cuda()

    masked_binary = (masked == mask_id).long()
    unmasked_wminus1 = (1-masked_binary) * (-1) + masked_binary * (full_unmasked)

    loss = 0
    for i in range(0, N, batch_size):
        batch = masked[i:min(i+batch_size, N)]
        unmasked_wminus1_batch = unmasked_wminus1[i:i+batch_size]
        with torch.no_grad():
            model_outs = model(batch)
            loss_batch = cross_ent(model_outs["logits"].view(-1, vocab_size), unmasked_wminus1_batch.view(-1)).view(len(batch), -1)
            loss += torch.sum(loss_batch, dim=0)

    loss_per = loss / torch.sum(masked_binary.float(), dim=1)
    return loss_per

def compute_pseudologprob(text, stride=256, max_length=512):
    input_ids = tokenizer.encode(text)[1:-1] # len N
    all_masks = create_all_masks(input_ids) # len N
    N = len(input_ids)

    losses = []
    for i in range(0, N, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, N)
        input_ids_window = input_ids[begin_loc:end_loc]
        all_masks_window = np.array(all_masks)[begin_loc:end_loc, begin_loc:end_loc]

        loss_per = compute_logprob(all_masks_window, [input_ids_window] * len(all_masks_window), batch_size=32)
        losses.append(loss_per.mean().item())
    return np.mean(losses)


all_diffs = []
ite = tqdm.tqdm(dataset)
n_shuffles = 20

for II, d in enumerate(ite):
    doc_and_shuffled = [d["text"]] + [shuffle_doc(d["text"], block_size=args.block_size, sent_split=sent_split) for _ in range(n_shuffles)]

    losses = np.zeros(len(doc_and_shuffled))
    for i in range(len(doc_and_shuffled)):
        doc = doc_and_shuffled[i]
        score = compute_pseudologprob(doc)
        losses[i] = score

    diffs = (losses[1:] - losses[0]).tolist()
    all_diffs += diffs

    good_bad = Counter([diff_to_tag(diff) for diff in all_diffs])
    ite.set_description(printer(good_bad))
