from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch, tqdm, argparse, logging, numpy as np
from utils_dataset import load_shuffle_test_set, test_set_names
from utils_misc import diff_to_tag, printer, shuffle_doc
from collections import Counter

logging.disable(logging.WARNING)
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="wsj", choices=test_set_names, help="Can be `wsj`, `legal` or `stories` for now.")
parser.add_argument("--model_card", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large"], help="Which model card from HuggingFace to use.")
parser.add_argument("--block_size", type=int, default=1, help="Atomic-size of sentence-blocks being shuffled. Regular shuffle-test is usually block size of 1 sentence.")

args = parser.parse_args()
print("Using", args.model_card)
print("On", args.dataset, "data")

sent_split = "line" if args.dataset == "wsj" else "nltk"

# Load Data
dataset = load_shuffle_test_set(args.dataset)

all_diffs = []
ite = tqdm.tqdm(dataset)

n_shuffles = 20
tokenizer = GPT2Tokenizer.from_pretrained(args.model_card)
model = GPT2LMHeadModel.from_pretrained(args.model_card, return_dict=True).cuda().eval()
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

max_length = model.config.n_positions
stride=512

for II, d in enumerate(ite):
    doc_and_shuffled = [d["text"]] + [shuffle_doc(d["text"], block_size=args.block_size, sent_split=sent_split) for _ in range(n_shuffles)]

    # compute perplexity using sliding window
    perplexities = np.zeros(len(doc_and_shuffled))
    for k in range(len(doc_and_shuffled)):
        doc = doc_and_shuffled[k]

        inputs = tokenizer(doc, return_tensors='pt')
        lls = []
        for i in range(0, inputs['input_ids'].size(1), stride):
            # getting the coordinates of the window
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, inputs['input_ids'].size(1))
            target_len = end_loc - i

            input_ids = inputs['input_ids'][:, begin_loc:end_loc].cuda()
            target_ids = input_ids.clone().cuda()

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs.loss * target_len
            lls.append(log_likelihood)
        perplexities[k] = torch.exp(torch.stack(lls).sum() / end_loc)

    diffs = (perplexities[1:] - perplexities[0]).tolist() # inverted because the lower the perplexity, the better
    all_diffs += diffs

    good_bad = Counter([diff_to_tag(diff) for diff in all_diffs])
    ite.set_description(printer(good_bad))
