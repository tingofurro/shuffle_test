from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils_dataset import load_shuffle_test_set, test_set_names
from utils_misc import diff_to_tag, printer, shuffle_doc
import torch, tqdm, argparse, logging
from collections import Counter

logging.disable(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--model_card", type=str, default='roberta-large', help="What folder contains the model configuration.")
parser.add_argument("--model_file", type=str, default="/home/phillab/models/coherence/cls_shuffle_roberta-large_f1_0.9750.bin", help="What model file to use (actual parameters).")
parser.add_argument("--dataset", type=str, choices=test_set_names, default="wsj", help="Can be `wsj`, `legal` or `stories`.")
parser.add_argument("--block_size", type=int, default=1, help="Atomic-size of sentence-blocks being shuffled. Regular shuffle-test is usually block size of 1 sentence.")


n_shuffles = 20
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_card)
coherence_model = AutoModelForSequenceClassification.from_pretrained(args.model_card).cuda()
coherence_model.half().eval()
print(coherence_model.load_state_dict(torch.load(args.model_file)))

dataset = load_shuffle_test_set(args.dataset)
sent_split = "line" if args.dataset == "wsj" else "nltk"

all_diffs = []
ite = tqdm.tqdm(dataset, dynamic_ncols=True)
for II, d in enumerate(ite):
    shuffled_docs = []
    for _ in range(n_shuffles):
        s_doc = shuffle_doc(d["text"], block_size=args.block_size, sent_split=sent_split)
        shuffled_docs.append(s_doc)

    batch_toks = tokenizer.batch_encode_plus(([d["text"]] + shuffled_docs), padding=True, truncation=True, max_length=512, return_tensors="pt")
    model_outs = coherence_model(batch_toks["input_ids"].cuda())
    logits = model_outs["logits"][:, 1]
    diffs = (logits[0] - logits[1:]).tolist()

    all_diffs += diffs
    good_bad = Counter([diff_to_tag(diff) for diff in all_diffs])
    ite.set_description(printer(good_bad))
