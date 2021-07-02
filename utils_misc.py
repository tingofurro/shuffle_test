import nltk, random

def diff_to_tag(diff):
    if abs(diff) <= 0.0001:
        return "same"
    if diff > 0.0:
        return "good"
    return "bad"

def printer(good_bad):
    N = sum(good_bad.values())
    keys = ["good", "same", "bad"]
    return "["+"; ".join([("%s: %.1f" % (k, 100.0*good_bad[k]/N)) for k in keys])+"; N: %d]" % (N)

def doc2sents(doc, sent_split="line"):
    if sent_split == "line":
        sentences = doc.split("\n") # In this dataset, sentence tokenization has already been done.
    else:
        sentences = nltk.tokenize.sent_tokenize(doc.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences

def shuffle_doc(doc, block_size=1, sent_split="nltk"):
    sentences = doc2sents(doc, sent_split)
    N_sents = len(sentences)
    blocks = [sentences[i:(i+block_size)] for i in range(0, N_sents, block_size)]

    N = len(blocks)
    original_order = list(range(N))
    new_order = list(range(N))
    if N <= 1:
        return doc

    while new_order == original_order:
        random.shuffle(new_order)
    shuffled_doc = "\n".join([sentence for i in new_order for sentence in blocks[i]])
    return shuffled_doc
