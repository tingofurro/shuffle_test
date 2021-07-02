from datasets import load_dataset
import json

test_set_names = ["wsj", "legal", "stories"]

def load_shuffle_test_set(dataset_name):
    if dataset_name == "wsj":
        # This dataset is not available for free, as it is based on the test portion of the Penn Treebank
        # See "Extending the Entity Grid with Entity-Specific Features" Elsner et al. for more information
        # https://aclanthology.org/P11-2022.pdf
        # It must be obtained from the Linguistic Data Consortium (LDC)
        # Link: https://catalog.ldc.upenn.edu/LDC99T42
        with open("/home/phillab/data/coherence_test_set.json", "r") as f:
            return json.load(f)
    elif dataset_name == 'legal':
        return list(load_dataset('billsum')['test'])[:1000]
    elif dataset_name == 'stories':
        # taking last 1000 elements to create test set
        data = list(load_dataset('reddit_tifu', 'short')['train'])[-1000:]
        dataset = [{"text": d["documents"]} for d in data]
        return dataset
