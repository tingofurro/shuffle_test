# Re-Thinking the Shuffle Test

Codebase, data and models for the [Re-Thinking the Shuffle Test](https://people.eecs.berkeley.edu/~phillab/pdfs/ACL2021_Coherence_Shuffle.pdf) paper at ACL2021.

<p align="center">
  <img width="450" height="363" src="https://tingofurro.github.io/images/acl2021_coherence.png">
</p>

## Datasets

We perform Shuffle tests in three domains: news (Wall Street Journal standard dataset), legal (based on the Billsum dataset), and stories (based on the Reddit TIFU dataset). Data loaders are provided in the `utils_dataset.py` file [(link)](https://github.com/tingofurro/shuffle_test/blob/main/utils_dataset.py).

## Scripts

We provide the scripts to reproduce experimental results:
- `run_supervised.py`: is script to run the supervised `GPT2-large` model we finetuned on the binary classification task of "is shuffle".
- `run_gpt2_zeroshot.py`: is the script to run the zero-shot NLG models of the GPT2 type. It can be used to run various sizes of GPT2 architectures (base, medium, large).
- `run_bidir_zeroshot.py`: is the script to run NLU models of the BERT type. It can be used to run the `bert-base-uncased` and `roberta-base` experiments from the paper.

## Release

In the [release](https://github.com/tingofurro/shuffle_test/releases/tag/0.1), we provide the `roberta-large` checkpoint of the model we supervised to perform the shuffle test. It can be used in conjunction with the `run_supervised.py` script.

## Cite the work

If you make use of the code, models, or algorithm, please cite our paper:
```
@inproceedings{laban2021shuffle,
  title={Can Transformer Models Measure Coherence In Text? Re-Thinking the Shuffle Test},
  author={Laban, Philippe and Dai, Luke and Bandarkar, Lucas and Hearst, Marti A}
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  volume={1},
  year={2021}
}
```

## Contributing

If you'd like to contribute, or have questions or suggestions, you can contact us at phillab@berkeley.edu.
All contributions welcome!

