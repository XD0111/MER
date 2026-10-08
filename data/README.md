# Dataset

MER uses **EGC-MAVEN**, the processed event-graph dataset introduced by PLAF and based on MAVEN_ERE. Graphs contain sub-event, causality, temporal, and coreference relations.

## Source

- PLAF repository: <https://github.com/ChaoLiang-HUST/PLAF>
- PLAF paper: Chao Liang, Bang Wang, Chuanhong Zhan, Wei Xiang. *Multiplex Graph Prompt Learning and Attentive Fusion for Event Graph Completion*. Neural Networks, 199, 108730, 2026. DOI: [10.1016/j.neunet.2026.108730](https://doi.org/10.1016/j.neunet.2026.108730).

Obtain the processed splits from the PLAF release or its authors. The full dataset and the MAVEN_ERE-to-EGC-MAVEN construction pipeline are not included in this repository. No dataset Release attachment is provided here at present.

## Split sizes

| Split | Event graphs |
|---|---:|
| Train | 10,991 |
| Validation | 3,589 |
| Test | 3,601 |

The format examples in [samples/](samples/) contain the first event graph from each split. They illustrate the schema and are not the full train/validation/test splits.

## File placement

Place the processed split files at:

```text
mer/data/train.json
mer/data/valid.json
mer/data/test.json
```

Run the main model from `mer/`. Encoder baselines and API baselines accept `--train_data_path`, `--valid_data_path`, and `--test_data_path`, so the same files can be reused without duplicating the dataset. Llama data preparation instead accepts `--train_path`, `--valid_path`, and `--test_path`; see its [instructions](../baselines/llama_lora/README.md).

Full data files are excluded from Git. Distribute them separately under the source dataset's applicable terms.

## Format and preprocessing

Each event graph includes a document title, event nodes with mention/type/sentence/location fields, and relations grouped by type. Refer to [train_sample.json](samples/train_sample.json) and [processe_data.py](../mer/data/processe_data.py) for the actual input structure and preprocessing.

The main RoBERTa implementation preprocesses data when launched. Encoder variants may reuse `.pth` data caches. Remove the corresponding cache when changing the data, tokenizer, or preprocessing settings to avoid using stale inputs. Model weights and preprocessing caches are excluded from Git.
