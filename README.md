# gum_sum_salience

### Prepare datasets
- Download the GUM tsv and xml folders from [GUM](https://github.com/amir-zeldes/gum) and put them in `./data`. Place the `train`, `dev`, and `test` documents in their corresponding folders.
- To run the `coref_system` align method, put the prediction tsv files from [coref-mtl](https://github.com/yilunzhu/coref-mtl) under `./data/pred_tsv`. Name the folders this way: `tsv_pred_{train/dev/test}{summary_n}`. For example, `tsv_pred_train1` contains prediction tsv files from the `train` partition in summary 1 (the gold summary).
- The `./data` folder should look like this:
```
├── data
    └── gold_conll file                           # `v4_gold_conll` (e.g. `train.gum.english.v4_gold_conll`) file from `coref-mtl`
    └── input
        └── tsv                                       # gold tsv files from GUM
            └── train
            └── dev
            └── test
        └── xml                                       # gold xml files from GUM
            └── train
            └── dev
            └── test
    └── output
        └── xml                                   # output xml files with multiple generated summaries
        └── tsv                                   # output tsv files with graded salience information
        └── pred_tsv
            └── tsv_pred_{train/dev/test}{summary_n} # prediction tsv files from `coref-mtl`
        └── ...
```
#### `main.py`
  - Setup argument parsing for the script
  - Loading documents (xlsx, tsv, xml), generating summaries, parsing summaries, aligning mentions, and serializing results

#### `get_summary.py`
  - Define a function get_summary(doc_text, n=4) that interacts with the Huggingface API to generate n summaries

#### `parse.py`
  - Define a function parse(summary_text) that returns a list of noun phrase (NP) strings corresponding to all nominal mention strings (excluding pronouns)

#### `align.py`
  - Define a function align(doc_mentions, summary_text, mention_text) that aligns mentions from the summary with those in the document
  - Use one of three components (LLM, string_match, coref_system) to perform the alignment

#### `serialize.py`
  - Define a function serialize(tsv, xml, alignments) that takes the alignments and produces:
      - A TSV file with new annotations for salience
      - An XML file with new summaries embedded in the <text> element
   
#### `generate_conll.py`
  - Generates the merged conll files (document+summary) needed for running the [coref-mtl](https://github.com/yilunzhu/coref-mtl)
  - Download `v4_gold_conll` file (e.g. `train.gum.english.v4_gold_conll`) from the coref-mtl repo as well. Put it under `./data`
  - Follow the instructions there to generate the prediction tsv files needed for running `align.py`. Put the prediction files under `./data/pred_tsv`

#### `score.py`
  - Precision/Recall/F1 score of salient entities (not mentions) for each one of the alignment component approaches
