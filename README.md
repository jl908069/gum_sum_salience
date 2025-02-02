# gum_sum_salience

### Prepare datasets
- Download the GUM tsv and xml folders from [GUM](https://github.com/amir-zeldes/gum) and put them in `./data`. Place the `train`, `dev`, and `test` documents in their corresponding folders.
<!-- - To run the `coref_system` align method, put the prediction tsv files from [coref-mtl](https://github.com/yilunzhu/coref-mtl) under `./data/pred_tsv`. Name the folders this way: `tsv_pred_{train/dev/test}{summary_n}`. For example, `tsv_pred_train1` contains prediction tsv files from the `train` partition in summary 1 (i.e. the gold summary).-->
- The `./data` folder should look like this:
```
├── data
    └── input
        └── tsv                                   # gold tsv files from GUM (Note: use `_build/src/tsv/` for running `serialize.py` now , use `/coref/gum/tsv/` for running others)
            └── train
            └── dev
            └── test
        └── xml                                   # gold xml files from GUM
            └── train
            └── dev
            └── test
    └── output
        └── xml                                   # output xml files with multiple generated summaries
        └── tsv                                   # output tsv files with graded salience information
        └── alignment                             # alignment results from stanza, LLM, string_match
            └── stanza                            # json files with predicted salient entities using stanza
            └── LLM                               # json files with predicted salient entities using LLM (GPT4o)
            └── string_match                      # json files with predicted salient entities using string_match
        └── summaries                             # human or LLM generated summaries
            └── train                             # LLM generated summaries
                └── {model_name} folder
            └── dev                               # human crowdsourced summaries (h1~h5)
            └── test                              # human crowdsourced summaries (h1~h5)
        └── ensemble
            └── `graded_sal_meta_learner_dev.tsv` # training tsv file for the logistic regression model
            └── train                             # prediction tsv files obtained from `alignment` to run `ensemble.py`
            └── dev
            └── test 
```
#### `main.py`
  - Setup argument parsing for the script
  - Loading documents (tsv, xml), generating summaries, parsing summaries, aligning mentions, and serializing results

#### `get_summary.py`
  - Define a function get_summary(doc_text, n=4) that interacts with APIs (Huggingface, Anthropic, OpenAI) to generate n summaries

#### `parse.py`
  - Define a function parse(summary_text) that returns a list of noun phrase (NP) strings corresponding to all nominal mention strings (excluding pronouns) using `spacy`

#### `align.py`
  - Define a function align(doc_mentions, summary_text, mention_text) that aligns mentions from the summary with those in the document
  - Use one of these components (LLM, LLM_hf, string_match, stanza) to perform the alignment

#### `serialize.py`
  - Define a function serialize(tsv, xml, alignments) that takes the alignments and produces:
      - A TSV file with new annotations for salience
      - An XML file with new summaries embedded in the <text> element

#### `ensemble.py`
- Take alignments from {string_match, stanza, LLM}, train a logistic regression model for predicting salient entities, and write the annotations to tsv files
- Example:

    ```bash
    python3 ensemble.py \
        --data_folder ./data \
        --partition train \
        --alignment_components stanza LLM string_match \
        --model_names gold gpt4o claude-3-5-sonnet-20241022 meta-llama/Llama-3.2-3B-Instruct Qwen2.5-7B-Instruct
    ```

<!-- #### `generate_conll.py`
  - Generates the merged conll files (document+summary) needed for running the [coref-mtl](https://github.com/yilunzhu/coref-mtl) (Zhu et al., 2023)
  - Download `v4_gold_conll` file (e.g. `train.gum.english.v4_gold_conll`) from the coref-mtl repo as well. Put it under `./data`
  - Follow the instructions there to generate the prediction tsv files needed for running `align.py`. Put the prediction files under `./data/pred_tsv` -->

#### `score.py`
  - Micro/Macro Precision/Recall/F1 score of salient entities (not mentions) for each one of the alignment component approaches
  - Default to score 'test' set
