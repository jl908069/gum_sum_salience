# gum_sum_salience

### `main.py`
  - Setup argument parsing for the script
  - Loading documents (tsv, conllu, xml), generating summaries, parsing summaries, aligning mentions, and serializing results

### `get_summary.py`
  - Define a function get_summary(doc_text, n=4) that interacts with the Huggingface API to generate n summaries

### `parse.py`
  - Define a function parse(summary_text) that returns a list of noun phrase (NP) strings corresponding to all nominal mention strings (excluding pronouns)

### `align.py`
  - Define a function align(doc_conllu, summary_text, mention_text) that aligns mentions from the summary with those in the document
  - Use one of three components (LLM, string_match, coref_system) to perform the alignment

### `serialize.py`
  - Define a function serialize(tsv, xml, alignments) that takes the alignments and produces:
      - A TSV file with new annotations for salience
      - An XML file with new summaries embedded in the <text> element
