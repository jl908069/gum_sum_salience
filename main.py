import argparse
import os, glob
from get_summary import get_summary, read_documents, extract_gold_summaries_from_xml
from parse import parse_summaries
from align import align, extract_mentions_from_gold_tsv
from serialize import add_summaries_to_xml, add_anno_to_tsv
from score import get_sal_tsv, get_sal_mentions, sal_coref_cluster, extract_first_mentions, calculate_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate")
    parser.add_argument("--alignment_component", choices=["LLM", "LLM_hf", "string_match", "coref_system"], default="string_match", help="Component to use for alignment")
    parser.add_argument("--model_name", default="google/flan-t5-xl", help="Huggingface model name to use for summarization and/or alignment with component LLM_hf")
    parser.add_argument("--data_folder", type=str, default="data", help="Path to the data folder containing TSV, conllu or xml files")
    parser.add_argument("--max_docs", type=int, default=None, help="Maximum number of documents to processe (default: None = all; choose a small number to prototype)")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached summaries (default: False)")

    args = parser.parse_args()

    # Input data folder
    pred_tsv_folder =args.data_folder + '/output/pred_tsv'
    gold_tsv_folder = args.data_folder + '/input/tsv/test' # use 'test' for scoring

    folders_with_pred_tsv = [os.path.join(pred_tsv_folder, f'tsv_pred_test{i}') for i in range(1, args.n_summaries + 1) if glob.glob(os.path.join(pred_tsv_folder, f'tsv_pred_test{i}', '*.tsv'))] # default to test

    # Get document names and texts
    doc_ids, doc_texts = read_documents(args.data_folder)
    if args.max_docs is not None:
        doc_ids = doc_ids[:args.max_docs]
        doc_texts = doc_texts[:args.max_docs]

    gold_sal_ents=get_sal_tsv(doc_ids)
    sal_mentions=get_sal_mentions(doc_ids)
    sc=sal_coref_cluster(sal_mentions)

    # Extract mentions from gold TSV folder
    all_mentions_from_tsv = extract_mentions_from_gold_tsv(args.data_folder + '/input/tsv/test', docnames=doc_ids) # use 'test' for scoring

    # Get as many summaries as specified for each document
    summaries = get_summary(doc_texts, doc_ids, args.data_folder, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    gold_summaries=extract_gold_summaries_from_xml(args.data_folder + '/input/xml/test') # use 'test' for scoring

    # Get all mentions from each summary
    all_mentions = parse_summaries(list(summaries.values()))
    sum1_mentions = parse_summaries(list(gold_summaries.values()))

    # Detect which entities from the document are mentioned in each summary
    alignments = align(all_mentions_from_tsv, list(summaries.values()), all_mentions, data_folder=folders_with_pred_tsv, n_summaries=args.n_summaries , component=args.alignment_component)
    sum1_alignments = align(all_mentions_from_tsv, list(gold_summaries.values()), sum1_mentions, data_folder=folders_with_pred_tsv, n_summaries=1 , component=args.alignment_component)

    if args.alignment_component in ['coref_system', 'LLM_hf', 'LLM']: 
        pred=extract_first_mentions(sc, sum1_alignments[0]) #TODO
    else: # string_match
        pred=extract_first_mentions(sc, sum1_alignments)
    precision, recall, f1_score = calculate_scores(pred, gold_sal_ents)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

    # Add annotations in XML with summaries and in TSV with salience information
    add_summaries_to_xml(args.data_folder, summaries)
    add_anno_to_tsv(args.data_folder, alignments)


if __name__ == "__main__":
    main()
