import argparse
import os, glob
from get_summary import get_summary, extract_gold_summaries_from_xml, read_documents, extract_text_speaker_from_xml
from parse import parse_summaries
from align import align, extract_mentions_from_gold_tsv, get_entities_from_gold_tsv, replace_empty_strings
from serialize import add_summaries_to_xml, add_anno_to_tsv
from score import get_sal_tsv, get_sal_mentions, sal_coref_cluster, extract_first_mentions, calculate_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate")
    parser.add_argument("--alignment_component", choices=["LLM", "LLM_hf", "string_match", "coref_system", "stanza"], default="string_match", help="Component to use for alignment")
    parser.add_argument("--model_name", default="google/flan-t5-xl", help="Huggingface model name to use for summarization and/or alignment with component LLM_hf")
    parser.add_argument("--data_folder", type=str, default="data", help="Path to the data folder containing TSV, conllu or xml files")
    parser.add_argument("--max_docs", type=int, default=None, help="Maximum number of documents to processe (default: None = all; choose a small number to prototype)")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached summaries (default: False)")
    parser.add_argument("--partition", default="test", choices=["test", "dev", "train"], help="Data partition to use for alignment")

    args = parser.parse_args()

    # Input data folder
    pred_tsv_folder =args.data_folder + '/output/pred_tsv'
    gold_tsv_folder = args.data_folder + '/input/tsv/' + args.partition # use 'test' for scoring

    folders_with_pred_tsv = [os.path.join(pred_tsv_folder, f'tsv_pred_{args.partition}{i}') for i in range(1, args.n_summaries + 1) if glob.glob(os.path.join(pred_tsv_folder, f'tsv_pred_{args.partition}{i}', '*.tsv'))] 
    doc_sp_texts=extract_text_speaker_from_xml(args.data_folder + '/input/xml/'+ args.partition) # default to test

    # Get document names and texts
    doc_ids, doc_texts = read_documents(args.data_folder)
    if args.max_docs is not None:
        doc_ids = doc_ids[:args.max_docs]
        doc_texts = doc_texts[:args.max_docs]

    gold_sal_ents=get_sal_tsv(doc_ids)
    sal_mentions=get_sal_mentions(doc_ids)
    sc=sal_coref_cluster(sal_mentions)

    # Extract mentions from gold TSV folder
    all_entities_from_tsv =get_entities_from_gold_tsv(args.data_folder + '/input/tsv/'+ args.partition) #use 'test' for scoring
    
    # Get as many summaries as specified for each document
    summaries = get_summary(doc_texts, doc_ids, args.data_folder, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    gold_summaries=extract_gold_summaries_from_xml(args.data_folder + '/input/xml/'+ args.partition) # use 'test' for scoring

    # Get all mentions from each summary
    all_mentions = parse_summaries(list(summaries.values()))
    sum1_mentions = parse_summaries(list(gold_summaries.values()))

    # Detect which entities from the document are mentioned in each summary
    alignments = align(all_mentions_from_tsv, list(summaries.values()), all_mentions, data_folder=folders_with_pred_tsv, n_summaries=args.n_summaries , component=args.alignment_component, partition=args.partition)
    sum1_alignments = align(all_entities_from_tsv, list(gold_summaries.values()), sum1_mentions, doc_sp_texts, data_folder=folders_with_pred_tsv, n_summaries=1 , component=args.alignment_component, partition=args.partition)

    if args.alignment_component=="LLM": #Replace empty strings with "_"
        sum1_alignments=replace_empty_strings(sum1_alignments)

    if args.alignment_component in ['coref_system', 'LLM_hf', 'LLM']: 
        pred=extract_first_mentions(sc, sum1_alignments[0]) #TODO
    else: #string match, stanza
        pred=extract_first_mentions(sc, sum1_alignments)
    pred = [[item for item in inner_list if item != []] for inner_list in pred] #remove unnecessary empty lists
    
    scores = calculate_scores(pred, gold_sal_ents)
    macro_precision = scores['macro_precision']
    macro_recall = scores['macro_recall']
    macro_f1 = scores['macro_f1']
    micro_precision = scores['micro_precision']
    micro_recall = scores['micro_recall']
    micro_f1 = scores['micro_f1']
    per_document_scores = scores['per_document_scores']
    print("Micro Precision:", micro_precision)
    print("Micro Recall:", micro_recall)
    print("Micro F1 Score:", micro_f1)
    print("Macro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)
    print("Macro F1 Score:", macro_f1)
    

    # Add annotations in XML with summaries and in TSV with salience information
    add_summaries_to_xml(args.data_folder, summaries)
    add_anno_to_tsv(args.data_folder, alignments)


if __name__ == "__main__":
    main()
