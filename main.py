import argparse
import os, glob
from get_summary import get_summary, get_summary_gpt4o, get_summary_claude35, extract_gold_summaries_from_xml, read_documents, extract_text_speaker_from_xml
from parse import parse_summaries
from align import align, extract_mentions_from_gold_tsv, get_entities_from_gold_tsv, replace_empty_strings
from serialize import add_summaries_to_xml, add_anno_to_tsv
from score import get_sal_tsv, get_sal_mentions, sal_coref_cluster, extract_first_mentions, calculate_scores
from ensemble import process_tsv_files
import json


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
    doc_sp_texts=extract_text_speaker_from_xml(args.data_folder + '/input/xml/'+ args.partition) #use 'test' for scoring
    print('len of doc sp texts:', len(doc_sp_texts[:args.max_docs]))

    # Get document names and texts
    doc_ids, doc_texts = read_documents(args.data_folder, args.partition)
    if args.max_docs is not None:
        doc_ids = doc_ids[:args.max_docs]
        doc_texts = doc_texts[:args.max_docs]
    print('doc_ids:',doc_ids)
    #print('doc_texts:',doc_texts)

    gold_sal_ents=get_sal_tsv(gold_tsv_folder)
    sal_mentions=get_sal_mentions(gold_tsv_folder)
    sc=sal_coref_cluster(sal_mentions)

    # Extract mentions from gold TSV folder
    all_entities_from_tsv =get_entities_from_gold_tsv(args.data_folder + '/input/tsv/'+ args.partition) #use 'test' for scoring
    print('len of all_entities_from_tsv:', len(all_entities_from_tsv[:args.max_docs]))
    # Get as many summaries as specified for each document
    if args.model_name=="gpt4o":
        summaries = get_summary_gpt4o(doc_texts, doc_ids, args.data_folder, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    elif args.model_name=="claude-3-5-sonnet-20241022":
        summaries = get_summary_claude35(doc_texts, doc_ids, args.data_folder, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    else:
        summaries = get_summary(doc_texts, doc_ids, args.data_folder, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    print('summaries:',summaries)
    gold_summaries=extract_gold_summaries_from_xml(args.data_folder + '/input/xml/'+ args.partition) #use 'test' for scoring

    # Get all mentions from each summary
    all_mentions = parse_summaries(list(summaries.values()))
    sum1_mentions = parse_summaries(list(gold_summaries.values()))

    # Detect which entities from the document are mentioned in each summary
    alignments = align(all_entities_from_tsv[:args.max_docs], list(summaries.values()), all_mentions, doc_sp_texts[:args.max_docs], n_summaries=1 , component=args.alignment_component, partition=args.partition)
    sum1_alignments = align(all_entities_from_tsv, list(gold_summaries.values()), sum1_mentions, doc_sp_texts, data_folder=folders_with_pred_tsv, n_summaries=1 , component=args.alignment_component, partition=args.partition)

    if args.alignment_component=="LLM": #Fix the index error in LLM method
        alignments=replace_empty_strings(alignments)    
    # Debug
    # print('gold summaries:',gold_summaries)
    # print('All mentions:',all_mentions, '\n','len of All mentions:', len(all_mentions[0])) 
    # print('Alignments:',alignments, '\n', 'len of sum1 alignments:', len(alignments[0]))
    print('First doc Alignments:',alignments[0]) 
    #print('sc:',sc, '\n', 'len of sc:', len(sc)) 

    if args.alignment_component in ['coref_system', 'LLM_hf', 'LLM']:
        pred=extract_first_mentions(sc, alignments[0])
    else: 
        pred=extract_first_mentions(sc, alignments)
    pred = [[item for item in inner_list if item != []] for inner_list in pred] #remove unnecessary empty lists
    print('pred:',pred, '\n', 'len of pred:', len(pred)) #1
    # Create alignment data
    json.dump(pred, open(os.path.join(f"./data/output/alignment/{args.alignment_component}1", f"pred_{args.model_name.split('/')[-1]}_{args.partition}.json"), "w"), indent=4) if os.makedirs(f"./data/output/alignment/{args.alignment_component}1", exist_ok=True) is None else None
    #print('gold_sal_ents:',gold_sal_ents, '\n', 'len of gold:', len(gold_sal_ents)) 
    # Only calculate scores on test set
    # scores = calculate_scores(pred, gold_sal_ents) 
    # macro_precision = scores['macro_precision']
    # macro_recall = scores['macro_recall']
    # macro_f1 = scores['macro_f1']
    # micro_precision = scores['micro_precision']
    # micro_recall = scores['micro_recall']
    # micro_f1 = scores['micro_f1']
    # per_document_scores = scores['per_document_scores']
    # print('per_document_scores:',per_document_scores)
    # print("Macro Precision:", macro_precision)
    # print("Macro Recall:", macro_recall)
    # print("Macro F1 Score:", macro_f1)
    # print("Micro Precision:", micro_precision)
    # print("Micro Recall:", micro_recall)
    # print("Micro F1 Score:", micro_f1)
    
    results = process_tsv_files(args.partition)
    
    # Enrich annotations in XML with summaries and in TSV with salience information
    add_summaries_to_xml(args.data_folder, summaries)
    add_anno_to_tsv(data_folder=args.data_folder, model_predictions= results, partition=args.partition, max_docs=args.max_docs)


if __name__ == "__main__":
    main()
