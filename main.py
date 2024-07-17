import argparse
import os
from get_summary import get_summary, read_documents_from_excel
from parse import parse_summaries
from align import align, extract_mentions_from_gold_tsv
from serialize import serialize
#from generate_conll import generate_conll_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to gumsum.xlsx")
    parser.add_argument("output_tsv", type=str, help="Path to the output TSV file")
    parser.add_argument("output_xml", type=str, help="Path to the output XML file")
    parser.add_argument("--format", type=str, choices=["tsv", "conllu", "xml"], default="tsv", help="Input file format")
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate")
    parser.add_argument("--alignment_component", type=str, choices=["LLM", "string_match", "coref_system"], default="LLM", help="Component to use for alignment")
    parser.add_argument("--model_name", type=str, default="flan-t5-xl", help="Huggingface model name to use for summarization")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the folder containing TSV, conllu or xml files")


    args = parser.parse_args()
    
    doc_ids, doc_texts = read_documents_from_excel(args.input_file)
    
    # Extract mentions from TSV folder
    all_mentions_from_tsv = extract_mentions_from_gold_tsv(args.tsv_folder)
    
    summaries = get_summary(doc_texts, model_name=args.model_name, n=args.n_summaries)
    
    all_mentions = parse_summaries(summaries)
    
    alignments = []
    for i in range(len(all_mentions_from_tsv)):
        summary_alignments = []
        for summary_index in range(args.n_summaries):
            alignment = align(
                all_mentions_from_tsv[i],
                summary_text=summaries,
                mention_text=all_mentions,
                data_folder=args.data_folder,
                component=args.alignment_component
            )
            summary_alignments.append(alignment)
        alignments.append(summary_alignments)

    serialize(args.output_tsv, args.output_xml, alignments, summaries)
    
if __name__ == "__main__":
    main()
