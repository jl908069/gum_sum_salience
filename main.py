import argparse
import os
from get_summary import get_summary, read_documents_from_excel
from parse import parse_summaries
from align import align, extract_mentions_from_gold_tsv
from serialize import add_summaries_to_xml, add_anno_to_tsv
#from generate_conll import generate_conll_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to gumsum.xlsx")
    parser.add_argument("output_tsv", type=str, help="Path to the output TSV file")
    parser.add_argument("output_xml", type=str, help="Path to the output XML file")
    #parser.add_argument("--format", type=str, choices=["tsv", "conllu", "xml"], default="tsv", help="Input file format")
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate")
    parser.add_argument("--alignment_component", type=str, choices=["LLM", "string_match", "coref_system"], default="LLM", help="Component to use for alignment")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="Huggingface model name to use for summarization")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder containing TSV, conllu or xml files")


    args = parser.parse_args()
    # Input data folder
    xml_folder = args.data_folder + '/xml/train' #default to train
    tsv_folder = args.data_folder + '/tsv/train' #default to train
    pred_tsv_folder =args.data_folder + '/pred_tsv'

    folders_with_pred_tsv = [os.path.join(pred_tsv_folder, f'tsv_pred_train{i}') for i in range(1, args.n_summaries + 1) if glob.glob(os.path.join(pred_tsv_folder, f'tsv_pred_train{i}', '*.tsv'))] #default to train
    
    doc_ids, doc_texts = read_documents_from_excel(args.input_file)
    
    # Extract mentions from TSV folder
    all_mentions_from_tsv = extract_mentions_from_gold_tsv(args.tsv_folder)
    
    summaries = get_summary(doc_texts, doc_ids, model_name=args.model_name, n=args.n_summaries)
    
    all_mentions = parse_summaries(list(summaries.values()))
    
    alignments= align(all_mentions_from_tsv, list(summaries.values()), all_mentions, data_folder=folders_with_pred_tsv, n_summaries=args.n_summaries , component=args.alignment_component)

    add_summaries_to_xml(xml_folder, summaries, args.output_xml)
    add_anno_to_tsv(tsv_folder, args.output_tsv, alignments)
    
if __name__ == "__main__":
    main()
