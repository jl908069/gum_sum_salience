import spacy
import csv
import os
import sys
import pandas as pd
import argparse
from get_summary import get_summary, read_documents_from_excel

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

nlp = spacy.load("en_core_web_sm")

def generate_tokenized_tsv_files(tsv_folder, summaries, output_file):
    tsv_columns = [
        "Document ID", "Part number", "Word number", "Word itself",
        "col_5", "col_6", "col_7", "col_8", "col_9", "col_10", 
        "col_11", "col_12", "col_13", "col_14", "col_15", "col_16", "col_17"
    ]
    
    data_to_write = []

    # List all TSV files in the folder
    tsv_files = sorted([f for f in os.listdir(tsv_folder) if f.endswith(".tsv")])
    
    doc_ids = [f.replace(".tsv", "") for f in tsv_files]
    
    for doc_id, tsv_file, summary in zip(doc_ids, tsv_files, summaries):
        modified_doc_id = doc_id.replace("GUM_", "").replace("_", "/")
        file_path = os.path.join(tsv_folder, tsv_file)
        doc = nlp(summary)
        part_number = 0  # Assuming single part documents
        for word_number, token in enumerate(doc):
            row = [
                modified_doc_id,
                part_number,
                word_number,
                token.text,
            ] + ["-"] * 13  # Fill remaining columns with "-"
            data_to_write.append(row)
    
    # Write to the single output TSV file
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow(tsv_columns)
        tsv_writer.writerows(data_to_write)

def read_tsv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        return [row for row in tsv_reader]

def write_tsv(file_path, rows):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        current_doc_id = None
        for row in rows:
            if len(row) > 1:
                doc_id = row[0]
                if doc_id != current_doc_id:
                    if current_doc_id is not None:
                        file.write("#end document\n\n")
                    current_doc_id = doc_id
                    file.write(f"#begin document ({doc_id}); part 000\n")
                tsv_writer = csv.writer(file, delimiter='\t')
                tsv_writer.writerow(row)
            else:
                file.write('\n')

def merge_conll_files(conll_file, tsv_file, output_file):
    # Read the files
    conll_rows = read_tsv(conll_file)
    tsv_rows = read_tsv(tsv_file)

    # Create a dictionary to hold rows by the first column value
    conll_dict = {}
    for row in conll_rows:
        if len(row) > 0:
            key = row[0]
            if key not in conll_dict:
                conll_dict[key] = []
            conll_dict[key].append(row)
        else:
            # Ensure sentences are separated by a new line
            if key in conll_dict:
                conll_dict[key].append([])

    # Append tsv rows to the corresponding keys in conll_dict
    for row in tsv_rows:
        if len(row) > 0:
            key = row[0]
            if key in conll_dict:
                conll_dict[key].append(row)
            else:
                conll_dict[key] = [row]

    # Flatten the dictionary into a list of rows
    merged_rows = []
    for key in sorted(conll_dict.keys()):
        merged_rows.extend(conll_dict[key])
        merged_rows.append([])  # Add a new line between sentences

    # Write the merged rows to the output file
    write_tsv(output_file, merged_rows)

def process_conll_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file, 'w', encoding='utf-8') as file:
        for line in lines:
            columns = line.strip().split('\t')
            # Replace columns 11 to 17 with '-'
            columns[10:] = ['-' if col != '-' else col for col in columns[10:]]
            file.write('\t'.join(columns) + '\n')

def fix_conll_file(input_file: str, output_file: str) -> None:
    """
    Fixes a CoNLL file by grouping lines with the same document ID within the same
    "#begin document" and "#end document" blocks and writes the result to a new file.

    :param input_file: Path to the input CoNLL file.
    :param output_file: Path to the output CoNLL file.
    """
    # Step 1: Read the file and collect lines for each document ID
    doc_lines = {}
    current_doc_id = None

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("#begin document"):
                current_doc_id = line.split()[2][1:-2]  # Extract the document ID
                #print(current_doc_id)
                if current_doc_id not in doc_lines:
                    doc_lines[current_doc_id] = []
            elif line.startswith("#end document"):
                current_doc_id = None
            elif current_doc_id:
                doc_lines[current_doc_id].append(line)
            else:
                # Handle the case of lines outside any document block
                if None not in doc_lines:
                    doc_lines[None] = []
                doc_lines[None].append(line)

    # Step 2: Write the sorted and grouped lines back to a new file
    with open(output_file, 'w', encoding='utf-8') as file:
        for doc_id, lines in doc_lines.items():
            if doc_id is not None:
                file.write(f"#begin document ({doc_id}); part 000\n")
            file.writelines(lines)
            if doc_id is not None:
                file.write("#end document\n\n")

    # Copy any lines that were outside document blocks at the end of the file
    if None in doc_lines:
        with open(output_file, 'a', encoding='utf-8') as file:
            file.writelines(doc_lines[None])

def process_tsv_to_conll(tsv_folder, summaries, conll_file, final_output_file):
    intermediate_tsv_file = "intermediate_tsv_output.tsv"
    intermediate_conll_file = "intermediate_conll_output.conll"
    processed_output_file="processed_conll_output.conll"
    
    # Generate tokenized TSV file
    generate_tokenized_tsv_files(tsv_folder, summaries, intermediate_tsv_file)
    
    # Merge the generated TSV file with the CoNLL file
    merge_conll_files(conll_file, intermediate_tsv_file, intermediate_conll_file)
    
    # Process the merged CoNLL file to produce the final output
    process_conll_file(intermediate_conll_file, processed_output_file)

    #Fix some erroneous lines
    fix_conll_file(processed_output_file, final_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to gumsum.xlsx")
    parser.add_argument("conll_file", type=str, default="./data/train.gum.english.v4_gold_conll", help="Path to the gold CoNLL file")
    parser.add_argument("tsv_folder", type=str, default= "./data/tsv/train", help="Path to the folder containing TSV files")
    parser.add_argument("final_output_folder", type=str, help="Path to the folder to save final output files")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="Huggingface model name to use for summarization")
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate per document")

    args = parser.parse_args()

    # Read documents from Excel file
    doc_ids, doc_texts = read_documents_from_excel(args.input_file)

    # Generate summaries
    summaries = get_summary(doc_texts, doc_ids, model_name=args.model_name, n=args.n_summaries)

    # Process each summary
    for i in range(args.n_summaries):
        current_summaries = [doc_summaries[i] for doc_summaries in summaries.values()]
        final_output_file = os.path.join(args.final_output_folder, f"train{i + 1}.gum.english.v4_eval_conll")
        process_tsv_to_conll(args.tsv_folder, current_summaries, args.conll_file, final_output_file)

#Example usage:
# python generate_conll.py 'gumsum.xlsx' './data/train.gum.english.v4_gold_conll' './data/tsv/train' './merged_conll_output' --model_name 'google/flan-t5-base' 
