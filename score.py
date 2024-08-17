import argparse
import os
import glob
from parse import parse_summaries
from align import align, extract_mentions_from_gold_tsv
from get_summary import get_summary, read_documents


def get_sal_tsv(directory): #gold sal ents from gold tsv
    all_results = []

    def extract_bracketed_number(s):
        start = s.find("[")
        end = s.find("]")
        if start != -1 and end != -1:
            return s[start+1:end]
        return None

    filepaths = sorted(glob.glob(os.path.join(directory, "*.tsv")))

    for filepath in filepaths:
        file_result = []
        with open(filepath, 'r') as file:
            word_dict = {}
            word_indices = {}
            
            for line in file:
                columns = line.strip().split('\t')
                if len(columns) < 6:
                    continue
                
                word_index = columns[0]
                word = columns[2]
                col5_values = columns[4].split('|')
                col6_values = columns[5].split('|')
                
                for col5, col6 in zip(col5_values, col6_values):
                    if col6.startswith('sal') and (col5.startswith('new') or col5.startswith('acc:')):
                        sal_number = extract_bracketed_number(col6)
                        if sal_number:
                            if sal_number not in word_dict:
                                word_dict[sal_number] = []
                                word_indices[sal_number] = []
                            word_dict[sal_number].append(word)
                            word_indices[sal_number].append(word_index)

            for key in word_dict:
                concatenated_words = " ".join(word_dict[key])
                concatenated_indices = ", ".join(word_indices[key])
                file_result.append((concatenated_words, concatenated_indices))
        
        all_results.append(file_result)
    
    return all_results

def get_sal_mentions(directory): #gold tsv dir
    all_results = []

    def extract_bracketed_number(s):
        start = s.find("[")
        end = s.find("]")
        if start != -1 and end != -1:
            return s[start+1:end]
        return None

    def remove_bracketed_number(s):
        parts = s.split('|')  # Split by "|" to handle multiple coref indices
        cleaned_parts = [part.split('[')[0] for part in parts]  # Remove everything after "["
        return ','.join(cleaned_parts)  # Rejoin the cleaned parts

    filepaths = sorted(glob.glob(os.path.join(directory, "*.tsv")))

    for filepath in filepaths:
        file_result = []
        with open(filepath, 'r') as file:
            word_dict = {}
            word_indices = {}
            coref_indices = {}
            
            for line in file:
                columns = line.strip().split('\t')
                if len(columns) < 7:
                    continue
                
                word_index = columns[0]
                word = columns[2]
                col5_values = columns[4].split('|')
                col6_values = columns[5].split('|')
                coref_index = columns[9]  
                
                for col5, col6 in zip(col5_values, col6_values):
                    if col6.startswith('sal'):
                        sal_number = extract_bracketed_number(col6)
                        if sal_number:
                            if sal_number not in word_dict:
                                word_dict[sal_number] = []
                                word_indices[sal_number] = []
                                coref_indices[sal_number] = []
                            word_dict[sal_number].append(word)
                            word_indices[sal_number].append(word_index)
                            coref_indices[sal_number].append(coref_index)

            for key in word_dict:
                concatenated_words = " ".join(word_dict[key])
                concatenated_indices = ", ".join(word_indices[key])
                # Remove bracketed numbers and concatenate
                filtered_corefs = [remove_bracketed_number(coref) for coref in coref_indices[key] if coref and coref != "_"]
                concatenated_corefs = ", ".join(filtered_corefs)
                file_result.append((concatenated_words, concatenated_indices, concatenated_corefs))
        
        all_results.append(file_result)
    
    return all_results

def sal_coref_cluster(sal_mentions): # a list of lists of tuples
    coref_clusters = []

    for file_result in sal_mentions:
        cluster = []
        used_indices = set()

        def find_coref_chain(start_tuple):
            chain = [start_tuple[0]]  # Start with the word span of the first tuple
            current_word_index = start_tuple[1].split(',')[0].strip()  # Only use the first word index
            current_coref_indices = [ci.strip() for ci in start_tuple[2].split(',')]

            while True:
                found = False
                for tup in file_result:
                    if tup in used_indices:
                        continue

                    words, word_indices, coref_indices = tup
                    first_word_index = word_indices.split(',')[0].strip()  # Only check the first index
                    coref_indices_list = [ci.strip() for ci in coref_indices.split(',')]

                    if first_word_index in current_coref_indices:
                        chain.append(words)
                        used_indices.add(tup)
                        current_word_index = first_word_index
                        current_coref_indices = coref_indices_list
                        found = True
                        break

                if not found:
                    break

            return tuple(chain)

        for tup in file_result:
            words, word_indices, coref_indices = tup

            # Handle singletons by only including the word span as a single string
            if coref_indices == "":
                cluster.append((words,))
                continue

            if tup not in used_indices:
                coref_chain = find_coref_chain(tup)
                if len(coref_chain) > 1:
                    cluster.append(coref_chain)
                used_indices.add(tup)

        coref_clusters.append(cluster)

    return coref_clusters

def extract_first_mentions(sc, st): # a list of lists
    results = []

    for sc_doc, st_doc in zip(sc, st):
        doc_results = []
        
        for st_list in st_doc:
            found = False

            for st_tuple in st_list:
                st_mention = st_tuple[0].strip().lower()  # Get the salient mention in lowercase

                for sc_tuple in sc_doc:
                    sc_mentions = [mention.strip().lower() for mention in sc_tuple]  # Normalize to lowercase

                    if st_mention in sc_mentions:
                        doc_results.append(sc_tuple[0])  # Append the first mention from sc_tuple in sc
                        found = True
                        break
                
                if found:
                    break
            
            if not found:
                doc_results.append(None)  # If no match is found, append None

        results.append(doc_results)
    
    return results

def calculate_scores(pred, gold):
    total_matches = 0
    total_pred_mentions = 0
    total_gold_mentions = 0

    for pred_doc, gold_doc in zip(pred, gold):
        pred_mentions = [p for p in pred_doc if p is not None]  # Filter out None values
        gold_mentions = [g[0] for g in gold_doc]  # Extract the first element (word span) from gold tuples

        total_pred_mentions += len(pred_mentions)
        total_gold_mentions += len(gold_mentions)

        # Count matches
        matches = sum(1 for pm in pred_mentions if pm in gold_mentions)
        total_matches += matches

    # Calculate precision, recall, and F1 score
    precision = total_matches / total_pred_mentions if total_pred_mentions > 0 else 0
    recall = total_matches / total_gold_mentions if total_gold_mentions > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="data", help="Path to data folder")
    parser.add_argument("--alignment_component", choices=["LLM", "LLM_hf", "string_match", "coref_system"], default="string_match", help="Component to use for alignment")
    parser.add_argument("--max_docs", type=int, default=None, help="Maximum number of documents to processe (default: None = all; choose a small number to prototype)")
    parser.add_argument("--model_name", default="google/flan-t5-xl", help="Huggingface model name to use for summarization")
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate per document")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached summaries (default: False)")

    args = parser.parse_args()

    # Input data folder
    pred_tsv_folder =args.data_folder + '/output/pred_tsv'

    folders_with_pred_tsv = [os.path.join(pred_tsv_folder, f'tsv_pred_train{i}') for i in range(1, args.n_summaries + 1) if glob.glob(os.path.join(pred_tsv_folder, f'tsv_pred_train{i}', '*.tsv'))] #default to train

    gold_sal_ents=get_sal_tsv(args.data_folder)
    sal_mentions=get_sal_mentions(args.data_folder)
    sc=sal_coref_cluster(sal_mentions)

    # Get document names and texts
    doc_ids, doc_texts = read_documents(args.data_folder)
    if args.max_docs is not None:
        doc_ids = doc_ids[:args.max_docs]
        doc_texts = doc_texts[:args.max_docs]
    
    # Extract mentions from gold TSV folder
    all_mentions_from_tsv = extract_mentions_from_gold_tsv(args.data_folder + '/input/tsv/', docnames=doc_ids)

    # Get as many summaries as specified for each document
    summaries = get_summary(doc_texts, doc_ids, args.data_folder, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)

    # Get all mentions from each summary
    all_mentions = parse_summaries(list(summaries.values()))

    # Detect which entities from the document are mentioned in each summary
    alignments = align(all_mentions_from_tsv, list(summaries.values()), all_mentions, data_folder=folders_with_pred_tsv, n_summaries=args.n_summaries , component=args.alignment_component)
    pred=extract_first_mentions(sc, alignments)
    precision, recall, f1_score = calculate_scores(pred, gold_sal_ents)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

