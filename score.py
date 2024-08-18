import argparse
import os
import glob


def get_sal_tsv(input_paths):
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

    # Check if input_paths is a directory or a list of file paths
    if isinstance(input_paths, str) and os.path.isdir(input_paths):
        filepaths = sorted(glob.glob(os.path.join(input_paths, "*.tsv")))
    elif isinstance(input_paths, list):
        # Prepend './data/input/tsv/' and append '.tsv' to each file name in the list
        filepaths = [os.path.join('./data/input/tsv', f"{filename}.tsv") for filename in input_paths]
    else:
        raise ValueError("input_paths must be a directory or a list of file paths")

    for filepath in filepaths:
        # Convert to absolute path to ensure it's correct
        filepath = os.path.abspath(filepath)
        
        file_result = []
        try:
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
        
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            continue  # Optionally, skip to the next file if an error occurs
        
        all_results.append(file_result)
    
    return all_results

def get_sal_mentions(input_paths):
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

    # Check if input_paths is a directory or a list of file names
    if isinstance(input_paths, str) and os.path.isdir(input_paths):
        filepaths = sorted(glob.glob(os.path.join(input_paths, "*.tsv")))
    elif isinstance(input_paths, list):
        # Prepend './data/input/tsv/' and append '.tsv' to each file name in the list
        filepaths = [os.path.join('./data/input/tsv', f"{filename}.tsv") for filename in input_paths]
    else:
        raise ValueError("input_paths must be a directory or a list of file paths")

    for filepath in filepaths:
        # Convert to absolute path to ensure it's correct
        filepath = os.path.abspath(filepath)
        
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
