import argparse
import os
import glob

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

def find_coref_chain(start_tuple, file_result, used_indices): 
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

def get_sal_tsv(input_paths):
# get salient entities (first mentions) from gold tsv     
    all_results = []

    # Check if input_paths is a directory or a list of file paths
    if isinstance(input_paths, str) and os.path.isdir(input_paths):
        filepaths = sorted(glob.glob(os.path.join(input_paths, "*.tsv")))
    elif isinstance(input_paths, list):
        # Prepend './data/input/tsv/' and append '.tsv' to each file name in the list
        filepaths = [os.path.join('./data/tsv', f"{filename}.tsv") for filename in input_paths]
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
                first_ent_type = {}  # New dict to store the first entity type for each sal_number
                
                for line in file:
                    columns = line.strip().split('\t')
                    if len(columns) < 7:
                        continue
                    
                    word_index = columns[0]
                    word = columns[2]
                    ent_type = columns[3].split('|')  # Extract ent_type values
                    col5_values = columns[4].split('|')
                    col6_values = columns[5].split('|')
                    coref_index = columns[9]  
                    
                    for i, (col5, col6, ent) in enumerate(zip(col5_values, col6_values, ent_type)):
                        if col6.startswith('sal') and (col5.startswith('new') or col5.startswith('acc:com') or col5.startswith('acc:inf')):
                            sal_number = extract_bracketed_number(col6)
                            if sal_number:
                                if sal_number not in word_dict:
                                    word_dict[sal_number] = []
                                    word_indices[sal_number] = []
                                    coref_indices[sal_number] = []
                                    first_ent_type[sal_number] = ent  # Store only the first entity type for the word span
                                word_dict[sal_number].append(word)
                                word_indices[sal_number].append(word_index)
                                coref_indices[sal_number].append(coref_index)

                for key in word_dict:
                    concatenated_words = " ".join(word_dict[key])
                    concatenated_indices = ", ".join(word_indices[key])
                    # Remove bracketed numbers and concatenate
                    filtered_corefs = [remove_bracketed_number(coref) for coref in coref_indices[key] if coref and coref != "_"]
                    concatenated_corefs = ", ".join(filtered_corefs)
                    # Use only the first entity type for the first token of the span
                    file_result.append((concatenated_words, concatenated_indices, concatenated_corefs, first_ent_type[key]))
        
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            continue  # Optionally, skip to the next file if an error occurs
        
        all_results.append(file_result)
    
    return all_results

def get_sal_mentions(input_paths):
    all_results = []

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

def sal_coref_cluster(sal_mentions):
    coref_clusters = []
    pronouns = {'he', 'she', 'it', 'they', 'we', 'i', 'you', 
                'him', 'her', 'them', 'us', 'me', 'it', 'there',
                'his', 'hers', 'its', 'their', 'our', 'my', 'your',
                'He', 'She', 'It', 'They', 'We', 'I', 'You', 
                'Him', 'Her', 'Them', 'Us', 'Me', 'It', 'There',
                'His', 'Hers', 'Its', 'Their', 'Our', 'My', 'Your'}

    for file_result in sal_mentions:
        cluster = []
        used_indices = set()

        for tup in file_result:
            words, word_indices, coref_indices = tup

            if coref_indices == "":
                if words not in pronouns:
                    cluster.append((words,))
                continue

            if tup not in used_indices:
                coref_chain = find_coref_chain(tup, file_result, used_indices)
                if len(coref_chain) > 1:
                    cluster.append(coref_chain)
                used_indices.add(tup)

        coref_clusters.append(cluster)

    return coref_clusters

def extract_first_mentions(sc, sum1_alignments):
    results = []

    for doc_index, (sc_doc, st_doc) in enumerate(zip(sc, sum1_alignments)):
        doc_results = []
        seen_mentions = set()  # Set to keep track of unique mentions

        for alignment_list in st_doc:
            # If alignment_list is empty, continue with an empty list in the results
            if not alignment_list:
                doc_results.append([])
                continue

            found_match = False
            for alignment in alignment_list:
                salient_mention = alignment[0].strip().lower()  # Get the salient mention in lowercase

                for sc_tuple in sc_doc:
                    sc_mentions = [mention.strip().lower() for mention in sc_tuple]  # Normalize mentions in sc to lowercase

                    # Check if the salient mention is a substring of any mention in sc
                    if any(salient_mention in mention for mention in sc_mentions) and sc_tuple[0] not in seen_mentions:
                        doc_results.append(sc_tuple[0])  # Append the first matching mention from sc
                        seen_mentions.add(sc_tuple[0])  # Mark this mention as seen
                        found_match = True
                        break  # Break after finding the match for this alignment

            if not found_match:
                doc_results.append([])  # Append empty list if no match is found

        results.append(doc_results)

    return results

def calculate_scores(pred, gold):
    # Initialize variables for micro-average calculation
    total_matches = 0
    total_pred_mentions = 0
    total_gold_mentions = 0

    # Initialize lists for macro-average and per-document scores
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # List to hold scores for each document individually
    per_document_scores = []

    # Loop over each document's predictions and gold standard
    for pred_doc, gold_doc in zip(pred, gold):
        pred_mentions = [p for p in pred_doc if p is not None]  # Filter out None values
        gold_mentions = [g[0] for g in gold_doc]  # Extract the first element (word span) from gold tuples

        # Update micro-average counters
        total_pred_mentions += len(pred_mentions)
        total_gold_mentions += len(gold_mentions)

        # Count matches between predicted mentions and gold mentions
        matches = sum(1 for pm in pred_mentions if pm in gold_mentions)
        total_matches += matches

        # Calculate precision, recall, and F1 for this document (macro calculation)
        precision = matches / len(pred_mentions) if len(pred_mentions) > 0 else 0
        recall = matches / len(gold_mentions) if len(gold_mentions) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Append the scores for this document to the lists
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)

        # Store per-document scores in a dictionary
        per_document_scores.append({
            'precision': precision,
            'recall': recall,
            'f1': f1_score
        })

    # Macro-average calculation (average over documents)
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    # Micro-average calculation (aggregate totals across all documents)
    micro_precision = total_matches / total_pred_mentions if total_pred_mentions > 0 else 0
    micro_recall = total_matches / total_gold_mentions if total_gold_mentions > 0 else 0
    micro_f1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Return both macro, micro averages, and individual document scores
    return {
        'macro_precision': avg_precision,
        'macro_recall': avg_recall,
        'macro_f1': avg_f1_score,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1_score,
        'per_document_scores': per_document_scores
    }
