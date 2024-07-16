import os
import sys
import subprocess
import spacy
import json

# Add coref-mtl to the Python path
#sys.path.append(os.path.abspath("coref-mtl"))

nlp = spacy.load("en_core_web_sm") #for function word detection

def extract_mentions_from_gold_tsv(data_folder):
    all_mentions = []

    # List all TSV files in the folder
    tsv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".tsv")])

    for tsv_file in tsv_files:
        file_path = os.path.join(folder_path, tsv_file)
        mentions = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        current_mentions = {}

        for line in lines:
            if line.startswith("#") or not line.strip():
                continue

            columns = line.strip().split("\t")
            word_index = columns[0]
            word = columns[2]
            entity_type = columns[3]
            coref_info = columns[9]
            
            # Handle nested mentions
            entity_types = entity_type.split("|")

            # Track current mentions for each entity type
            for i, entity in enumerate(entity_types):
                if entity == "_" or not entity:
                    continue

                if entity not in current_mentions:
                    current_mentions[entity] = ([], [], coref_info)

                current_mentions[entity][0].append(word)
                current_mentions[entity][1].append(word_index)

            # Check for completion of current mentions
            completed_mentions = []
            for entity in list(current_mentions.keys()):
                if entity not in entity_types:
                    completed_mentions.append(entity)

            # Add completed mentions to the result
            for entity in completed_mentions:
                word_span, indices, coref_index = current_mentions.pop(entity)
                mentions.append((" ".join(word_span), ",".join(indices), coref_index))

        # Add any remaining mention
        for entity, (word_span, indices, coref_index) in current_mentions.items():
            mentions.append((" ".join(word_span), ",".join(indices), coref_index))

        all_mentions.append(mentions)

    return all_mentions

def extract_mentions_from_pred_tsv(data_folder):
    all_mentions = []

    # List all TSV files in the folder
    tsv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".tsv")])

    for tsv_file in tsv_files:
        file_path = os.path.join(folder_path, tsv_file)
        mentions = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        current_mentions = {}
        largest_sentence_index = -1

        # Determine the largest sentence index
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            columns = line.strip().split("\t")
            sentence_index = int(columns[0].split("-")[0])
            if sentence_index > largest_sentence_index:
                largest_sentence_index = sentence_index

        # Parse only the lines with the largest sentence index
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue

            columns = line.strip().split("\t")
            sentence_index = int(columns[0].split("-")[0])
            if sentence_index == largest_sentence_index:
                word_index = columns[0]
                word = columns[2]
                entity_type = columns[3]
                coref_info = columns[-1]

                # Handle nested mentions
                entity_types = entity_type.split("|")
                coref_infos = coref_info.split("|")

                # Track current mentions for each entity type
                for i, entity in enumerate(entity_types):
                    if entity == "_" or not entity:
                        continue

                    if entity not in current_mentions:
                        current_mentions[entity] = ([], [], coref_infos[i] if i < len(coref_infos) else "_")

                    current_mentions[entity][0].append(word)
                    current_mentions[entity][1].append(word_index)

                # Check for completion of current mentions
                completed_mentions = []
                for entity in list(current_mentions.keys()):
                    if entity not in entity_types:
                        completed_mentions.append(entity)

                # Add completed mentions to the result
                for entity in completed_mentions:
                    word_span, indices, coref_index = current_mentions.pop(entity)
                    mentions.append((" ".join(word_span), ",".join(indices), coref_index))

        # Add any remaining mention
        for entity, (word_span, indices, coref_index) in current_mentions.items():
            mentions.append((" ".join(word_span), ",".join(indices), coref_index))

        all_mentions.append(mentions)

    return all_mentions

def align_llm(doc_mentions, summary_text, mention_text):
    # Implementation for alignment using LLM (Huggingface API)
    pass

def align_string_match(doc_mentions, mention_text):
    """
    Extracts tuples from `doc_mentions` where `mention_text` is found in the corresponding document.

    Args:
        doc_mentions (list of list of tuples): List of lists of tuples where each tuple contains (word_span, word_index, coref_index).
        mention_text (list of str): List of mention strings from the summary.

    Returns:
        list of list of tuples: A new list of lists of tuples where each tuple's `word_span` is found in the corresponding document.
    """
    doc_mentions = [[(span.lower(), idx, coref) for span, idx, coref in mentions] for mentions in doc_mentions]
    mention_text = [text.lower() for text in mention_text]

    # Automatically detect function words using spacy
    stop_words = {word for word in nlp.Defaults.stop_words}

    results = []

    for doc, mentions in zip(doc_mentions, mention_text):
        extracted_mentions = []
        for mention in mentions:
            word_span = mention[0]
            words = word_span.split()
            if len(words) < 2:
                if word_span in doc:
                    extracted_mentions.append(mention)
            elif len(words) == 2:
                if word_span in doc:
                    extracted_mentions.append(mention)
                else:
                    for word in words:
                        if word in doc and word not in stop_words:
                            extracted_mentions.append(mention)
                            break
            else:
                if word_span in doc:
                    extracted_mentions.append(mention)
                else:
                    for i in range(len(words) - 1):
                        match_span = ' '.join(words[i:i + 3])
                        if match_span in doc:
                            extracted_mentions.append(mention)
                            break
        results.append(extracted_mentions)

    return results

def align_coref_system(data_folder):

    # Extract mentions from TSV folder
    predictions = extract_mentions_from_pred_tsv(data_folder) # "path/to/predictions tsv files" 

    return predictions

def align(doc_mentions, summary_text, mention_text, data_folder, component="LLM"):
    if component == "LLM":
        return align_llm(doc_mentions, summary_text, mention_text)
    elif component == "string_match":
        return align_string_match(doc_mentions, mention_text)
    elif component == "coref_system":
        return align_coref_system(data_folder)
    else:
        raise ValueError(f"Unknown alignment component: {component}")
