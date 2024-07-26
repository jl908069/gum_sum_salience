import os
import sys
import subprocess
import spacy
import json
import openai

# Add coref-mtl to the Python path
#sys.path.append(os.path.abspath("coref-mtl"))
openai.api_key = "your_openai_api_key"

nlp = spacy.load("en_core_web_sm") #for function word detection

def extract_mentions_from_gold_tsv(data_folder):
    all_mentions = []

    # List all TSV files in the folder
    tsv_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".tsv")])

    for tsv_file in tsv_files:
        file_path = os.path.join(data_folder, tsv_file)
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
    tsv_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".tsv")])

    for tsv_file in tsv_files:
        file_path = os.path.join(data_folder, tsv_file)
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

def align_llm(doc_mentions, summary_text):
    """
    Align mentions using GPT-4 API.

    Args:
        doc_mentions (list of list of tuples): List of lists of tuples where each tuple contains (word_span, word_index, coref_index).
        summary_text (list of list of str): List of lists of summaries.

    Returns:
        list of list of list of tuples: A list of lists of lists of tuples where each tuple's `word_span` is found in the corresponding document.
    """
    client = openai.OpenAI()
    
    assistant = client.beta.assistants.create(
        name="Document Aligner",
        instructions="You are an assistant for aligning mentions from summary text with document text. For each mention in the summary, determine if it aligns with any word span in the document.",
        model="gpt-4o"
    )
    
    prompt_template = (
        "Document: {doc_text}\n"
        "Summary: {summary}\n"
        "For each mention in the summary, determine if it aligns with (or make an equivalent reference to) any word span in the document. "
        "Return a list of matching word spans from the document."
    )

    results = []

    # Lowercase all words in doc_mentions and summary_text
    doc_mentions_lower = [[(span.lower(), idx, coref) for span, idx, coref in mentions] for mentions in doc_mentions]
    summary_text_lower = [[summary.lower() for summary in summaries] for summaries in summary_text]

    # Extract each summary through all documents to a list of summaries
    num_summaries = len(summary_text_lower[0])
    summaries_by_index = [[] for _ in range(num_summaries)]
    
    for doc_summaries in summary_text_lower:
        for i, summary in enumerate(doc_summaries):
            summaries_by_index[i].append(summary)
    
    # Process each list of summaries
    for summary_idx in range(num_summaries):
        summary_results = []
        for doc_idx in range(len(doc_mentions_lower)):
            summary = summaries_by_index[summary_idx][doc_idx]
            doc = doc_mentions_lower[doc_idx]
            prompt = prompt_template.format(
                doc_text=" ".join([span for span, _, _ in doc]),
                summary=summary
            )

            response = client.completions.create(
                assistant=assistant,
                prompt=prompt,
                max_tokens=150  # Adjust as necessary to handle multiple mentions
            )

            answer = response.choices[0].text.strip().split("\n")
            extracted_mentions = []

            for ans in answer:
                for span, idx, coref in doc:
                    if ans in span:
                        extracted_mentions.append((span, idx, coref))
                        break

            if extracted_mentions:
                summary_results.append(extracted_mentions)
        
        results.append(summary_results)

    return results

def align_llm_hf(doc_mentions, summary_text, model_name="google/flan-t5-xl"): # Make user specify this
    """
    Align mentions using a Huggingface model.

    Args:
        doc_mentions (list of list of tuples): List of lists of tuples where each tuple contains (word_span, word_index, coref_index).
        summary_text (list of list of str): List of lists of summaries.
        model_name (str): Name of the Huggingface model to use.

    Returns:
        list of list of list of tuples: A list of lists of lists of tuples where each tuple's `word_span` is found in the corresponding document.
    """
    aligner = pipeline("text2text-generation", model=model_name)
    
    prompt_template = (
        "Document: {doc_text}\n"
        "Summary: {summary}\n"
        "For each mention in the summary, determine if it aligns with (or make an equivalent reference to) any word span in the document. "
        "Return a list of matching word spans from the document."
    )

    results = []

    # Lowercase all words in doc_mentions and summary_text
    doc_mentions_lower = [[(span.lower(), idx, coref) for span, idx, coref in mentions] for mentions in doc_mentions]
    summary_text_lower = [[summary.lower() for summary in summaries] for summaries in summary_text]

    # Extract each summary through all documents to a list of summaries
    num_summaries = len(summary_text_lower[0])
    summaries_by_index = [[] for _ in range(num_summaries)]
    
    for doc_summaries in summary_text_lower:
        for i, summary in enumerate(doc_summaries):
            summaries_by_index[i].append(summary)
    
    # Process each list of summaries
    for summary_idx in range(num_summaries):
        summary_results = []
        for doc_idx in range(len(doc_mentions_lower)):
            summary = summaries_by_index[summary_idx][doc_idx]
            doc = doc_mentions_lower[doc_idx]
            prompt = prompt_template.format(
                doc_text=" ".join([span for span, _, _ in doc]),
                summary=summary
            )

            response = aligner(prompt, max_length=150, num_return_sequences=1)

            answer = response[0]['generated_text'].strip().split("\n")
            extracted_mentions = []

            for ans in answer:
                for span, idx, coref in doc:
                    if ans in span:
                        extracted_mentions.append((span, idx, coref))
                        break

            summary_results.append(extracted_mentions)
        
        results.append(summary_results)

    # Remove empty lists from results
    results = [[mentions for mentions in doc_results if mentions] for doc_results in results]

    return results

def align_string_match(doc_mentions, mention_text):
    """
    Extracts tuples from `doc_mentions` where `mention_text` is found in the corresponding document.

    Args:
        doc_mentions (list of list of tuples): List of lists of tuples where each tuple contains (word_span, word_index, coref_index).
        mention_text (list of list of list of list of str): List of lists of lists of mention strings from the summary.

    Returns:
        list of list of list of tuples: A new list of lists of lists of tuples where each tuple's `word_span` is found in the corresponding document.
    """
    # Ensure doc_mentions is in lower case
    doc_mentions = [[(span.lower(), idx, coref) for span, idx, coref in mentions] for mentions in doc_mentions]

    # Flatten mention_text structure and convert to lower case
    mention_text = [[[[mention.lower() for mention in summary] for summary in document_summaries] for document_summaries in document] for document in mention_text]

    # Automatically detect function words using spacy
    stop_words = {word for word in spacy_nlp.Defaults.stop_words}

    results = []

    num_summaries = len(mention_text[0])
    for summary_idx in range(num_summaries):
        summary_alignments = []
        for doc_idx, doc in enumerate(doc_mentions):
            mentions = mention_text[doc_idx][summary_idx]  # Extract the list of mentions for the summary
            extracted_mentions = []

            for summary in mentions:
                for mention in summary:
                    words = mention.split()
                    if len(words) < 2:
                        if mention in [span for span, _, _ in doc] and mention not in stop_words:
                            for span, idx, coref in doc:
                                if span == mention:
                                    extracted_mentions.append((span, idx, coref))
                    elif len(words) == 2:
                        if mention in [span for span, _, _ in doc]:
                            for span, idx, coref in doc:
                                if span == mention:
                                    extracted_mentions.append((span, idx, coref))
                        else:
                            for word in words:
                                if word in [span for span, _, _ in doc] and word not in stop_words:
                                    for span, idx, coref in doc:
                                        if word in span:
                                            extracted_mentions.append((span, idx, coref))
                                            break
                    else:
                        if mention in [span for span, _, _ in doc]:
                            for span, idx, coref in doc:
                                if span == mention:
                                    extracted_mentions.append((span, idx, coref))
                        else:
                            for i in range(len(words) - 1):
                                match_span = ' '.join(words[i:i + 3])
                                if match_span in [span for span, _, _ in doc]:
                                    for span, idx, coref in doc:
                                        if match_span in span:
                                            extracted_mentions.append((span, idx, coref))
                                            break

            if not extracted_mentions:  # Add "No match" if no matches found
                extracted_mentions.append("No match")

            summary_alignments.append(extracted_mentions)
        
        results.append(summary_alignments)

    return results

def align_coref_system(data_folders, n_summaries):
    """
    Align mentions using a coreference system.

    Args:
        data_folders (list of str): List of paths to folders containing TSV files with predictions for each summary.
        n_summaries (int): Number of summaries to use.

    Returns:
        list of list of list of tuples: Organized predictions.
    """
    predictions_list = []

    for i in range(n_summaries):
        folder_path = data_folders[i]
        predictions = extract_mentions_from_pred_tsv(folder_path)
        predictions_list.append(predictions)

    num_documents = len(predictions_list[0])  # Number of documents
    organized_predictions = [[[] for _ in range(num_documents)] for _ in range(n_summaries)]

    for summary_idx, predictions in enumerate(predictions_list):
        for doc_idx, doc_predictions in enumerate(predictions):
            organized_predictions[summary_idx][doc_idx] = doc_predictions

    # Fill "No match" where there are empty lists
    for summary_idx in range(n_summaries):
        for doc_idx in range(num_documents):
            if not organized_predictions[summary_idx][doc_idx]:
                organized_predictions[summary_idx][doc_idx] = ["No match"]

    return organized_predictions

def align(doc_mentions, summary_text, mention_text, data_folder, n_summaries, component="LLM"):
    if component == "LLM":
        return align_llm(doc_mentions, summary_text)
    elif component == "string_match":
        return align_string_match(doc_mentions, mention_text)
    elif component == "coref_system":
        return align_coref_system(data_folder, n_summaries)
    else:
        raise ValueError(f"Unknown alignment component: {component}")
