import os
import sys
import argparse
import subprocess
#import spacy
import json
from openai import OpenAI
from transformers import pipeline
import random
import glob
from get_summary import get_summary, get_summary_gpt4o, get_summary_claude35, extract_gold_summaries_from_xml, extract_text_speaker_from_xml, read_documents
from parse import parse_summaries


client = OpenAI(api_key="your_openai_api_key")

#nlp = spacy.load("en_core_web_sm") #for function word detection

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

def replace_empty_strings(data):
    # Replace empty strings with "_"
    if isinstance(data, list):
        # Recursively process each item in the list
        return [replace_empty_strings(item) for item in data]
    elif isinstance(data, tuple):
        # Convert tuple to list, process, and convert back to tuple
        return tuple(replace_empty_strings(item) for item in data)
    elif isinstance(data, str):
        return "_" if data == "" else data
    else:
        # Return data as is if not list, tuple, or string
        return data

def extract_mentions_from_gold_tsv(data_folder, docnames=None):
    all_mentions = []

    # List all TSV files in the folder
    tsv_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".tsv")])

    for tsv_file in tsv_files:
        file_path = os.path.join(data_folder, tsv_file)
        docname = os.path.basename(tsv_file).split(".")[0]
        if docnames is not None:
            if docname not in docnames:
                continue
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

def get_entities_from_gold_tsv(data_folder):
    all_results = []
    tsv_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".tsv")])

    for tsv_file in tsv_files:
        filepath = os.path.join(data_folder, tsv_file)
        
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
                
                # Now only extract mentions that are first mentions 
                for col5, col6 in zip(col5_values, col6_values):
                    if col5.startswith('new') or col5.startswith('acc:com') or col5.startswith('acc:inf'):  # First mentions
                        cls_number = extract_bracketed_number(col5)
                        if cls_number:
                            if cls_number not in word_dict:
                                word_dict[cls_number] = []
                                word_indices[cls_number] = []
                                coref_indices[cls_number] = []
                            word_dict[cls_number].append(word)
                            word_indices[cls_number].append(word_index)
                            coref_indices[cls_number].append(coref_index)

            for key in word_dict:
                concatenated_words = " ".join(word_dict[key])
                concatenated_indices = ", ".join(word_indices[key])
                # Remove bracketed numbers and concatenate
                filtered_corefs = [remove_bracketed_number(coref) for coref in coref_indices[key] if coref and coref != "_"]
                concatenated_corefs = ", ".join(filtered_corefs)
                file_result.append((concatenated_words, concatenated_indices, concatenated_corefs))
        
        all_results.append(file_result)
    
    return all_results

def align_llm(doc_mentions, summary_text, doc_text):
    """
    Align mentions using GPT-4o API (chat model) with custom parameters for temperature and top_p.

    Args:
        doc_mentions (list of list of tuples): List of lists of tuples where each tuple contains (word_span, word_index, coref_index).
        summary_text (list of list of str): List of lists of summaries.
        doc_text (list of str): List of document texts (one for each summary).

    Returns:
        list of list of list of tuples: A list of lists of lists of tuples where each tuple's `word_span` is found in the corresponding document.
    """

    prompt_template = (
        "Consider the following document and summary:\n"
        "Document: {doc_text}\n"
        "Summary: {summary}\n"
        "For each of the following entities in the document, if it matches with (refers to) the ones in the summary, "
        "return the exact same entity in the phrase in bullet points. Otherwise, don't return anything. "
        "When matching, please also consider synonyms or alternative phrases that refer to the same entity. If a speaker says 'I' or is mentioned as 'you', then the speaker's name or label is considered mentioned (e.g. Kim)\n"
        "Entities: {entities}"
        "Be very precise and only return entities that match those in the summary. Do not add extra or unrelated entities.\n\n"
        "Example:\n"
        "Document:\nJennifer: We need a —  Jennifer: Do you have any sharp objects on you ? Dan: No . Dan: Keys ? Jennifer: No I need like a little pin or something . Jennifer: You have a pencil ? Dan: You have anything in your hair ? Jennifer: No . Jennifer: Fuck . Dan: What do you have to hit ? Jennifer: See this is the little -  Jennifer: Oh . Jennifer: Oh oh . Dan: Cool ? Jennifer: Okay . Jennifer: Alright . Jennifer: See , it was just slow . Jennifer: Okay . Jennifer: This is me ? Jennifer: Is this me ? Dan: Yeah . Dan: Yeah . Dan: Jennifer . Jennifer: Oh . Jennifer: That 's right . Dan: There you go thinking again . Jennifer: Smart ass . Jennifer: Smart ass . Jennifer: Alright . Dan: Wow . Dan: Who took over uh ... Jennifer: Oh . Jennifer: They got North America . Jennifer: But not for long . Jennifer: Oh , my God . Jennifer:  Oh my God , did you see that ? Dan: Because player thr- player three is aggressive , so he 's gon na like go for everything . Jennifer: How do you know ? Jennifer: Did I make him aggressive ? Dan: Yeah , you made him aggressive , so , he 's gon na like , try to tear everything up now . Dan: Um , that 's pretty well , like secure right there , so maybe —  Dan: That 's me . Jennifer: Oh fuck . Dan: Wow , he wiped my ass out . Jennifer: Ah , you suck . Jennifer: Watch this . Jennifer: Loser . Jennifer: What else can we do tomorrow ? Jennifer: Besides go to the movies , t- ? Dan: Go out to dinner ? Jennifer: I 'm so not hungry right now , it 's hard for me to think about food . Dan: Alright . Jennifer: I 'd like to go out to dinner though . Jennifer: Think we can find a hot dog ? Dan: Yeah , that 's a good idea . Dan: That 's an excellent idea . Jennifer: There you go thinking again again . Dan: There you go thinking again . Jennifer: I 'm gon na whip your butt . Dan: You think so , hunh ? Jennifer: Yeah . Dan: Un-unh . Dan: That 's all I get ? Dan: That 's me , right ? Jennifer: Yeah you get a percentage of the amount of countries you own , and then , for continents you get another set amount . Dan: So can I get something on this bad boy ? Jennifer: Yeah . Jennifer: See ? Dan: So I hit okay ? Jennifer: Yeah . Jennifer: Hit okay . Jennifer: See you got one of each kind of card . Dan: Excellent . Dan: Oh okay . Dan: So I get ... Jennifer: So you got ten , looks like sixteen . Dan: Sixteen ? Jennifer: Who you gon na trounce on ? Jennifer: That 's you up there , too , right there , you know . Dan: That 's me right there , too . Jennifer: Oh yeah . Dan: Um ... Jennifer: When w- you take over another person , you take a — you get , their cards . Jennifer: The MSG in that Chinese food really got me high for a little bit . Jennifer: Does MSG affect you ? Dan: No . Dan: Not really . Dan: It affects my mother . Dan: Gives her headaches . Jennifer: Are you gon na attack over there ? Dan: I do n't know . Dan: Thirteen . Dan: That leaves me with thirteen . Dan: I wan na fortify . Jennifer: You ca n't move those to there , because they 're not touching . Dan:  W- w- well that 's kind of bogus . Jennifer: Nun-unh . Dan: Maybe I 'll move em right there . Jennifer: Done . Dan: Done . Jennifer: Oh fuck . Jennifer: Oh . Jennifer: Who 's this guy ? Dan: Player six . Jennifer: Yakutsk . Jennifer: Look at that . Jennifer: See if I have any cards . Jennifer: Oh , I got a set . Jennifer: You know what I think , I think the first time that it does the card mode , it takes a long time . Dan: Yeah . Dan: Yeah . Jennifer: You remember the last time , that 's what happened . Dan: Yeah . Jennifer: You remember ? Dan: Yeah . Jennifer: Look at you being smart . Dan: I 'm not smart ? Jennifer: You 're stupid . Dan: Do n't call me stupid . Jennifer: Mm . Jennifer: Alright . Dan: Look at you with the uh little armies down here . Jennifer: Big armies . Dan:  Trying to — trying to win . Jennifer: I got big armies , buddy . Dan: Trying to conquer the world . Jennifer:  I 'm gon na conquer — I 'm gon na conquer you . Dan: Probably . Dan: Ooh . Dan: He 's giving you some problems over there . Jennifer: He is indeed . Dan: Go for that one . Dan: Go into Europe . Dan: Get Europe . Jennifer: Oops . Jennifer: You wo n't attack me yet . Jennifer: I think I 'll stop there . Dan: Hmm . Dan: I only have uh , that many cards , so ... Jennifer: How many cards you have ? Jennifer: You only have two . Dan: Just two . Jennifer: So you ca n't have a set . Dan: When do you get — h- — when do you get cards though ? Dan: I do n't understand that . Jennifer: Every time you take over a country you get cards . Dan: What row ? Jennifer: Attack with the twenty - two . Jennifer: Press twenty - two , attack . Dan: Wow . Jennifer: Look at that . Jennifer: Oh , see look , you just got all of his cards . Jennifer: Press okay . Dan: Bonus ? Jennifer: Oh my God . Jennifer: Fuck . Jennifer: Fuck . Jennifer: Fuck fuck fuck . Jennifer: Oh man . Jennifer: Look at that . Jennifer:  Twenty - seven . Jennifer:  Twenty - nine . Jennifer:  Th –  twenty - one two three four five six seven eight . Jennifer:  Twenty - eight . Jennifer: Do n't you fucking attack me . Jennifer: You ass . Jennifer: You asshole . Dan:  Two — hmm . Jennifer: I 'm tired ."
        "Summary: \nTwo people are playing a strategy game online involving cards and attacking countries, while discussing dinner plans.\n"
        "Entities: we, you, I, countries you own, dinner, a percentage of the amount of countries you own, player six, their cards, one of each kind of card.\n"
        "Answer: we, dan, jennifer, countries you own, dinner, their cards.\n\n"
    )

    results = []

    # Lowercase all words in doc_mentions, summary_text, and doc_text for normalization
    doc_mentions_lower = [[(span.lower(), idx, coref) for span, idx, coref in mentions] for mentions in doc_mentions]
    summary_text_lower = [[summary.lower() for summary in summaries] for summaries in summary_text]
    doc_text_lower = [doc.lower() for doc in doc_text]  # Ensure document text is lowercased

    num_summaries = len(summary_text_lower[0])
    summaries_by_index = [[] for _ in range(num_summaries)]

    for doc_summaries in summary_text_lower:
        for i, summary in enumerate(doc_summaries):
            summaries_by_index[i].append(summary)

    # Process each summary
    for summary_idx in range(num_summaries):
        summary_results = []
        for doc_idx in range(len(doc_mentions_lower)):
            summary = summaries_by_index[summary_idx][doc_idx]
            doc = doc_mentions_lower[doc_idx]
            total_entities = len(doc)

            # Split doc_mentions into non-overlapping chunks of 15-20 entities
            entity_indices = list(range(total_entities))
            random.shuffle(entity_indices)  # Shuffle the indices for randomness

            # Create chunks of 15-20 entities (non-overlapping)
            chunks = [entity_indices[i:i + random.randint(15, 20)] for i in range(0, total_entities, 20)]

            all_extracted_mentions = []  # To store the results for each document

            for chunk in chunks:
                # Select the entities corresponding to the current chunk
                selected_entities = [doc[i] for i in chunk]
                entities_str = ", ".join([span for span, _, _ in selected_entities])  # Join entities into a single string

                prompt = prompt_template.format(
                    doc_text=doc_text_lower[doc_idx],  # Use lowercased doc_text
                    summary=summary,
                    entities=entities_str
                )

                # Making a chat completion request using the client object
                response = client.chat.completions.create(
                    model="gpt-4o",  # Use the gpt-4o chat model
                    messages=[
                        {"role": "system", "content": "You are an assistant for aligning entity mentions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,  # Adjust as necessary to handle multiple mentions
                    temperature=0.2,  # Lower temperature for more deterministic results
                    top_p=0.7  # Lower top_p for higher precision and less diversity
                )

                # Extract and parse the model response
                answer = response.choices[0].message.content.strip().split("\n")
                cleaned_ans = [s.lstrip("- ").replace("**", "").replace("*", "").replace('"', '').replace('“ ', '').replace('#', '').replace(' [', '').replace(']', '').replace('\n', '').split(':', 1)[-1].lower().strip() for s in answer]

                # Extract mentions from the API response
                extracted_mentions = []
                for ans in cleaned_ans:
                    for span, idx, coref in selected_entities:
                        if ans == span:
                            extracted_mentions.append((span, idx, coref))
                            break
                # Store the extracted mentions
                all_extracted_mentions.extend(extracted_mentions)
            # Append the combined results from multiple queries for one document
            summary_results.append(all_extracted_mentions if all_extracted_mentions else [])

        results.append(summary_results)

    return results

def align_llm_hf(doc_mentions, summary_text, model_name="google/flan-t5-xl"):
    """
    Align mentions using a Huggingface model.

    Args:
        doc_mentions (list of list of tuples): List of lists of tuples where each tuple contains (word_span, word_index, coref_index).
        summary_text (list of list of str): List of lists of summaries.
        model_name (str): Name of the Huggingface model to use.

    Returns:
        list of list of list of tuples: A list of lists of lists of tuples where each tuple's `word_span` is found in the corresponding document.
    """
    aligner = pipeline("text2text-generation", model=model_name, device=0) # Run on GPU
    
    prompt_template = (
        "Document: {doc_text}\n"
        "Summary: {summary}\n"
        "For each entity in the document, determine if it aligns with (or makes an equivalent reference to) any word span in the summary. "
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

            # Append extracted mentions or empty list
            summary_results.append(extracted_mentions if extracted_mentions else [])
        
        results.append(summary_results)

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
    #stop_words = {word for word in spacy.lang.en.stop_words.STOP_WORDS}
    stop_words = {'should', 'my', 'because', 'yourselves', 'not', 'made', '‘ve', 'even', 'almost', 'more', 'mine', 'from', 'nor', "'d", 'six', 'whence', 'they', 'due', 'twenty', 'serious', 'could', 'fifty', 'whoever', 'put', 'move', 'an', 'back', 'meanwhile', 'used', 'next', 'somewhere', 'unless', 'once', 'somehow', 'other', 'amount', 'rather', 'elsewhere', 'at', 'indeed', 'were', 'mostly', 'top', 'bottom', 'enough', 'anything', 'never', 'beyond', 'than', 'thru', 'about', 'either', 'wherein', 'itself', 'until', 'hundred', 'per', 'except', 'is', 'becomes', 'below', 'toward', 'therein', '’s', 'doing', 'nobody', 'so', 'seemed', 'over', 'them', 'during', 'eight', '’ve', 'anyone', 'have', 'yourself', 'much', 'nevertheless', 'then', 'towards', 'was', 'least', 'while', 'i', 'namely', 'since', 'well', "'m", 'been', 'where', 'which', "n't", 'such', 'yours', 'ca', 'nothing', 'name', 'something', 'five', 'really', 'whose', 'else', 'seeming', 're', 'of', 'thereby', 'but', 'under', 'show', 'being', 'neither', 'thereafter', 'whether', 'thereupon', 'it', 'every', 'again', '’m', 'noone', 'you', 'take', 'only', 'hers', 'already', 'out', 'into', 'wherever', 'down', 'within', 'also', 'there', 'now', 'say', "'ll", 'him', 'its', '’d', 'himself', 'former', 'another', 'any', 'empty', '‘s', 'however', 'their', 'besides', 'by', 'her', 'how', 'one', 'cannot', 'afterwards', 'front', 'seems', 'would', 'herein', 'anyway', 'yet', 'n‘t', 'will', 'does', '’re', 'anywhere', 'those', 'between', "'s", 'the', 'above', 'around', 'none', 'when', 'sixty', 'keep', 'everyone', 'done', 'sometime', 'whole', 'she', 'further', 'anyhow', 'for', 'upon', 'someone', 'us', 'whereupon', '‘ll', 'n’t', 'among', 'nine', 'see', 'twelve', 'go', 'become', 'regarding', 'through', 'therefore', 'ours', 'beside', 'together', 'first', 'everything', 'must', 'alone', 'very', 'two', 'amongst', 'using', 'ever', 'beforehand', 'thus', 'before', '‘d', 'whenever', 'many', '’ll', 'always', 'whereas', 'hereafter', 'if', 'everywhere', 'on', 'has', 'nowhere', 'throughout', 'thence', 'without', 'may', 'still', 'along', 'whatever', 'please', 'that', "'ve", 'in', 'hereby', 'me', 'he', 'or', 'part', 'your', 'third', 'as', 'perhaps', 'various', 'although', 'against', 'formerly', 'full', 'off', 'eleven', 'too', 'with', 'herself', 'had', 'themselves', 'are', 'myself', 'though', 'ten', 'his', 'what', 'make', 'own', 'be', 'and', 'forty', 'whom', 'moreover', 'hereupon', 'ourselves', 'who', 'some', 'these', 'last', 'just', 'others', 'each', 'might', 'call', 'no', 'becoming', 'all', 'did', '‘re', 'to', 'can', 'less', 'few', 'same', 'why', 'most', 'fifteen', 'do', '‘m', 'onto', 'whereafter', 'our', 'four', 'often', 'am', 'via', 'latterly', 'hence', 'we', 'whereby', 'whither', 'a', 'side', 'several', 'after', 'three', 'seem', 'behind', 'here', "'re", 'latter', 'otherwise', 'quite', 'this', 'get', 'both', 'became', 'sometimes', 'across', 'give', 'up'}

    results = []

    num_summaries = len(mention_text[0])
    for doc_idx, doc in enumerate(doc_mentions):
        summary_alignments = []
        for summary_idx in range(num_summaries):
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
                        candidates = [span for span, _, _ in doc]
                        if mention in candidates:
                            for span, idx, coref in doc:
                                if span == mention:
                                    extracted_mentions.append((span, idx, coref))
                        else:
                            for i in range(len(words) - 1):
                                match_span = ' '.join(words[i:i + 3])
                                if match_span in candidates:
                                    for span, idx, coref in doc:
                                        if match_span in span:
                                            extracted_mentions.append((span, idx, coref))
                                            break

            # If no matches found this will append an empty list for that summary
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

    # Fill empty lists where there are no matches
    for summary_idx in range(n_summaries):
        for doc_idx in range(num_documents):
            if not organized_predictions[summary_idx][doc_idx]:
                organized_predictions[summary_idx][doc_idx] = []  # Append an empty list if no matches found

    return organized_predictions


def align_stanza(summary_text, doc_mentions, doc_text):
    import stanza
    tokenizer = stanza.Pipeline("en", processors="tokenize")
    coref = stanza.Pipeline("en", processors="tokenize,coref")

    output = []
    for i, doc in enumerate(doc_text):
        doc_output = []
        for summary in summary_text[i]:
            summary_output = []

            # Tokenize the summary and prepare for coreference
            tokenized_summary = tokenizer(summary)
            summary_tokens = [word.text for sent in tokenized_summary.sentences for word in sent.words]
            section_marker = "==="

            # Concatenate document and summary text with a marker
            tokenized_doc_with_summary = doc.strip() + " " + section_marker + " " + " ".join(summary_tokens).strip()
            doc_coref = coref(tokenized_doc_with_summary)

            # Identify sentences that belong to the summary section
            summary_sents = len(tokenized_summary.sentences)
            all_sents = list(range(len(doc_coref.sentences)))
            summary_sents = all_sents[-summary_sents:]

            # Extract mentions that have antecedents in the document section
            for coref_chain in doc_coref.coref:
                # Check if any mention exists in both document and summary sections
                doc_mentions_in_chain = []
                summary_mentions_in_chain = []
                for mention in coref_chain.mentions:
                    if mention.sentence in summary_sents:
                        summary_mentions_in_chain.append(mention)
                    else:
                        doc_mentions_in_chain.append(mention)

                # Only add document mentions that have corresponding summary mentions
                if summary_mentions_in_chain and doc_mentions_in_chain:
                    for mention in doc_mentions_in_chain:
                        start, end = mention.start_word, mention.end_word
                        mention_text = " ".join(doc_coref.sentences[mention.sentence].words[i].text for i in range(start, end))

                    for mention in summary_mentions_in_chain:
                        start, end = mention.start_word, mention.end_word
                        mention_text = " ".join(doc_coref.sentences[mention.sentence].words[i].text for i in range(start, end))

                    # Append mentions from document section to summary_output if they appear in summary section
                    for mention in doc_mentions_in_chain:
                        mention_text = " ".join(
                            doc_coref.sentences[mention.sentence].words[i].text for i in range(mention.start_word, mention.end_word)
                        )
                        indices = ",".join(f"{mention.sentence + 1}-{i + 1}" for i in range(mention.start_word, mention.end_word))
                        summary_output.append((mention_text, indices, str(coref_chain.index)))

            # Append only if summary_output has content to ensure unique doc_output
            if summary_output:
                doc_output.append(summary_output)
        output.append(doc_output)

    return output

def align(doc_mentions, summary_text, mention_text, doc_text, data_folder, n_summaries, component="string_match", partition="test"):
    if component == "LLM":
        return align_llm(doc_mentions, summary_text, doc_text)
    elif component == "LLM_hf":
        return align_llm_hf(doc_mentions, summary_text)
    elif component == "string_match":
        return align_string_match(doc_mentions, mention_text)
    elif component == "coref_system":
        return align_coref_system(data_folder, n_summaries)
    elif component == "stanza":
        return align_stanza(summary_text, doc_mentions, doc_text)
    else:
        raise ValueError(f"Unknown alignment component: {component}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align document mentions based on the selected component")
    parser.add_argument("--data_folder", required=False, default="data", help="Path to the data folder")
    parser.add_argument("--model_name", default="google/flan-t5-xl", choices=["gpt4o", "claude-3-5-sonnet-20241022","meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"], help="Model name to use for summarization")
    parser.add_argument("--n_summaries", type=int, required=False, default=1, help="Number of summaries")
    parser.add_argument("--max_docs", type=int, default=None, help="Maximum number of documents to processe (default: None = all; choose a small number to prototype)")
    parser.add_argument("--component", required=False, default="string_match", choices=["LLM", "LLM_hf", "string_match", "coref_system", "stanza"], help="Component to use for alignment")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached summaries (default: False)")
    parser.add_argument("--partition", required=False, default="test", choices=["test","dev", "train"], help="Data partition to use for alignment")

    args = parser.parse_args()

    #pred_tsv_folder =args.data_folder + '/output/pred_tsv'
    doc_ids, doc_texts = read_documents(args.data_folder, args.partition)
    if args.max_docs is not None:
        doc_ids = doc_ids[:args.max_docs]
        doc_texts = doc_texts[:args.max_docs]
    #print('doc_ids:',doc_ids)

    all_entities_from_tsv = get_entities_from_gold_tsv(args.data_folder + '/input/tsv/'+ args.partition)
    gold_summaries = extract_gold_summaries_from_xml(args.data_folder + '/input/xml/'+ args.partition)
    if args.model_name=="gpt4o":
        summaries = get_summary_gpt4o(doc_texts, doc_ids, args.data_folder, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    elif args.model_name=="claude-3-5-sonnet-20241022":
        summaries = get_summary_claude35(doc_texts, doc_ids, args.data_folder, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    else:
        summaries = get_summary(doc_texts, doc_ids, args.data_folder, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    
    sum1_mentions = parse_summaries(list(gold_summaries.values()))
    all_mentions = parse_summaries(list(summaries.values()))
    #print('all mentions:',all_mentions, '\n','len of all mentions:', len(all_mentions[0])) 
    #folders_with_pred_tsv = [os.path.join(pred_tsv_folder, f'tsv_pred_{args.partition}{i}') for i in range(1, args.n_summaries + 1) if glob.glob(os.path.join(pred_tsv_folder, f'tsv_pred_{args.partition}{i}', '*.tsv'))] 
    doc_sp_texts = extract_text_speaker_from_xml(args.data_folder + '/input/xml/'+ args.partition)
    #print('doc_sp_texts:',doc_sp_texts, '\n','len of doc_sp_texts:', len(doc_sp_texts))

    alignments = align(
        doc_mentions=all_entities_from_tsv,
        summary_text=list(summaries.values()),
        mention_text=all_mentions,
        doc_text=doc_sp_texts,
        n_summaries=args.n_summaries,
        component=args.component,
        partition=args.partition
    )

    if args.component == "LLM":
        print(f"LLM Alignment Result:\n{alignments}")
    elif args.component == "LLM_hf":
        print(f"LLM_hf Alignment Result:\n{alignments}")
    elif args.component == "string_match":
        print(f"String Match Alignment Result:\n{alignments}")
    elif args.component == "coref_system":
        print(f"Coreference MTL Alignment Result:\n{alignments}")
    elif args.component == "stanza":
        print(f"Stanza Alignment Result:\n{alignments}")
