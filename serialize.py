import xml.etree.ElementTree as ET
import os
import re


def add_summaries_to_xml(data_folder, summaries):
    """
    Adds summaries to XML files.

    Args:
        data_folder (str): Directory containing input and output folders.
        summaries (dict): Dictionary where keys are doc_ids and values are lists of summaries.
    """
    xml_folder = os.path.join(data_folder, "input/xml")
    output_folder = os.path.join(data_folder, "output/xml")

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Regex to match "Summary {n}:" pattern
    summary_pattern = re.compile(r"^Summary \d+:\s*")

    # Process each XML file
    for doc_id, summaries_list in summaries.items():
        # Construct the full path to the XML file
        xml_file = os.path.join(xml_folder, f"{doc_id}.xml")

        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Check if the root tag is 'text'
        if root.tag == "text":
            text_element = root
        else:
            # Find the <text> element
            text_element = root.find(".//text")

        if text_element is not None:
            # Find existing summary elements
            existing_summaries = [elem for elem in text_element.attrib if elem.startswith("summary")]

            # Add new summaries
            for i, summary in enumerate(summaries_list, start=len(existing_summaries) + 1):
                # Remove "Summary {n}:" from the beginning of the summary text
                summary = summary_pattern.sub("", summary)
                summary_tag = f"summary{i}"
                text_element.set(summary_tag, summary)

            # Write the modified XML to the output directory
            output_path = os.path.join(output_folder, f"{doc_id}.xml")
            tree.write(output_path, encoding="utf-8", xml_declaration=True)


def add_anno_to_tsv(data_folder, model_predictions, partition, max_docs):
    """
    Modify the salience columns in TSV files using model predictions.
    Maintains order of annotations and groups tokens by bracket numbers.

    Args:
        data_folder (str): Directory containing input and output folders.
        model_predictions (list): List of lists containing (entity, predictions) pairs.
        partition (str): Data partition to process.
        max_docs (int): Maximum number of documents to process.
    """
    tsv_dir = os.path.join(data_folder, "input/tsv", partition)
    output_dir = os.path.join(data_folder, "output/tsv", partition)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each TSV file
    for filename in sorted(os.listdir(tsv_dir))[:max_docs]:
        if filename.endswith(".tsv"):
            input_file = os.path.join(tsv_dir, filename)
            output_file = os.path.join(output_dir, filename)

            with open(input_file, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()

            # First pass: Group tokens by bracket numbers to identify entities
            entities = {}  # {bracket_num: {'tokens': [], 'indices': [], 'original_sal': []}}
            current_line_idx = 0

            while current_line_idx < len(lines):
                line = lines[current_line_idx].strip()
                if line and not line.startswith('#'):
                    columns = line.strip().split('\t')
                    if len(columns) >= 9:
                        token = columns[2]
                        salience = columns[5]

                        if salience != "_":
                            parts = salience.split('|')
                            for part_idx, part in enumerate(parts):
                                if '[' in part:
                                    bracket_num = part[part.find('[')+1:part.find(']')]
                                    key = f"{bracket_num}_{part_idx}"
                                    if key not in entities:
                                        entities[key] = {'tokens': [], 'indices': [], 'original_sal': []}
                                    entities[key]['tokens'].append(token)
                                    entities[key]['indices'].append(current_line_idx)
                                    entities[key]['original_sal'].append(part[:part.find('[')])

                current_line_idx += 1

            # Second pass: Update annotations
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for line_idx, line in enumerate(lines):
                    if line.startswith('#') or not line.strip():
                        outfile.write(line)
                        continue

                    columns = line.strip().split('\t')
                    if len(columns) < 9 or columns[5] == "_":
                        outfile.write(line)
                        continue

                    salience_parts = columns[5].split('|')
                    new_salience_parts = []

                    for part_idx, part in enumerate(salience_parts):
                        if '[' not in part:
                            new_salience_parts.append(part)
                            continue

                        bracket_num = part[part.find('[')+1:part.find(']')]
                        key = f"{bracket_num}_{part_idx}"

                        if key in entities:
                            entity_tokens = entities[key]['tokens']
                            entity_text = ' '.join(entity_tokens).lower()
                            
                            # Get base annotation (s/n)
                            base = 's' if part.startswith('sal') else 'n'

                            # Look for entity in model_predictions while maintaining order
                            model_pred = None
                            for doc_predictions in model_predictions:
                                for mention, predictions in doc_predictions:
                                    if mention.lower() == entity_text:
                                        model_pred = predictions[:]  # Get all 4 characters
                                        break
                                if model_pred:
                                    break

                            # Construct new salience string
                            if model_pred:
                                new_part = base + model_pred
                            else:
                                new_part = base + '_'*4 

                            # Add back the bracket information
                            new_part += part[part.find('['):]
                            new_salience_parts.append(new_part)
                        else:
                            new_salience_parts.append(part)

                    columns[5] = '|'.join(new_salience_parts)
                    outfile.write('\t'.join(columns) + '\n')
