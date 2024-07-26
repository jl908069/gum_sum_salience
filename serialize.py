import xml.etree.ElementTree as ET
import os
import re
#from get_summary import get_summary
import argparse

def add_summaries_to_xml(xml_folder, summaries, output_folder):
    """
    Adds summaries to XML files.

    Args:
        xml_folder (str): Path to the folder containing XML files.
        summaries (dict): Dictionary where keys are doc_ids and values are lists of summaries.
        output_folder (str): Path to the output folder to save modified XML files.
    """
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

def add_anno_to_tsv(tsv_dir, output_dir, alignments):
    """
    Modify the salience columns in TSV files and add alignments.

    Args:
        tsv_dir (str): Directory containing input TSV files.
        output_dir (str): Directory to save output TSV files.
        alignments (list): List of alignments for each summary.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each file in the input directory
    for filename in os.listdir(tsv_dir):
        if filename.endswith(".tsv"):
            input_file = os.path.join(tsv_dir, filename)
            output_file = os.path.join(output_dir, filename)
            with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
                lines = infile.readlines()
                for line in lines:
                    if line.startswith('#') or not line.strip():
                        outfile.write(line)
                        continue
                    columns = line.strip().split('\t')
                    if len(columns) < 10:
                        outfile.write(line)
                        continue
                    token = columns[2]
                    salience = columns[5]

                    if salience == "_":
                        outfile.write(line)
                        continue

                    # Split and replace salience values
                    salience_parts = salience.split('|')
                    new_salience_parts = []

                    for part in salience_parts:
                        bracket_idx = part.find('[')
                        base = part[:bracket_idx] if bracket_idx != -1 else part
                        if base.startswith('sal'):
                            new_part = 's'
                        elif base.startswith('nonsal'):
                            new_part = 'n'
                        else:
                            new_part = base
                        if bracket_idx != -1:
                            new_part += part[bracket_idx:]
                        new_salience_parts.append(new_part)

                    # Initial salience marks from the original TSV file
                    salience_marks_list = [""] * len(new_salience_parts)

                    for i in range(len(new_salience_parts)):
                        if new_salience_parts[i][0] in ['s', 'n']:
                            salience_marks_list[i] = new_salience_parts[i][0]

                    # Add initial salience annotation
                    for i in range(len(salience_marks_list)):
                        if not salience_marks_list[i]:
                            salience_marks_list[i] = 'n'  # Default to 'n' if no initial annotation found

                    # Loop through alignments for all summaries
                    for summary_idx, summary_alignment in enumerate(alignments):
                        for doc_alignment in summary_alignment:
                            found = False
                            for mention, _, _ in doc_alignment:
                                if token.lower() in mention.split():
                                    for i in range(len(salience_marks_list)):
                                        salience_marks_list[i] += 's'
                                    found = True
                                    break
                            if not found:
                                for i in range(len(salience_marks_list)):
                                    salience_marks_list[i] += 'n'

                    # Ensure each part has exactly 5 annotations and keep the original bracketed number
                    for i in range(len(salience_marks_list)):
                        bracket_idx = new_salience_parts[i].find('[')
                        if bracket_idx != -1:
                            salience_marks_list[i] = salience_marks_list[i][:5] + new_salience_parts[i][bracket_idx:]
                        else:
                            salience_marks_list[i] = salience_marks_list[i][:5]

                    # Join all salience marks to form the new salience column value
                    salience_marks_str = '|'.join(salience_marks_list)
                    columns[5] = salience_marks_str

                    outfile.write('\t'.join(columns) + '\n')
