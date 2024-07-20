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
    os.makedirs(output_folder, exist_ok=True)

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

def add_anno_to_tsv(input_dir, output_dir, alignments):
    """
    Process TSV files to modify salience columns and mark tokens based on alignments.

    Args:
        input_dir (str): Directory containing input TSV files.
        output_dir (str): Directory to save output TSV files.
        alignments (list): List of alignments for each summary.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".tsv"):
            input_file = os.path.join(input_dir, filename)
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
                        if part.startswith('sal'):
                            new_salience_parts.append('s')
                        elif part.startswith('nonsal'):
                            new_salience_parts.append('n')
                        else:
                            new_salience_parts.append(part)

                    # Initialize the salience marks for each token
                    salience_marks = new_salience_parts[:1]  # Keep the first summary annotation as it is

                    # Check alignments and mark tokens
                    for summary_idx, summary_alignment in enumerate(alignments):
                        found = False
                        for doc_alignment in summary_alignment:
                            for mention, _, _ in doc_alignment:
                                if token.lower() in mention.split():
                                    salience_marks.append('s')
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            salience_marks.append('n')

                    # Join all salience marks to form the new salience column value
                    columns[5] = ''.join(salience_marks)
                    
                    outfile.write('\t'.join(columns) + '\n')
