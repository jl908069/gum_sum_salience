import xml.etree.ElementTree as ET
import os
import re
from get_summary import get_summary
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, help="Path to the XML files")
    parser.add_argument("--output_xml", , type=str, help="Path to the output XML file")
    parser.add_argument("--file_path", type=str, default="gumsum.xlsx", help="Path to the Excel file containing GUM documents")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="Huggingface model name to use for summarization")
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate per document")

    args = parser.parse_args()

    # Read documents from the Excel file
    doc_ids, doc_texts = read_documents_from_excel(args.file_path)

    # Get summaries
    summaries = get_summary(doc_texts, doc_ids, model_name=args.model_name, n=args.n_summaries)

    add_summaries_to_xml(args.data_folder, summaries, args.output_xml)
