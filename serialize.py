import xml.etree.ElementTree as ET

def serialize(tsv_path, xml_path, alignments, summaries):
    with open(tsv_path, 'w', encoding='utf-8') as tsv_file:
        for alignment in alignments:
            salience_annotations = "".join(['s' if a else 'n' for a in alignment])
            tsv_file.write(f"{salience_annotations}\n")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for i, summary in enumerate(summaries):
        summary_elem = ET.SubElement(root, f"summary{i+1}")
        summary_elem.text = summary
    
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)