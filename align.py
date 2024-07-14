def align(doc_text, summary_text, mention_text, component="LLM"):
    if component == "LLM":
        return align_llm(doc_text, summary_text, mention_text)
    elif component == "string_match":
        return align_string_match(doc_text, mention_text)
    elif component == "coref_system":
        return align_coref_system(doc_text, summary_text, mention_text)
    else:
        raise ValueError(f"Unknown alignment component: {component}")

def align_llm(doc_text, summary_text, mention_text):
    # Implementation for alignment using LLM (Huggingface API)
    pass

def align_string_match(doc_text, mention_text):
    # Implementation for alignment using string match
    pass

def align_coref_system(doc_text, summary_text, mention_text):
    # Implementation for alignment using a coreference resolution system
    pass