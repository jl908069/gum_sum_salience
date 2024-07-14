import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def get_summary(doc_texts, model_name="flan-t5-xl", n=4):
    # Load the tokenizer and model from Huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Initialize the summarization pipeline
    summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

    summaries = []
    for doc_text in doc_texts:
        doc_summaries = []
        for _ in range(n):
            prompt = f"Summarize the following article in 1 sentence. Make sure your summary is one sentence long and may not exceed 380 characters.\n\n{doc_text}"
            summary = summarizer(prompt, max_length=380, num_return_sequences=1)
            doc_summaries.append(summary[0]['generated_text'].strip())
        summaries.append(doc_summaries)
    return summaries

def read_documents_from_excel(file_path):
    df = pd.read_excel(file_path, sheet_name='full')
    doc_texts = df['fulltext'].tolist()
    doc_ids = df['doc_id'].tolist()
    return doc_ids, doc_texts

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate summaries for GUM documents")
    parser.add_argument("file_path", type=str, help="Path to the Excel file containing GUM documents")
    parser.add_argument("--model_name", type=str, default="flan-t5-xl", help="Huggingface model name to use for summarization")
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate per document")

    args = parser.parse_args()

    doc_ids, doc_texts = read_documents_from_excel(args.file_path)
    summaries = get_summary(doc_texts, model_name=args.model_name, n=args.n_summaries)

    for doc_id, doc_summaries in zip(doc_ids, summaries):
        print(f"Document ID: {doc_id}")
        for i, summary in enumerate(doc_summaries, 1):
            print(f"Summary {i}: {summary}")
        print()