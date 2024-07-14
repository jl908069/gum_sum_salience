import spacy

nlp = spacy.load("en_core_web_sm")

def parse(summaries):
    all_mentions = []
    for summary in summaries:
        doc = nlp(summary)
        mentions = [chunk.text for chunk in doc.noun_chunks if chunk.root.pos_ in {"NOUN", "PROPN"}]
        all_mentions.append(mentions)
    return all_mentions