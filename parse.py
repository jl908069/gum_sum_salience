import stanza
from nltk.tree import Tree

stanza.download('en')  # Download English models
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')


def extract_noun_phrases(tree):
    noun_phrases = []
    for subtree in tree.subtrees():
        # Find NP or ADJP modifiers headed by noun (with NN but without JJ)
        pos_tags = [tok[1][:2] for tok in subtree.pos()]
        if subtree.label() in ['NP','NML'] or (subtree.label() == 'ADJP' and 'NN' in pos_tags and not any([t.startswith('J') for t in pos_tags])):
            leaves = subtree.leaves()
            # Filter out pronouns: should think of better ways
            if not all([tag in ["DT","PRP","PRP$"] for tag in pos_tags]):
                noun_phrases.append(' '.join(leaves))
    return noun_phrases


def parse_summaries(summaries):
    all_noun_phrases = []
    for summary_list in summaries:
        summary_noun_phrases = []
        for summary in summary_list:
            doc = nlp(summary)
            summary_noun_phrases.append([extract_noun_phrases(Tree.fromstring(str(sent.constituency))) for sent in doc.sentences])
        all_noun_phrases.append(summary_noun_phrases)
    return all_noun_phrases
