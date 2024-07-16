import stanza

stanza.download('en')  # Download English models
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

def extract_noun_phrases(tree):
    noun_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            leaves = subtree.leaves()
            # Filter out pronouns: should think of better ways
            if all(not word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 
                                        'you', 'your', 'yours', 'he', 'him', 'his', 'she', 
                                        'her', 'hers', 'it', 'its', 'they', 'them', 'their', 
                                        'theirs', 'this', 'that', 'these', 'those'] for word in leaves):
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
        flattened_data = [item for sublist in all_noun_phrases for subsublist in sublist for item in subsublist] # a list of lists
    return flattened_data
