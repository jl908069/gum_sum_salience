import pandas as pd
from random import choice, shuffle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig


examples = {'academic': [
                'This study shows that limited exposure to a second language (L2) after it is no longer being actively used generally causes attrition of L2 competence.',
                'This poster paper presents a plan to hold up to six Digital Humanities (DH) clinics, one day events which include lectures and hands-on training, to help Dutch librarians in the Netherlands and Belgium to provide researchers and students with services and follow literature and research in the area of DH.'],
            'bio': [
                "Some details of Lord Byron's early life including his education at Aberdeen Grammar School, Harrow and Trinity College, as well as his romantic involvements and friendships with Mary Chaworth, John FitzGibbon and John Edleston.",
                'Joshua Norton was an eccentric resident of San Francisco who proclaimed himself Emperor of the United States in the second half of the 19th Century, and came to be known as Emperor Norton.'],
            'conversation': [
                "After being grounded for staying out at night, Sabrina fights with her mother, who does not permit her to join the volleyball team and later talks to her partner about cleaning the house's insulation to prevent mice from spreading an ongoing Hantavirus epidemic.",
                'Two people are playing a strategy game online involving cards and attacking countries, while discussing dinner plans.'],
            'court': [
                'General Elizabeth B. Prelogar, representing the Biden administration before the Chief Justice and the Court in a challenge by Nebraska and other states, argues that the HEROES Act authorizes Secretary Cardona to suspend payment obligations for federal student loans since it expressly allows waiver or modification of any Title IV provision in a national emergency, like COVID-19.'],
            'essay': [
                'Psychiatrist Arash Javanbakht suggests that to feel happier, humans must live the life they evolved to live, which includes physical activity, avoiding high-calorie sugary foods that were not available to our ancestors, changing our behavior around sleep by avoiding late caffeine and screens, and exposing ourselves to a healthy dose of excitement and a bit of fear.'],
            'fiction': [
                'A thirteen year old girl goes to mass with her father to take communion on a rainy day in March, then speculates who or what might be making two frightening noises she hears on her way back home as they pass through her school gym to get out of the rain.',
                'A protagonist recounts the day when his father, a patriotic islander, returned from a long journey and unexpectedly brought home a foreign Olondrian tutor from Bain named Master Lunre.'],
            'interview': [
                "Wikinews interviews several meteorologists about the prognosis for Cyclone Phailin as it approaches land in the Bay of Bengal at 190 km/h, and the Indian government's preparedness for the storm given past deaths caused by cyclones in the area.",
                'In an interview with Wikinews, Mario J. Lucero and Isabel Ruiz, who run a website called Heaven Sent Gaming, talk about how they met in school, the people involved in their project and the motivation for their work.'],
            'letter': [
                "On August 19, 1975, Bill writes a letter to Hannah, where he describes his stay at his friend's villa in Mallorca, details a conversation about war and politics with a Spanish prince at the villa, compares financial life in New York to London, updates her on his illness, tells her the books he is reading and editing, and asks how she is and how her manuscript has progressed."],
            'news': [
                'Indian Australian couple Thomas and Manju Sam are being prosecuted in Australia for manslaughter by gross negligence in the death of their nine-month old daughter Gloria, whose severe eczema they refused to treat using conventional medicing, instead opting for homeopathic treatments common in India, which are known to be ineffective.',
                'A 2006 Australian study has shown that almost half of all Australian children suffer from mild iodine deficiency, which can cause health problems especially for children and pregnant women, and is probably due to the replacement of iodine-containing sanitisers in the dairy industry and the lack of iodized salt in Australia.'],
            'podcast': [
                "On the show Beyond the Mat, Dave and Alex discuss wrestling news in the week of WrestleMania, Raw and Smackdown, including being glad that their favorite wrestler the Undertaker has finally retired despite people's objections and 90s nostalgia, and Alex recounts drinking tequila on his birthday and discusses the new DLC for King of Fighters XIV, which includes Rock Howard."],
            'reddit': [
                'In a post answering the question how and to whom countries can be in debt, the author explains how money is a form of debt whose value depends on the amount of money in circulation, and how fiat currency developed as a substitute for gold reserves which formerly backed bank debt certificates.',
                'Some Reddit forum users discuss whether humans are the only species which practices birth control to prevent reproduction, leading to a discussion of whether or not pandas are poor at reproducing, and some other animals which may become less reproductive when food is scarce, such as rats and rabbits.'],
            'speech': [
                'In a speech in the US Congress, a Democratic member of Congress accuses Republican Senators of failing to fulfill their oath to conduct an impartial trial in the impeachment of President Donald Trump for abuse of power and attempts to solicit foreign interference in the 2020 elections.',
                'In his inaugural address, US President Ronald Reagan praises the peaceful transition of power to a new Presidency and lays out his plans to reduce the role of government and spending in order to revive the economy and combat inflation and unemployment.'],
            'textbook': [
                "This section of a textbook explains and exemplifies different types of government, including representative democracy as in the United States, direct democracy as in ancient Athens, monarchy as in Saudi Arabia, oligarchy as in Cuba's reigning Communist Party, and totalitarianism as in North Korea.",
                'This excerpt explains three reasons why specialization of labor increases production: it allows workers to specialize in work they have a talent for, it allows them to improve particularly in certain tasks, and it allows businesses to take advantage of economies of scale, such as by setting up assembly lines to lower production costs.'],
            'vlog': [
                "In a video, Katie tells about her vacation to Portland, Oregon, and gives her top 4 recommendations for the region: Crater Lake, shopping at Powell's indie bookstore which sells used and new books, hiking in Forest Park and visiting the Rose Garden and the Japanese Gardens.",
                'A radiology resident vlogging on YouTube tells about his week on general nuclear medicine rotation, where he did three lymphoscintigraphies and some ultrasounds, his plans to work out after he gets off early from work, and taking Dutch cough drops to treat his sore throat ahead of a big trip he is planning the following weekend.'],
            'voyage': [
                'This article presents Athens, an ancient city and capital of modern Greece with a metropolitan population of 3.7 million, which hosted the 2004 Olympic games and features archeological sites and restored neoclassical as well as post-modern buildings, and is best visited in spring and late autumn.',
                'This overview of the history of Coron, a fishing town on the island of Busuanga in the Philippines, tells about the local people (Tagbanuas), the shift from farming to mining, fishing, and more recently tourism, and attractions such as stunning lakes, snorkeling, and wreck diving to see around ten Japanese ships sunk by the US Navy in World War II.'],
            'whow': [
                "This section from a guide on how to tell a joke advises practicing but not memorizing jokes, knowing your audience to pick appropriate jokes, choosing material from your life, online or repurposing jokes you've heard, using a realistic but exaggerated setup your audience can relate to, followed by a surprising punchline and possibly an additional 'topper' punchline.",
                "This guide to washing overalls suggests washing them with like clothing, avoiding clothes which can get twisted up with the straps, fastening straps to the bib with twist ties (also in the dryer), emptying pockets, moving the strap adjusters to make them last longer, using less detergent if washing overalls alone, and taking care plastic ties don't melt in the dryer."]}


def get_summary(doc_texts, doc_ids, model_name="google/flan-t5-base", n=4, dummy_mode=False):
    global examples

    if dummy_mode:  # Use for fast prototyping, returns cached summaries
        return {"GUM_academic_art":['Developing a pilot project of eye-tracking methodologies in the study of Zurbarán’s unique collection of 17th Century Josefinés and his Sons:', 'Eye-tracking, in first-phase research for exploring how an audience can be influenced to view and interpret the oeuvre of art in a visual, narrative way', 'We report upon the novel insights eye-tracking techniques have provided into the unconscious processes of viewing the unique collection of 17th Century Zurbarán paintings.', 'Aesthetic Appreciation of Paintings: Insights from Eye-Tracking'], "GUM_academic_census":['Providing an accurate and reliable survey of and measurement of the scientific workforce.', 'A web crawler to collect census data for computer science', 'A web crawler based on an existing database, based on real data generated for all academic fields.', 'A simple and efficient system that collects the information required to make a full census of computing fieldes.']}

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False
    )

    # Load the tokenizer and model from Huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quantization_config)

    all_summaries = {}
    for i, doc_text in enumerate(doc_texts):
        doc_id = doc_ids[i]
        genre = doc_id.split("_")[1]
        example = choice(examples[genre])
        prompt = f"Summarize the following article in 1 sentence. Make sure your summary is one sentence long and may not exceed 380 characters. Example of summary style: {example}\n\n{doc_text}\n\nSummary:"
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to(0)
        out = model.generate(**input_ids, max_new_tokens=80, num_return_sequences=n, do_sample=True)
        doc_summaries = tokenizer.batch_decode(out, skip_special_tokens=True)
        all_summaries[doc_id] = doc_summaries

    return all_summaries

def read_documents_from_excel(file_path):
    df = pd.read_excel(file_path, sheet_name='train') #default to 'train'
    doc_texts = df['fulltext'].tolist()
    doc_ids = df['doc_id'].tolist()
    return doc_ids, doc_texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate summaries for GUM documents")
    parser.add_argument("--file_path", type=str, default="gumsum.xlsx", help="Path to the Excel file containing GUM documents")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="Huggingface model name to use for summarization")
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate per document")

    args = parser.parse_args()

    doc_ids, doc_texts = read_documents_from_excel(args.file_path)
    docs = list(zip(doc_texts, doc_ids))

    # Sample just a few docs to test - comment this out to use all documents
    shuffle(docs)
    doc_texts, doc_ids = zip(*docs)
    doc_texts = doc_texts[:2]

    summaries = get_summary(doc_texts, doc_ids, model_name=args.model_name, n=args.n_summaries)

    for doc_id, doc_summaries in summaries.items():
        print(f"Document ID: {doc_id}\n")
        for i, summary in enumerate(doc_summaries, 1):
            print(f"Summary {i}: {summary}")
            print()
