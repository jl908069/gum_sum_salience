import os, re, sys
from glob import glob
from random import choice, shuffle, seed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoModelForCausalLM
import pandas as pd
import xml.etree.ElementTree as ET
from openai import OpenAI
import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

client = OpenAI(api_key="<your_API_key>")
claude_client = anthropic.Anthropic(api_key="<your_API_key>")

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


def get_summary(doc_texts, doc_ids, data_folder, partition, model_name="google/flan-t5-xl", n=4, overwrite=False):
    global examples

    # Adjust the quantization config to enable double quantization and optimize memory
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,  # Enable double quantization to reduce VRAM usage
        bnb_8bit_quant_type="nf8"        # Can also try 'fp4' depending on memory/performance needs
    )

    # Extract the actual model name (strip the company/organization prefix)
    model_name_short = model_name.split("/")[-1]

    # Load the tokenizer and model only once for all documents
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Conditional loading with or without quantization
    if model_name == "google/flan-t5-xl":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quantization_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

    # Ensure output directory exists
    if not data_folder.endswith(os.sep):
        data_folder += os.sep
    summary_folder = data_folder + "output" + os.sep + "summaries" + os.sep + partition + os.sep + model_name_short + os.sep
    os.makedirs(summary_folder, exist_ok=True)

    all_summaries = {}
    cached_summaries = 0
    cached_summary_docs = 0
    written_summaries = 0
    written_summary_docs = 0

    # Process all documents in a single loop (no batches)
    for i, doc_text in enumerate(doc_texts):
        doc_id = doc_ids[i]
        doc_summaries = []

        # Check if summaries exist in the specified filename format and load them if overwrite is False
        summaries_exist = all(
            os.path.exists(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt") for j in range(n)
        )

        if summaries_exist and not overwrite:
            # Load existing summaries
            for j in range(n):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt", "r", encoding="utf-8") as f:
                    doc_summaries.append(f.read().strip())
                    cached_summaries += 1
            cached_summary_docs += 1
        else:
            # Generate prompt for the document
            genre = doc_id.split("_")[1]
            example = choice(examples[genre])
            prompt = f"Summarize the following article in 1 sentence. Example: {example}\n\n{doc_text}\n\nSummary:"

            # Tokenize input and calculate input tokens
            input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
            input_token_count = input_ids.input_ids.shape[-1]
          
            # Generate summaries
            out = model.generate(
                **input_ids, max_new_tokens=120, num_return_sequences=n, do_sample=True, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1
            )
            doc_summaries = tokenizer.batch_decode(out, skip_special_tokens=True)

            # Print output tokens for each summary
            for j, summary in enumerate(doc_summaries):
                output_token_count = len(tokenizer(summary).input_ids)
                print(f"Document {doc_id} - Output tokens (Summary {j + 1}): {output_token_count}")

            # Write summaries in the specified filename format
            for k, summary in enumerate(doc_summaries):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{k}.txt", "w", encoding="utf-8", newline="\n") as f:
                    f.write(summary)
                    written_summaries += 1
            written_summary_docs += 1

        all_summaries[doc_id] = doc_summaries

    if cached_summaries > 0:
        sys.stderr.write(f"Loaded {cached_summaries} cached summaries for {cached_summary_docs} documents.\n")
    if written_summaries > 0:
        sys.stderr.write(f"Wrote {written_summaries} new summaries for {written_summary_docs} documents.\n")

    return all_summaries

def get_summary_gpt4o(doc_texts, doc_ids, data_folder, partition, model_name="gpt4o", n=4, overwrite=False):
    # Extract the actual model name (strip the company/organization prefix)
    model_name_short = model_name.split("/")[-1]

    if not data_folder.endswith(os.sep):
        data_folder += os.sep
    summary_folder = data_folder + "output" + os.sep + "summaries" + os.sep + partition + os.sep + model_name_short + os.sep

    # Ensure the output directory exists
    os.makedirs(summary_folder, exist_ok=True)

    all_summaries = {}
    cached_summaries = 0
    cached_summary_docs = 0
    written_summaries = 0
    written_summary_docs = 0

    for i, doc_text in enumerate(doc_texts):
        doc_id = doc_ids[i]
        doc_summaries = []

        # Check if summaries exist in the specified filename format and load them if overwrite is False
        summaries_exist = all(
            os.path.exists(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt") for j in range(n)
        )
        
        if summaries_exist and not overwrite:
            # Load existing summaries
            for j in range(n):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt", "r", encoding="utf-8") as f:
                    doc_summaries.append(f.read().strip())
                    cached_summaries += 1
            cached_summary_docs += 1
        else:
            # Generate new summaries
            genre = doc_id.split("_")[1]
            example = choice(examples[genre])
            prompt = f"Summarize the following article in 1 sentence. Make sure your summary is one sentence long and may not exceed 380 characters. Example of summary style: {example}\n\n{doc_text}\n\nSummary:"
            
            # Call the GPT4o API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an assistant for generating one-sentence summaries."},
                    {"role": "user", "content": prompt}
                ]
            )         

            # Extract summaries from response
            choices = response.choices
            if len(choices) < n:
                print(f"Warning: Only {len(choices)} summaries returned for document {doc_id}, expected {n}.")
                while len(choices) < n:
                    # Generate additional summaries if needed
                    additional_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an assistant for generating one-sentence summaries."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    choices.extend(additional_response.choices)
            doc_summaries = [choice.message.content.strip() for choice in choices[:n]]

            # Write summaries in the specified filename format
            for k, summary in enumerate(doc_summaries):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{k}.txt", "w", encoding="utf-8", newline="\n") as f:
                    f.write(summary)
                    written_summaries += 1
            written_summary_docs += 1
            
        all_summaries[doc_id] = doc_summaries

    if cached_summaries > 0:
        sys.stderr.write(f"Loaded {cached_summaries} cached summaries for {cached_summary_docs} documents.\n")
    if written_summaries > 0:
        sys.stderr.write(f"Wrote {written_summaries} new summaries for {written_summary_docs} documents.\n")

    return all_summaries

def get_summary_claude35(doc_texts, doc_ids, data_folder, partition, model_name="claude-3-5-sonnet-20241022", n=4, overwrite=False):

    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "
    # Extract the actual model name
    model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name

    # Ensure the output directory exists
    if not data_folder.endswith(os.sep):
        data_folder += os.sep
    summary_folder = data_folder + "output" + os.sep + "summaries" + os.sep + partition + os.sep + model_name_short + os.sep
    os.makedirs(summary_folder, exist_ok=True)

    all_summaries = {}
    cached_summaries = 0
    cached_summary_docs = 0
    written_summaries = 0
    written_summary_docs = 0

    for i, doc_text in enumerate(doc_texts):
        doc_id = doc_ids[i]
        doc_summaries = []

        # Check if summaries exist in the specified filename format and load them if overwrite is False
        summaries_exist = all(
            os.path.exists(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt") for j in range(n)
        )
        
        if summaries_exist and not overwrite:
            # Load existing summaries
            for j in range(n):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt", "r", encoding="utf-8") as f:
                    doc_summaries.append(f.read().strip())
                    cached_summaries += 1
            cached_summary_docs += 1
        else:
            # Generate prompt for the document
            genre = doc_id.split("_")[1]
            example = choice(examples[genre])
            prompt = f"Summarize the following article in 1 sentence. Make sure your summary is one sentence long and does not exceed 380 characters. Example of summary style: {example}\n\n{doc_text}\n\nSummary:"

            # Call the Claude API to generate multiple summaries if needed
            for _ in range(n):
                response = claude_client.messages.create(
                    model=model_name,
                    max_tokens=120,  # Adjust as needed
                    system="You are a summarization assistant generating concise one-sentence summaries.",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "Here is the summary:"}
                    ]
                )

                summary = ''.join([block.text for block in response.content])
                doc_summaries.append(summary)

            # Write summaries in the specified filename format
            for k, summary in enumerate(doc_summaries):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{k}.txt", "w", encoding="utf-8", newline="\n") as f:
                    f.write(summary)
                    written_summaries += 1
            written_summary_docs += 1

        all_summaries[doc_id] = doc_summaries

    if cached_summaries > 0:
        sys.stderr.write(f"Loaded {cached_summaries} cached summaries for {cached_summary_docs} documents.\n")
    if written_summaries > 0:
        sys.stderr.write(f"Wrote {written_summaries} new summaries for {written_summary_docs} documents.\n")

    return all_summaries

def extract_gold_summaries_from_xml(directory):
    # List to store extracted data
    data = []
    
    # Iterate over all XML files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            filepath = os.path.join(directory, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Extract doc_id (from the 'id' attribute of the root element)
            doc_id = root.get('id')
            
            # Extract the summary text
            summary = root.get('summary', '')
            
            # Append the data to the list
            data.append({'doc_id': doc_id, 'summary': summary})
            
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    result_dict = df.groupby('doc_id')['summary'].apply(list).to_dict()
    
    return result_dict

def extract_text_speaker_from_xml(directory):
    # List to store the concatenated text from each document
    all_documents = []
    
    # Get and sort all XML files in the directory
    xml_files = sorted([f for f in os.listdir(directory) if f.endswith('.xml')])

    # Iterate through sorted XML files
    for filename in xml_files:
        filepath = os.path.join(directory, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()

        # List to store the text from the current document
        document_text = []

        # Track the current speaker and whether we're inside an <sp> tag
        current_speaker = ''
        inside_sp = False
        
        # Loop through all elements in the XML
        for elem in root.iter():
            # Check if the element is an <sp> tag
            if elem.tag == 'sp':
                inside_sp = True
                current_speaker = elem.get('who', '').lstrip('#')
                
                # Loop through all <s> tags within this <sp>
                for s in elem.iter('s'):
                    tokens = []
                    for token_elem in s.itertext():
                        split_elem = token_elem.strip().split('\n')
                        for line in split_elem:
                            token_parts = line.split('\t')
                            if token_parts:
                                tokens.append(token_parts[0])  # Extract only the first part (the token)
                    sentence = ' '.join(tokens)
                    if current_speaker:
                        sentence = f"{current_speaker}: {sentence}"
                    document_text.append(sentence)
                
                inside_sp = False  # Reset after processing <sp> block
            
            # Process <s> tags outside of any <sp> tag
            elif elem.tag == 's' and not inside_sp:
                tokens = []
                for token_elem in elem.itertext():
                    split_elem = token_elem.strip().split('\n')
                    for line in split_elem:
                        token_parts = line.split('\t')
                        if token_parts:
                            tokens.append(token_parts[0])  # Extract only the first part (the token)
                sentence = ' '.join(tokens)
                document_text.append(sentence)
        
        # Concatenate all sentences from the current document into a single string
        full_text = ' '.join(document_text)
        
        # Add the full text of the current document to the list
        all_documents.append(full_text)
    
    return all_documents

def read_documents(data_folder, partition):
    # Read documents directly from tsv/ folder, since we have it as input
    files = sorted(glob(data_folder + os.sep + 'input' + os.sep + 'tsv' + os.sep + partition + os.sep + '*.tsv'))
    doc_ids = []
    doc_texts = []
    for file_ in files:
        docname = os.path.basename(file_).split(".")[0]
        sents = re.findall(r'#Text=([^\n]+)', open(file_, "r", encoding="utf-8").read())
        text = " ".join([s.strip() for s in sents])
        doc_texts.append(text)
        doc_ids.append(docname)

    return doc_ids, doc_texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate summaries for GUM documents")
    parser.add_argument("--data_folder", default="data", help="Path to data folder")
    parser.add_argument("--model_name", default="google/flan-t5-xl", choices=["gpt4o", "claude-3-5-sonnet-20241022", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"], help="Model name to use for summarization")
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate per document")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached summaries (default: False)")
    parser.add_argument("--partition", default="train", choices=["test", "dev", "train"], help="Data partition to use for generating and storing summaries")

    args = parser.parse_args()

    doc_ids, doc_texts = read_documents(args.data_folder, args.partition)
    docs = list(zip(doc_texts, doc_ids))

    # Sample just a few docs to test - comment this out to use all documents
    # seed(42) 
    # shuffle(docs)
    # doc_texts, doc_ids = zip(*docs)
    # doc_texts = doc_texts[:12]
    if args.model_name=="gpt4o":
        summaries =get_summary_gpt4o(doc_texts, doc_ids, args.data_folder, args.partition, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    elif args.model_name=="claude-3-5-sonnet-20241022":
        summaries =get_summary_claude35(doc_texts, doc_ids, args.data_folder, args.partition, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    else:
        summaries = get_summary(doc_texts, doc_ids, args.data_folder, args.partition, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)

    for doc_id, doc_summaries in summaries.items():
        print(f"Document ID: {doc_id}\n")
        for i, summary in enumerate(doc_summaries, 1):
            print(f"Summary {i}: {summary}")
            print()
