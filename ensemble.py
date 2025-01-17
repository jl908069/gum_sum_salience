import os
import json
import pandas as pd
import argparse
from score import get_sal_tsv
from align import get_entities_from_gold_tsv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import defaultdict
from serialize import add_anno_to_tsv

def process_alignment_data(data_folder, partition, alignment_components, model_name):
    # Process model_name to ignore anything before "/", if any
    model_name = model_name.split("/")[-1]

    # Step 1: Get Gold Entities
    gold_entities_raw = get_entities_from_gold_tsv(os.path.join(data_folder, 'input', 'tsv', partition))
    gold_entities = [(tup[0].lower(), doc_id, tup[-1]) for doc_id, doc in enumerate(gold_entities_raw) for tup in doc]  # Extract "entity", "doc_id", and last tuple element
    print('gold_entities:', gold_entities)
    # Create initial DataFrame with "Gold Entity" and document ID
    df = pd.DataFrame([(tup[0].lower(), tup[1]) for tup in gold_entities], columns=['Gold Entity', 'Doc ID'])

    # Add "Doc Name" based on filenames (strip out ".tsv")
    doc_names = [os.path.basename(file).replace('.tsv', '') for file in sorted(os.listdir(os.path.join(data_folder, 'input', 'tsv', partition))) if file.endswith('.tsv')]
    doc_name_map = {doc_id: doc_name for doc_id, doc_name in enumerate(doc_names)}
    df['Doc Name'] = df['Doc ID'].map(doc_name_map)

    # Derive "Genre" from "Doc Name"
    df['Genre'] = df['Doc Name'].str.split('_').str[1]

    # Extract "Ent_Type" and "Position" from the last tuple element
    df['Ent_Type'] = [tup[-1].split('[')[0].strip() for tup in gold_entities]
    df['Position'] = [tup[-1].split('[')[-1].strip(']') for tup in gold_entities]

    # Step 2: Process each alignment_component directory
    for component in alignment_components:
        component_dir = os.path.join(data_folder, 'output', 'alignment', component)
        if not os.path.exists(component_dir):
            print(f"Directory {component_dir} does not exist. Skipping...")
            continue

        # Determine file name based on model_name and partition
        if model_name == "gold":
            file = f"pred_gold_{partition}.json"
        else:
            file = f"pred_{model_name}.json"

        pred_column = f"Pred_{component}"
    
        # Initialize a column with 0
        df[pred_column] = 0

        file_path = os.path.join(component_dir, file)
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping...")
            continue

        with open(file_path, 'r') as f:
            pred_entities = json.load(f)

        # Flatten predictions with document IDs
        pred_with_doc_ids = [(entity, doc_id) for doc_id, doc in enumerate(pred_entities) for entity in doc]

        # Mark matches in the DataFrame
        for entity, doc_id in pred_with_doc_ids:
            df.loc[(df['Gold Entity'].str.lower() == entity.lower()) & (df['Doc ID'] == doc_id), pred_column] = 1

    # Step 3: Get Gold Labels
    gold_labels_raw = get_sal_tsv(os.path.join(data_folder, 'input', 'tsv', partition))
    gold_labels = [(tup[0], doc_id) for doc_id, doc in enumerate(gold_labels_raw) for tup in doc]  # Extract "entity" and "doc_id"

    # Initialize "Gold_label" column with 0
    df['Gold_label'] = 0

    # Mark matches in the DataFrame
    for entity, doc_id in gold_labels:
        df.loc[(df['Gold Entity'].str.lower() == entity.lower()) & (df['Doc ID'] == doc_id), 'Gold_label'] = 1

    # Drop "Doc ID" column to match the desired output
    df = df.drop(columns=['Doc ID'])

    # Create output directory if it doesn't exist
    output_dir = os.path.join(data_folder, 'output', 'ensemble', partition)
    os.makedirs(output_dir, exist_ok=True)

    # Save the DataFrame to a TSV file
    output_file = os.path.join(output_dir, f"{model_name}.tsv")
    df.to_csv(output_file, sep='\t', index=False)

    return df

def train_and_predict(data_folder, partition):
    # Load the dev data for training
    dev_file_path = os.path.join(data_folder, 'output', 'ensemble', 'graded_sal_meta_learner_dev.tsv')
    dev_df = pd.read_csv(dev_file_path, sep='\t')

    # Prepare training data
    X_train = dev_df.drop(columns=['Gold_label'])
    y_train = dev_df['Gold_label']

    # Load all test TSV files from the output directory
    output_dir = os.path.join(data_folder, 'output', 'ensemble', partition)
    tsv_files = [file for file in os.listdir(output_dir) if file.endswith('.tsv')]

    # Identify string and numeric columns
    string_columns = X_train.select_dtypes(include=['object']).columns
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), string_columns),
            ('scaler', StandardScaler(), numeric_columns)
        ]
    )

    # Logistic Regression Model
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced')

    # Create a pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', logreg)
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Print classification report on the training data
    y_train_pred = model_pipeline.predict(X_train)
    print("Classification Report on Training Data:")
    print(classification_report(y_train, y_train_pred))

    # Make predictions for each test TSV file
    for file in tsv_files:
        file_path = os.path.join(output_dir, file)
        test_df = pd.read_csv(file_path, sep='\t')

        # Prepare test data
        X_test = test_df.drop(columns=['Gold_label'])
        predictions = model_pipeline.predict(X_test)

        # Write predictions back to the file
        test_df['Predictions'] = predictions
        test_df.to_csv(file_path, sep='\t', index=False)

def count_gold_entity_appearances_grouped(tsv_file_paths, output_json_path):
    """
    Counts the appearance of each "Gold Entity" across multiple TSV files based on predictions,
    grouped by "Doc Name".

    :param tsv_file_paths: List of file paths to the 5 input TSV files.
    :param output_json_path: Path to save the output JSON file.
    """
    # Nested dictionary to group counts by "Doc Name" and "Gold Entity"
    grouped_counts = defaultdict(lambda: defaultdict(int))

    # Loop through each TSV file and process
    for file_path in tsv_file_paths:
        # Read the TSV file
        df = pd.read_csv(file_path, sep='\t')

        # Ensure required columns exist
        if 'Gold Entity' not in df.columns or 'Predictions' not in df.columns or 'Doc Name' not in df.columns:
            raise ValueError(f"File {file_path} is missing required columns.")

        # Process each row
        for _, row in df.iterrows():
            gold_entity = row['Gold Entity']
            doc_name = row['Doc Name']
            prediction = row['Predictions']

            # Only count predictions that are 1
            if prediction == 1:  # Assuming 1 indicates a valid prediction
                grouped_counts[doc_name][gold_entity] += 1

    # Convert the grouped counts to the desired nested list format
    output_data = [
        [
            [gold_entity, count]
            for gold_entity, count in gold_entities.items()
        ]
        for doc_name, gold_entities in grouped_counts.items()
    ]

    # Save the data to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    return output_data

def process_tsv_files(partition):
    """
    Process multiple TSV files and generate predictions based on the Gold_label and Predictions columns.
    
    Args:
        data_partition (str): The partition of data to process (e.g., "test", "train")
        
    Returns:
        list: A list of lists, where each inner list contains tuples of (entity, predictions) for one document.
    """
    base_path = f"./data/output/ensemble/{partition}"
    
    # Get all TSV files in the directory
    tsv_files = [f for f in os.listdir(base_path) if f.endswith('.tsv')]
    
    # Define the preferred order of filenames
    preferred_prefixes = ['gold', 'claude', 'gpt4o', 'Llama', 'Qwen']
    
    # Sort files based on preferred prefixes
    ordered_files = []
    remaining_files = tsv_files.copy()
    
    # First, add files that match preferred prefixes in order
    for prefix in preferred_prefixes:
        matching_files = [f for f in remaining_files if f.lower().startswith(prefix.lower())]
        if matching_files:
            # If multiple files match a prefix, sort them alphabetically
            matching_files.sort()
            ordered_files.extend(matching_files)
            # Remove matched files from remaining_files
            for f in matching_files:
                remaining_files.remove(f)
    
    # Add any remaining files in alphabetical order
    remaining_files.sort()
    ordered_files.extend(remaining_files)
    
    # Read all TSV files in the determined order
    dataframes = []
    for filename in ordered_files:
        file_path = os.path.join(base_path, filename)
        df = pd.read_csv(file_path, sep='\t')
        dataframes.append(df)
    
    # Create a dictionary to store results for each document
    doc_results = {}
    
    # Process each row in the first dataset (assumed to be gold standard)
    for idx, row in dataframes[0].iterrows():
        doc_name = row['Doc Name']
        gold_entity = row['Gold Entity']
        
        # Initialize predictions string
        predictions = ''
        
        # Add predictions from each dataframe
        for df in dataframes:
            # Check if the column name is 'Gold_label' or 'Predictions'
            pred_column = 'Gold_label' if 'Gold_label' in df.columns else 'Predictions'
            predictions += 's' if df.iloc[idx][pred_column] == 1 else 'n'
        
        # Create or append to document's list of tuples
        if doc_name not in doc_results:
            doc_results[doc_name] = []
        doc_results[doc_name].append((gold_entity, predictions))
    
    # Convert dictionary to list of lists, maintaining document grouping
    result = list(doc_results.values())
    # Only the non-gold ones
    res=[[(item[0], item[1][1:]) for item in inner_list] for inner_list in results]
    
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and analyze salient entities.")
    parser.add_argument("--data_folder", type=str, required=True, help="Base data folder path.")
    parser.add_argument("--partition", type=str, required=True, help="Partition name (e.g., dev, test).")
    parser.add_argument("--alignment_components", nargs='+', required=True, help="List of alignment components.")
    parser.add_argument("--max_docs", type=int, default=None, help="Maximum number of documents to processe (default: None = all; choose a small number to prototype)")
    parser.add_argument("--model_names", nargs='+', required=True, help="List of model names.")

    args = parser.parse_args()

    # Step 1: Process alignment data
    for model_name in args.model_names:
        process_alignment_data(args.data_folder, args.partition, args.alignment_components, model_name)

    # Step 2: Train and predict
    train_and_predict(args.data_folder, args.partition)

    # Step 3: Write the salience annotations into the tsv files
    #output_dir = os.path.join(args.data_folder, 'output', 'ensemble', args.partition)
    #tsv_file_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.tsv')]
    results = process_tsv_files(args.partition)
    print('Salience annotations:', results)
    # Write the final salience annotations to tsv files
    add_anno_to_tsv(data_folder=args.data_folder, model_predictions= results, partition=args.partition, max_docs=args.max_docs)
    #output_json_path = os.path.join(output_dir, "ens_sal_score.json")

    #count_gold_entity_appearances_grouped(tsv_file_paths, output_json_path)
