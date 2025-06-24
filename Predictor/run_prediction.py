import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import joblib
import xgboost as xgb

def run_command(command):
    """
    Run a shell command and capture output/errors
    """
    try:
        result = subprocess.run(
            command.split(),
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"\nError running command: {command}")
        print("Error output:")
        print(e.stderr)
        sys.exit(1)


def process_input(input_path, output_dir=None):
    """
    Process input through calc_additional_features.py
    """
    if output_dir is None:
        output_dir = '.'

    os.makedirs(output_dir, exist_ok=True)

    # Update path to calc_additional_features.py
    predictor_dir = os.path.join(os.getcwd(), 'Predictor')
    script_path = os.path.join(predictor_dir, 'calc_additional_features.py')
    command = f"python {script_path} {input_path}"
    if output_dir:
        command += f" -out_dir {output_dir}"

    print(f"\nProcessing input: {input_path}")
    run_command(command)


def load_encoder(encoder_path):
    """
    Load the OneHotEncoder from pickle file
    
    Parameters:
    -----------
    encoder_path : str
        Path to the saved OneHotEncoder pickle file
        
    Returns:
    --------
    OneHotEncoder
    """
    try:
        encoder = joblib.load(encoder_path)
        return encoder
    except Exception as e:
        print(f"Error loading encoder: {e}")
        sys.exit(1)

def preprocess_data(input_path, drop_columns, categorical_cols, continuous_cols, 
                   encoder_path="Predictor\onehot_encoder.pkl"):
    """
    Preprocess the data according to specified requirements
    
    Parameters:
    -----------
    input_path : str
        Path to input (TSV file, directory, or ZIP file)
    drop_columns : list
        List of columns to drop from the dataframe
    categorical_cols : list
        List of categorical columns to one-hot encode
    continuous_cols : list
        List of continuous columns
    encoder_path : str
        Path to the saved OneHotEncoder pickle file
    
    Returns:
    --------
    tuple
        (preprocessed DataFrame, encoder)
    """
    print("\nStarting preprocessing...")

    # Update encoder path
    if encoder_path is None:
        encoder_path = os.path.join(os.getcwd(), 'Predictor', 'onehot_encoder.pkl')
    
    # Load all TSV files into a single dataframe
    dfs = []
    
    if os.path.isdir(input_path):
        print(f"Reading TSV files from directory: {input_path}")
        tsv_files = [f for f in os.listdir(input_path) if f.endswith('.tsv')]
        for tsv_file in tsv_files:
            df = pd.read_csv(os.path.join(input_path, tsv_file), sep='\t')
            dfs.append(df)
            
    elif input_path.endswith('.zip'):
        print(f"Reading TSV files from ZIP archive: {input_path}")
        import zipfile
        with zipfile.ZipFile(input_path) as zip_ref:
            tsv_files = [f for f in zip_ref.namelist() if f.endswith('.tsv')]
            for tsv_file in tsv_files:
                df = pd.read_csv(zip_ref.open(tsv_file), sep='\t')
                dfs.append(df)
                
    elif input_path.endswith('.tsv'):
        print(f"Reading single TSV file: {input_path}")
        df = pd.read_csv(input_path, sep='\t')
        dfs.append(df)
    
    if not dfs:
        print("Error: No TSV files found in input")
        sys.exit(1)
    
    # Combine all dataframes
    df = pd.concat(dfs, axis=0, ignore_index=True)
    initial_shape = df.shape
    print(f"Combined dataframe shape: {initial_shape}")

    # Handle NaN values
    nan_count_before = df.isna().sum().sum()
    if nan_count_before > 0:
        print(f"\nFound {nan_count_before} NaN values before preprocessing")
        print("\nColumns with NaN values:")
        print(df.isna().sum()[df.isna().sum() > 0])

        # Drop rows with any NaN values
        df = df.dropna()
        print(f"\nShape after dropping NaN rows: {df.shape}")
        print(f"Dropped {initial_shape[0] - df.shape[0]} rows containing NaN values")

    # Drop specified columns
    if 'Interaction' in df.columns:
        df = df.drop(columns=['Interaction'])
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    print(f"Shape after dropping columns: {df.shape}")

    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Load encoder
    encoder = load_encoder(encoder_path)
    
    # One-hot encode categorical columns
    existing_cat_cols = [col for col in categorical_cols if col in df.columns]
    if existing_cat_cols:
        # Transform categorical columns
        cat_encoded = encoder.transform(df[existing_cat_cols])
        
        # Get feature names from encoder
        feature_names = encoder.get_feature_names_out(existing_cat_cols)

        # Create dataframe with encoded features - handle both sparse and dense matrices
        if hasattr(cat_encoded, 'toarray'):
            cat_encoded_data = cat_encoded.toarray()
        else:
            cat_encoded_data = cat_encoded

        cat_encoded_df = pd.DataFrame(
            cat_encoded_data,
            columns=feature_names,
            index=df.index
        )

        # Drop original categorical columns and add encoded ones
        df = df.drop(columns=existing_cat_cols)
        df = pd.concat([df, cat_encoded_df], axis=1)
        
        print(f"Shape after one-hot encoding: {df.shape}")
    
    return df, encoder

def load_all_models(model_dir):
    """
    Load all XGBoost models from the models directory

    Parameters:
    -----------
    model_dir : str
        Directory containing the model files

    Returns:
    --------
    dict
        Dictionary mapping interaction types to their models
    """
    interaction_types = ['HBOND', 'IONIC', 'PICATION', 'PIHBOND', 'PIPISTACK', 'SSBOND', 'VDW']
    interaction_models = {}

    try:
        for interaction in interaction_types:
            model_path = os.path.join(model_dir, f"{interaction}.joblib")
            print(f"Loading model: {model_path}")
            model = joblib.load(model_path)
            interaction_models[interaction] = model

        return interaction_models
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

def predict_interactions(df, models):
    """
    Make predictions using all interaction models

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed input features
    models : dict
        Dictionary of loaded models for each interaction type

    Returns:
    --------
    pd.DataFrame
        DataFrame with prediction columns added
    """
    try:
        predictions_df = df.copy()
        features = df.columns.tolist()

        for interaction, model in models.items():
            print(f"\nPredicting {interaction}...")
            label = model.predict(df[features])
            score = model.predict_proba(df[features])[:, 1]
            predictions_df[interaction] = label.astype(int)
            predictions_df[f'{interaction}_SCORE'] = score

        return predictions_df

    except Exception as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)

def process_output(predictions_df, encoder, df_restored):
    """
    Process the predictions output by reversing one-hot encoding and reformatting predictions

    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with raw predictions from predict_interactions()
    encoder : OneHotEncoder
        The fitted OneHotEncoder object used in preprocessing
    df_restored : pd.DataFrame
        DataFrame with original identifier columns to be restored

    Returns:
    --------
    pd.DataFrame
        Final processed DataFrame with formatted predictions
    """
    try:
        # Define categorical columns and interaction types
        categorical_cols = ['s_resn', 't_resn', 's_ss8', 't_ss8']
        interaction_types = ['HBOND', 'IONIC', 'PICATION', 'PIHBOND', 'PIPISTACK', 'SSBOND', 'VDW']

        # Step 1: Reverse one-hot encoding
        print("\nReversing one-hot encoding...")
        encoded_columns = encoder.get_feature_names_out(categorical_cols)
        df_encoded_part = predictions_df[encoded_columns]

        decoded_data = encoder.inverse_transform(df_encoded_part)
        decoded_df = pd.DataFrame(decoded_data, columns=categorical_cols)

        df_pred_dropped_encoded = predictions_df.drop(columns=encoded_columns)
        df_reversed_ohe = pd.concat([df_pred_dropped_encoded.reset_index(drop=True),
                                     decoded_df.reset_index(drop=True)], axis=1)

        # Step 2: Reformat predictions
        print("Reformatting predictions...")
        interaction_list = []
        score_list = []

        for index, row in df_reversed_ohe.iterrows():
            predicted_interactions = []
            predicted_scores = []
            for interaction in interaction_types:
                if row[interaction] == 1:
                    predicted_interactions.append(interaction)
                    # Round the score to 4 decimal places
                    predicted_scores.append(round(float(row[f'{interaction}_SCORE']), 4))

            interaction_list.append(predicted_interactions)
            score_list.append(predicted_scores)

        # Add interaction and score lists
        df_reversed_ohe['Interaction'] = interaction_list
        df_reversed_ohe['score'] = score_list

        # Drop temporary columns
        columns_to_drop = []
        for inter_type in interaction_types:
            columns_to_drop.extend([inter_type, f'{inter_type}_SCORE'])
        df_reversed_ohe = df_reversed_ohe.drop(columns=columns_to_drop)

        # Drop duplicate columns from OHE
        df_reversed_ohe = df_reversed_ohe.drop(columns=['s_resn', 't_resn'])

        # Restore original columns
        df_restored = df_restored.loc[df_reversed_ohe.index].reset_index(drop=True)
        df_final = pd.concat([df_restored, df_reversed_ohe], axis=1)

        print("Output processing complete.")
        return df_final

    except Exception as e:
        print(f"Error processing output: {e}")
        sys.exit(1)

def model_prediction(preprocessed_df, model_dir, df_restored, encoder):
    """
    Load models and run predictions on preprocessed data
    """
    try:
        print(f"\nLoading XGBoost models from: {model_dir}")
        interaction_models = load_all_models(model_dir)
        
        print("\nRunning predictions...")
        predictions_df = predict_interactions(preprocessed_df, interaction_models)
        
        final_df = process_output(predictions_df, encoder, df_restored)
        
        return final_df
        
    except Exception as e:
        print(f"Error in model prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python run_prediction.py <input_path> [output_dir]")
        print("input_path can be:")
        print("  - A single TSV file")
        print("  - A directory containing TSV files")
        print("  - A ZIP file containing TSV files")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else None
    
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' not found")
        sys.exit(1)
    
    # Define preprocessing parameters
    drop_columns = ['s_3di_letter', 't_3di_letter', 's_3di_state', 't_3di_state',
                    'pdb_id', 's_ch', 't_ch', 's_ins', 't_ins']

    categorical_cols = ['s_resn', 't_resn', 's_ss8', 't_ss8']
    
    continuous_cols = [
        's_resi', 's_rsa', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
        't_resi', 't_rsa', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5',
        'delta_rsa', 'delta_atchley_1', 'delta_atchley_2', 'delta_atchley_3', 'delta_atchley_4', 'delta_atchley_5',
        'ca_distance', 's_centroid_x', 's_centroid_y', 't_centroid_x', 't_centroid_y'
    ]

    # Process input
    process_input(input_path, output_dir)

    # Run preprocessing on the output
    output_path = output_dir if output_dir else '.'
    if os.path.isfile(os.path.join(output_path, 'features_ring_extended.zip')):
        # Update encoder path
        encoder_path = os.path.join(os.getcwd(), 'Predictor', 'onehot_encoder.pkl')

        preprocessed_df, encoder = preprocess_data(
            os.path.join(output_path, 'features_ring_extended.zip'),
            drop_columns,
            categorical_cols,
            continuous_cols,
            encoder_path=encoder_path
        )

        # Save preprocessed data
        preprocessed_path = os.path.join(output_path, 'preprocessed_features.tsv')
        preprocessed_df.to_csv(preprocessed_path, sep='\t', index=False)
        print(f"\nPreprocessed data saved to: {preprocessed_path}")
        
        # Restore original dataframe
        df = pd.read_csv(os.path.join(output_path, 'features_ring_extended.zip'), sep='\t')

        # Save original columns for later restoration
        restored_cols = ['s_ch', 's_resi', 's_ins', 's_resn',
                         't_ch', 't_resi', 't_ins', 't_resn']
        df_restored = df[restored_cols]

        # Update model directory path
        model_dir = os.path.join(os.getcwd(), 'Predictor', 'models', 'xgboost')

        # Run model predictions
        predictions = model_prediction(
            preprocessed_df,
            model_dir,
            df_restored,
            encoder
        )

        # Get the base input filename without extension
        input_filename = os.path.basename(input_path)
        pdb_id = input_filename.split('.')[0]  # removes .tsv extension

        # Save predictions
        predictions_path = os.path.join(output_path, f'{pdb_id}_prediction.tsv')
        pd.DataFrame(predictions).to_csv(predictions_path, sep='\t', index=False)
        print(f"\nPredictions saved to: {predictions_path}")