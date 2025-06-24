import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf

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
    
    command = f"python calc_additional_features.py {input_path}"
    if output_dir:
        command += f" -out_dir {output_dir}"
    
    print(f"\nProcessing input: {input_path}")
    run_command(command)

def load_preprocessors(encoder_path, scaler_path):
    """
    Load the OneHotEncoder and StandardScaler from pickle files
    
    Parameters:
    -----------
    encoder_path : str
        Path to the saved OneHotEncoder pickle file
    scaler_path : str
        Path to the saved StandardScaler pickle file
        
    Returns:
    --------
    tuple
        (OneHotEncoder, StandardScaler)
    """
    try:
        encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        return encoder, scaler
    except Exception as e:
        print(f"Error loading preprocessors: {e}")
        sys.exit(1)

def preprocess_data(input_path, drop_columns, categorical_cols, continuous_cols, 
                   encoder_path="onehot_encoder.pkl", scaler_path="scaler.pkl"):
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
        List of continuous columns to scale
    encoder_path : str
        Path to the saved OneHotEncoder pickle file
    scaler_path : str
        Path to the saved StandardScaler pickle file
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    print("\nStarting preprocessing...")
    
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
    print(f"Combined dataframe shape: {df.shape}")
    
    # Drop specified columns
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    print(f"Shape after dropping columns: {df.shape}")
    
    # Load preprocessors
    encoder, scaler = load_preprocessors(encoder_path, scaler_path)
    
    # One-hot encode categorical columns
    existing_cat_cols = [col for col in categorical_cols if col in df.columns]
    if existing_cat_cols:
        # Transform categorical columns
        cat_encoded = encoder.transform(df[existing_cat_cols])
        
        # Get feature names from encoder
        feature_names = encoder.get_feature_names_out(existing_cat_cols)
        
        # Create dataframe with encoded features
        cat_encoded_df = pd.DataFrame(
            cat_encoded.toarray(),
            columns=feature_names,
            index=df.index
        )
        
        # Drop original categorical columns and add encoded ones
        df = df.drop(columns=existing_cat_cols)
        df = pd.concat([df, cat_encoded_df], axis=1)
        
        print(f"Shape after one-hot encoding: {df.shape}")
    
    # Scale continuous columns
    existing_cont_cols = [col for col in continuous_cols if col in df.columns]
    if existing_cont_cols:
        # Transform continuous columns
        scaled_features = scaler.transform(df[existing_cont_cols])
        
        # Replace original columns with scaled values
        df[existing_cont_cols] = scaled_features
        
        print(f"Applied scaling to {len(existing_cont_cols)} continuous columns")
    
    return df


def model_prediction(preprocessed_df, model_name):
    """
    Load the specified model and run predictions on preprocessed data

    Parameters:
    -----------
    preprocessed_df : pd.DataFrame
        The preprocessed DataFrame output from preprocess_data()
    model_name : str
        Name of the model file (without path) to load from 'models' directory

    Returns:
    --------
    numpy.ndarray
        Model predictions
    """
    try:
        # Construct model path
        model_path = os.path.join('models', model_name)

        # Load the model
        print(f"\nLoading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Make predictions
        print("Running predictions...")
        predictions = model.predict(preprocessed_df)

        return predictions

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
    drop_columns = [
        # Add columns to drop here
        # Example: 'column1', 'column2'
    ]
    
    categorical_cols = [
        # Add categorical columns here
        # Example: 'category1', 'category2'
    ]
    
    continuous_cols = [
        # Add continuous columns here
        # Example: 'numeric1', 'numeric2'
    ]

    # Specify model name
    model_name = 'model.h5'  # Update this with your actual model filename

    # First run calc_additional_features.py
    process_input(input_path, output_dir)
    
    # Run preprocessing on the output
    output_path = output_dir if output_dir else '.'
    if os.path.isfile(os.path.join(output_path, 'features_ring_extended.zip')):
        preprocessed_df = preprocess_data(
            os.path.join(output_path, 'features_ring_extended.zip'),
            drop_columns,
            categorical_cols,
            continuous_cols
        )
        
        # Save preprocessed data
        preprocessed_path = os.path.join(output_path, 'preprocessed_features.tsv')
        preprocessed_df.to_csv(preprocessed_path, sep='\t', index=False)
        print(f"\nPreprocessed data saved to: {preprocessed_path}")

        # Run model predictions
        predictions = model_prediction(preprocessed_df, model_name)

        # Save predictions (basic format for now)
        predictions_path = os.path.join(output_path, 'predictions.csv')
        pd.DataFrame(predictions).to_csv(predictions_path, index=False)

# TODO the model name, standard scaler, OHE, categorical columns, continuos columns, dropped columns are all model/dataset
#  specific. We should have a way to pass these parameters to the script, or we implement a single model and hardcode them here.