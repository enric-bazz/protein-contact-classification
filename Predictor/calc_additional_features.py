import pandas as pd
import numpy as np
from Bio.PDB import PDBList, PDBParser
from pathlib import Path
import argparse
import logging
import os
import zipfile
import tempfile


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Input path: can be either a TSV file, directory containing TSV files, or a ZIP file containing TSV files')
    parser.add_argument('-out_dir', help='Output directory', default='.')
    return parser.parse_args()


def read_input_files(input_path):
    """
    Read input files from either a single TSV, a directory of TSVs, or a ZIP file.
    Returns a dictionary mapping PDB IDs to their corresponding DataFrames.
    """
    pdb_dataframes = {}
    
    if zipfile.is_zipfile(input_path):
        logging.info(f"Reading TSV files from ZIP archive: {input_path}")
        with zipfile.ZipFile(input_path) as zip_ref:
            tsv_files = [f for f in zip_ref.namelist() if f.endswith('.tsv')]
            for tsv_file in tsv_files:
                pdb_id = Path(tsv_file).stem
                df = pd.read_csv(zip_ref.open(tsv_file), sep='\t')
                pdb_dataframes[pdb_id] = df
                
    elif os.path.isdir(input_path):
        logging.info(f"Reading TSV files from directory: {input_path}")
        tsv_files = [f for f in os.listdir(input_path) if f.endswith('.tsv')]
        for tsv_file in tsv_files:
            pdb_id = Path(tsv_file).stem
            df = pd.read_csv(os.path.join(input_path, tsv_file), sep='\t')
            pdb_dataframes[pdb_id] = df
            
    elif os.path.isfile(input_path) and input_path.endswith('.tsv'):
        logging.info(f"Reading single TSV file: {input_path}")
        df = pd.read_csv(input_path, sep='\t')
        # Group the data by PDB ID
        for pdb_id, group_df in df.groupby('pdb_id'):
            pdb_dataframes[pdb_id] = group_df
            
    else:
        raise ValueError(f"Invalid input path: {input_path}. Must be a TSV file, directory, or ZIP file.")
    
    logging.info(f"Found data for {len(pdb_dataframes)} PDB structures")
    return pdb_dataframes


def add_same_chain(df):
    """Add boolean feature indicating if residues are in the same chain."""
    df['same_chain'] = (df['s_ch'] == df['t_ch']).astype(int)
    return df


def add_delta_rsa(df):
    """Add absolute difference in relative solvent accessibility."""
    df['delta_rsa'] = abs(df['s_rsa'] - df['t_rsa'])
    return df


def add_delta_atchley(df):
    """Add absolute differences between Atchley factors."""
    for i in range(1, 6):
        df[f'delta_atchley_{i}'] = (df[f's_a{i}'] - df[f't_a{i}']).abs()
    return df


def get_ca_coords(pdb_file):
    """Extract CA coordinates from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    ca_coords = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca = residue['CA'].get_coord()
                    res_id = residue.get_id()
                    key = (chain.id, res_id[1], res_id[2].strip())
                    ca_coords[key] = ca
    return ca_coords


def calculate_ca_distance(row, ca_coords_dict):
    """Calculate distance between CA atoms of two residues."""
    s_key = (row['s_ch'], row['s_resi'], str(row['s_ins']).strip())
    t_key = (row['t_ch'], row['t_resi'], str(row['t_ins']).strip())

    if s_key in ca_coords_dict and t_key in ca_coords_dict:
        dist = np.linalg.norm(ca_coords_dict[s_key] - ca_coords_dict[t_key])
        return dist
    return np.nan


def add_ca_distances(df, pdb_id):
    """Add CA distances for all residue pairs in a PDB structure."""
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id.lower(), pdir='pdb_files', file_format='pdb')
    
    if not os.path.exists(pdb_file):
        logging.warning(f"PDB file not found for {pdb_id}, skipping CA distance calculation")
        return df
    
    ca_coords = get_ca_coords(pdb_file)
    df['ca_distance'] = df.apply(lambda row: calculate_ca_distance(row, ca_coords), axis=1)
    return df


def add_centroid_coordinates(df, states_file):
    """Add centroid coordinates for 3di states."""
    try:
        centroids = np.loadtxt(states_file)
        
        def map_centroid(coord_index, axis):
            try:
                return centroids[int(coord_index), axis]
            except (ValueError, TypeError, IndexError):
                return np.nan

        df['s_centroid_x'] = df['s_3di_state'].apply(lambda i: map_centroid(i, 0))
        df['s_centroid_y'] = df['s_3di_state'].apply(lambda i: map_centroid(i, 1))
        df['t_centroid_x'] = df['t_3di_state'].apply(lambda i: map_centroid(i, 0))
        df['t_centroid_y'] = df['t_3di_state'].apply(lambda i: map_centroid(i, 1))
        
    except Exception as e:
        logging.warning(f"Error loading centroids file: {e}")
    
    return df


def save_to_zip(processed_dfs, output_dir):
    """
    Save all processed DataFrames to individual TSV files and pack them into a ZIP file.
    Also saves individual TSV files in the output directory.
    """
    zip_path = os.path.join(output_dir, 'features_ring_extended.zip')
    
    # Create a temporary directory to store TSV files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for pdb_id, df in processed_dfs.items():
                # Save individual TSV file
                tsv_filename = f"{pdb_id}.tsv"
                tsv_path = os.path.join(output_dir, tsv_filename)
                df.to_csv(tsv_path, sep='\t', index=False)
                logging.info(f"Saved {tsv_filename}")
                
                # Add to ZIP file
                zip_file.write(tsv_path, tsv_filename)
    
    logging.info(f"Created ZIP archive: {zip_path}")
    return zip_path


if __name__ == '__main__':
    args = arg_parser()

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        level=logging.INFO
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Read input files
    try:
        pdb_dataframes = read_input_files(args.input_path)
    except Exception as e:
        logging.error(f"Error reading input files: {e}")
        exit(1)

    # Dictionary to store processed DataFrames
    processed_dataframes = {}

    # Process each PDB structure
    for pdb_id, df in pdb_dataframes.items():
        logging.info(f"Processing {pdb_id}")
        
        try:
            # Add new features
            df = add_same_chain(df)
            df = add_delta_rsa(df)
            df = add_delta_atchley(df)
            df = add_ca_distances(df, pdb_id)
            
            # Add centroid coordinates if states file exists
            states_file = "states.txt"  # Update path as needed
            if os.path.exists(states_file):
                df = add_centroid_coordinates(df, states_file)
            
            # Store the processed DataFrame
            processed_dataframes[pdb_id] = df
            
        except Exception as e:
            logging.error(f"Error processing {pdb_id}: {e}")
            continue

    # Save all processed files to individual TSVs and create ZIP archive
    if processed_dataframes:
        try:
            zip_path = save_to_zip(processed_dataframes, args.out_dir)
            logging.info(f"Successfully processed {len(processed_dataframes)} structures")
            logging.info(f"Results saved to: {zip_path}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
    else:
        logging.error("No structures were successfully processed")

    # TODO Run with conda activate sb-env ... python Predictor\calc_additional_features.py features_ring.zip -out_dir C:\Users\enric\sb_project