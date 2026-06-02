import streamlit as st
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
import gdown
import os
import zipfile
import glob
import joblib 

# --- MAIN INTERFACE ---
st.title("OX1R pIC₅₀ Prediction Ensemble Model 🧬")
st.write("This application utilizes an Ensemble XGBoost model (50 models) and ECFP4 features, applying a consensus prediction strategy.")

# --- DOWNLOAD AND EXTRACT MODEL FROM GOOGLE DRIVE ---
@st.cache_resource
def load_model_from_drive():
    file_id = '1_cCTY3euT-yPtBsp1wYW9P0yOadVB9Db'
    url = f'https://drive.google.com/uc?id={file_id}'
    zip_path = "model_orx1.zip"
    extract_folder = "model_extracted"
    
    if not os.path.exists(zip_path):
        with st.spinner("Downloading and extracting ZIP file from Drive (68MB)... Please wait!"):
            gdown.download(url, zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder) 
                
    pkl_files = glob.glob(f"{extract_folder}/**/*.pkl", recursive=True)
    if not pkl_files:
        raise FileNotFoundError("No .pkl file found inside the ZIP archive!")
        
    ensemble_data = joblib.load(pkl_files[0])
    
    preprocessor = ensemble_data['preprocessor']
    models = ensemble_data['models']
    
    return preprocessor, models

try:
    preprocessor, models = load_model_from_drive()
    st.success(f"Successfully loaded Preprocessor and {len(models)} XGBoost models! Ready. 🚀")
except Exception as e:
    st.error(f"Error loading model. Details: {e}")
    st.stop()

# --- CHEMICAL PROCESSING: SMILES -> ECFP4 ---
def smiles_to_ecfp4(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp).reshape(1, -1)

# --- INPUT FORM & CONSENSUS PREDICTION ---
st.markdown("---")
smiles_input = st.text_input("Enter the SMILES string of the compound here:", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F")

if st.button("Predict pIC₅₀", type="primary"):
    if smiles_input.strip() == "":
        st.warning("Please enter a SMILES string!")
    else:
        with st.spinner("Running consensus prediction across 50 models..."):
            features_raw = smiles_to_ecfp4(smiles_input)
            
            if features_raw is not None:
                try:
                    features_proc = preprocessor.transform(features_raw)
                    
                    predictions = []
                    for sub_model in models:
                        pred = sub_model.predict(features_proc)[0]
                        predictions.append(pred)
                        
                    final_pIC50 = np.mean(predictions)
                    uncertainty = np.std(predictions)
                    
                    col1, col2 = st.columns(2)
                    col1.metric(label="pIC₅₀ (Consensus Score)", value=f"{final_pIC50:.4f}")
                    col2.metric(label="Uncertainty (Std Dev)", value=f"{uncertainty:.4f}")
                    
                    st.balloons()
                except Exception as e:
                    st.error(f"Error during feature calculation: {e}")
            else:
                st.error("Invalid SMILES string. Please double-check the chemical structure.")
