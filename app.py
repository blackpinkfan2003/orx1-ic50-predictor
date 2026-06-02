import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
import gdown
import os
import zipfile
import glob
import joblib 

# --- NEW IMPORTS FOR DOCKING ---
import tempfile
from vina import Vina

# --- MAIN INTERFACE ---
st.set_page_config(page_title="OX1R Virtual Screening", page_icon="🧬", layout="wide")
st.title("OX1R pIC₅₀ Prediction & Molecular Docking 🧬")
st.write("This application utilizes an Ensemble XGBoost model (50 models) and AutoDock Vina for comprehensive virtual screening.")

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


# --- DOCKING CORE FUNCTION ---
def run_vina_docking(ligand_pdbqt_content):
    """Run AutoDock Vina utilizing temporary files to avoid Streamlit Cloud permission errors"""
    # 1. Read the read-only protein file from GitHub repository
    with open('protein.pdbqt', 'r') as f:
        protein_content = f.read()

    # 2. Create writable temporary files for Vina
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdbqt", mode="w") as tmp_ligand, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".pdbqt", mode="w") as tmp_protein:
        
        tmp_ligand.write(ligand_pdbqt_content)
        ligand_path = tmp_ligand.name
        
        tmp_protein.write(protein_content)
        protein_path = tmp_protein.name
        
    try:
        # 3. Setup Vina with user configurations
        v = Vina(sf_name='vina', seed=42)
        v.set_receptor(protein_path)
        v.set_ligand_from_file(ligand_path)
        
        # Grid box from config.txt
        v.compute_vina_maps(center=[-12.535, -61.881, -34.156], box_size=[18, 18, 18])
        
        # Docking (exhaustiveness = 8, num_modes = 10)
        v.dock(exhaustiveness=8, n_poses=10)
        energies = v.energies(n_poses=10)
        
        # Save output poses to a string for downloading
        out_path = ligand_path + "_out.pdbqt"
        v.write_poses(out_path, n_poses=10, energy_range=8)
        
        with open(out_path, 'r') as f:
            docked_poses_content = f.read()
            
        return energies, docked_poses_content
    finally:
        # 4. Clean up
        if os.path.exists(ligand_path): os.remove(ligand_path)
        if os.path.exists(protein_path): os.remove(protein_path)
        if 'out_path' in locals() and os.path.exists(out_path): os.remove(out_path)


# --- DIVIDE UI INTO 2 TABS ---
tab1, tab2 = st.tabs(["🔮 pIC₅₀ Prediction (QSAR Model)", "🧩 Molecular Docking (AutoDock Vina)"])

# ==========================================
# ====== TAB 1: QSAR (YOUR EXACT CODE) =====
# ==========================================
with tab1:
    st.success(f"Successfully loaded Preprocessor and {len(models)} XGBoost models! Ready. 🚀")
    st.markdown("---")
    input_mode = st.radio("Choose Input Method:", ["Single Compound", "Batch Prediction (.txt)"], horizontal=True)

    if input_mode == "Single Compound":
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

    elif input_mode == "Batch Prediction (.txt)":
        st.info("Upload a text file (.txt) containing one SMILES string per line.")
        uploaded_file = st.file_uploader("Choose a file", type=["txt"])
        
        if uploaded_file is not None:
            if st.button("Run Batch Prediction", type="primary"):
                content = uploaded_file.getvalue().decode("utf-8").splitlines()
                smiles_list = [line.strip() for line in content if line.strip()]
                
                if not smiles_list:
                    st.warning("The uploaded file is empty or contains no valid lines.")
                else:
                    results = []
                    progress_text = "Processing compounds. Please wait..."
                    my_bar = st.progress(0, text=progress_text)
                    total = len(smiles_list)
                    
                    for idx, sm in enumerate(smiles_list):
                        my_bar.progress((idx + 1) / total, text=f"Processing {idx + 1}/{total}: {sm[:20]}...")
                        features_raw = smiles_to_ecfp4(sm)
                        
                        if features_raw is not None:
                            try:
                                features_proc = preprocessor.transform(features_raw)
                                predictions = [sub_model.predict(features_proc)[0] for sub_model in models]
                                final_pIC50 = np.mean(predictions)
                                uncertainty = np.std(predictions)
                                
                                results.append({
                                    "SMILES": sm,
                                    "pIC50_Consensus": round(final_pIC50, 4),
                                    "Uncertainty_StdDev": round(uncertainty, 4),
                                    "Status": "Success"
                                })
                            except Exception as e:
                                results.append({
                                    "SMILES": sm,
                                    "pIC50_Consensus": None,
                                    "Uncertainty_StdDev": None,
                                    "Status": f"Error: {str(e)}"
                                })
                        else:
                            results.append({
                                "SMILES": sm,
                                "pIC50_Consensus": None,
                                "Uncertainty_StdDev": None,
                                "Status": "Invalid SMILES"
                            })
                    
                    my_bar.empty()
                    st.success(f"Batch prediction completed for {total} compounds!")
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                    
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='batch_predictions_orx1.csv',
                        mime='text/csv',
                    )

# ==========================================
# ====== TAB 2: MOLECULAR DOCKING ==========
# ==========================================
with tab2:
    st.subheader("Target: Orexin-1 Receptor (OX1R)")
    st.info("Upload your prepared Ligand `.pdbqt` file. The target protein and grid box parameters are pre-configured.")
    
    # Cấu hình hiển thị thông tin lưới (Grid Box) cho pro
    with st.expander("⚙️ View Grid Box Configuration"):
        st.code("""
        center_x = -12.535   | size_x = 18
        center_y = -61.881   | size_y = 18
        center_z = -34.156   | size_z = 18
        exhaustiveness = 8   | num_modes = 10
        energy_range = 8     | seed = 42
        """)
        
    uploaded_ligand = st.file_uploader("Upload Ligand (.pdbqt)", type=['pdbqt'])
    
    if uploaded_ligand is not None:
        if st.button("▶️ Run AutoDock Vina", type="primary"):
            if not os.path.exists('protein.pdbqt'):
                st.error("🚨 `protein.pdbqt` file is missing in the root directory!")
            else:
                ligand_content = uploaded_ligand.getvalue().decode("utf-8")
                
                with st.spinner("Running molecular docking... This may take a minute depending on ligand complexity."):
                    try:
                        energies, docked_poses = run_vina_docking(ligand_content)
                        
                        st.success("✅ Docking completed successfully!")
                        st.write("### Binding Affinity Results")
                        
                        # Hiển thị bảng kết quả
                        res_data = [{"Pose": i+1, "Affinity (kcal/mol)": round(e[0], 2)} for i, e in enumerate(energies)]
                        st.dataframe(res_data, use_container_width=True)
                        
                        # Nút tải file cấu trúc sau docking
                        st.download_button(
                            label="📥 Download Docked Poses (.pdbqt)",
                            data=docked_poses,
                            file_name=f"docked_{uploaded_ligand.name}",
                            mime='chemical/x-pdb',
                        )
                        
                    except Exception as e:
                        st.error(f"An error occurred during Vina execution: {e}")
