from pathlib import Path
import joblib
import pickle
import pandas as pd
import numpy as np

import logging
import warnings
logging.getLogger("deepchem").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\..*")
warnings.filterwarnings("ignore", message=r".*IProgress not found.*")

import deepchem as dc
from rdkit import Chem


def _impute_feature(pdXY, pd_param):
    feature_names = pd_param["feature_name"].values.tolist()
    impute_value = pd_param.set_index("feature_name")["impute_value"].to_dict()

    df00 = pdXY.copy()
    for cname in feature_names:
        assert cname in df00.columns, cname + " not in dataframe"

        if df00[cname].isnull().sum() > 0:
            df00[cname] = df00[cname].fillna(impute_value[cname])
    return df00


def _standardize_feature(pdXY, pd_param):
    feature_names = pd_param["feature_name"].values.tolist()

    train_mean = pd_param.set_index("feature_name")["train_mean"].to_dict()
    train_std = pd_param.set_index("feature_name")["train_std"].to_dict()

    df00 = pdXY.copy()
    for cname in feature_names:
        assert cname in df00.columns
    
        m = train_mean[cname]
        s = train_std[cname]
        assert (s == s) and (s >= 0) and (m == m)

        if s == 0:
            print("WARNING: " + cname + " has zero std.")
            s = 1
        df00[cname] = (df00[cname] - m) / s
    return df00

def _transform_feature(pdXY, pd_param):
    df00 = pdXY.copy()
    df00 = _impute_feature(df00, pd_param)
    df00 = _standardize_feature(df00, pd_param)
    return df00


def _predict_rf_xgb(model, feature_names, pdXY_transformed, pred_cname="predicted_dG"):
    df00 = pdXY_transformed.copy()
    df00[pred_cname] = model.predict(df00[feature_names].values)
    return df00


def _extract_rdkit_features(smiles_lst, smiles_col="smiles"):
    smiles_lst = list(set([s.strip() for s in smiles_lst]))
    rdkit_featurizer = dc.feat.RDKitDescriptors()
    X = rdkit_featurizer(smiles_lst)
    X1 = []
    nr_feat = 217
    for y in X:
        if y.shape[0] > 0:
            assert y.shape[0] == nr_feat, "Please use DeepChem Version 2.8.0. Current Version is " + dc.__version__
            X1.append(y.tolist())
        else:
            y = [np.nan] * nr_feat
            X1.append(y)
    X1 = np.array(X1)
    feature_names = [c for c,_ in rdkit_featurizer.reqd_properties.items()]
    X2 = pd.DataFrame(X1, columns=feature_names)
    X2[smiles_col] = smiles_lst
    return X2


def predict_rf_xgb(model_path, model_name, df, smiles_col="smiles"):
    assert model_name in ["RF", "XGBoost"]
    model_info = pickle.load(open(model_path, "rb"))
    model = model_info["model"]
    if model_name == "XGBoost":
        model.set_params(device="cpu")
    feat_transform_pd = model_info["feat_transform_pd"]
    feature_names = feat_transform_pd["feature_name"].values.tolist()

    smiles_lst = df[smiles_col].values.tolist()
    X = _extract_rdkit_features(smiles_lst, smiles_col=smiles_col)
    df[smiles_col] = df[smiles_col].apply(lambda s: s.strip())
    pdX = df.merge(X, how="left", on=smiles_col)

    pdX_transformed = _transform_feature(pdX, feat_transform_pd)
    pred_cname = model_name + '_pred'
    df00 = _predict_rf_xgb(model, feature_names, pdX_transformed, pred_cname=pred_cname)
    cnames = df.columns.tolist() + [pred_cname]
    df00 = df00[cnames]
    return df00

def predict_dmpnn_dG(
    model_dir: str,
    model_name: str,
    df: pd.DataFrame,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """
    Predict binding free energy (dG) for unique, valid SMILES using a saved DeepChem DMPNNModel.

    Returns
    -------
    pd.DataFrame
        DataFrame with two columns: `smiles` and `predicted_dG`.
    """
    assert model_name in ["DMPNN"]
    
    batch_size = 64
    enc_hidden = 300
    depth = 3
    enc_dropout_p = 0.1
    ffn_hidden = 300
    ffn_layers = 2
    ffn_dropout_p = 0.1
    features_generators = None

    # 1. Extract unique, non-null SMILES and filter for valid RDKit molecules
    unique_smiles = df[smiles_col].dropna().astype(str).unique().tolist()
    valid_smiles = [s.strip() for s in unique_smiles if Chem.MolFromSmiles(s) is not None]

    if not valid_smiles:
        return pd.DataFrame(columns=[smiles_col, "predicted_dG"])

    # 2. Featurize and keep only molecules that produce valid GraphData
    featurizer = dc.feat.DMPNNFeaturizer(features_generators=features_generators)
    X_all = featurizer.featurize(valid_smiles)

    final_smiles = []
    X_valid = []
    for s, g in zip(valid_smiles, X_all):
        if isinstance(g, dc.feat.graph_data.GraphData):
            final_smiles.append(s)
            X_valid.append(g)

    if not final_smiles:
        return pd.DataFrame(columns=[smiles_col, "predicted_dG"])

    predict_dataset = dc.data.NumpyDataset(X=np.array(X_valid, dtype=object))

    # 3. Load transformer and restore model weights
    transformer = joblib.load(Path(model_dir) / "y_transformer.joblib")

    model = dc.models.DMPNNModel(
        mode="regression",
        n_tasks=1,
        batch_size=batch_size,
        enc_hidden=enc_hidden,
        depth=depth,
        enc_dropout_p=enc_dropout_p,
        ffn_hidden=ffn_hidden,
        ffn_layers=ffn_layers,
        ffn_dropout_p=ffn_dropout_p,
        model_dir=model_dir,
    )
    model.restore()

    # 4. Predict and un-normalize back to original dG scale
    preds_norm = model.predict(predict_dataset)
    preds_dG = transformer.untransform(preds_norm).flatten()

    pred_cname = model_name + '_pred'
    df00 = pd.DataFrame({smiles_col: final_smiles, pred_cname: preds_dG,})

    df00 = df.merge(df00, how="left", on=smiles_col)
    return df00

def predict(model_path, model_name, df, smiles_col="smiles"):
    if model_name in ["RF", "XGBoost"]:
        df00 = predict_rf_xgb(model_path, model_name, df, smiles_col=smiles_col)
    elif model_name in ["DMPNN"]:
        df00 = predict_dmpnn_dG(model_path, model_name, df, smiles_col=smiles_col)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return df00


def predict_ensemble(model_paths: dict, df: pd.DataFrame, smiles_col: str, pred_col: str):
    assert smiles_col in df.columns
    df00 = df[[smiles_col]].drop_duplicates(subset=smiles_col)
    pred_cnames = []
    for model_name, model_path in model_paths.items():
        df00 = predict(model_path, model_name, df00, smiles_col=smiles_col)
        pred_cname = model_name + "_pred"
        pred_cnames.append(pred_cname)
    df00["ensemble_pred"] = df00[pred_cnames].mean(axis=1, skipna=True)
    df00 = df00[[smiles_col, "ensemble_pred"]].rename(columns={"ensemble_pred": pred_col})
    df01 = df.merge(df00, how="left", on=smiles_col)
    return df01