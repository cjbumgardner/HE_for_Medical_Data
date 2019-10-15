import pickle
import pandas as pd
from pathlib import Path
import streamlit as st
import sys
import re

TESTDIR =  Path(__file__).parent
MODEL_PARAMS = Path(__file__).parent.parent/"server"/"model_params"
st.write(f"The model_params directory: {MODEL_PARAMS}")
#Model directories and tags for pandas df
MODELS = [("nn_poly12_mortality", "12"),
        ("nn_poly12small_mortality", "12small"),
        ("nn_poly13_mortality", "13"),
        ("nn_poly13small_mortality", "13small"),
        ("nn_poly14_mortality", "14"),
        ("nn_poly14small_mortality", "14small"),
        ("nn_relu_mortality", "relu"),
        ("nn_relusmall_mortality", "relusmall"),
         ]
#define model params directory modelparams
def get_np_predictions(modeldir, tag):
    modeldir = MODEL_PARAMS/modeldir
    list_of_paths = modeldir.glob("*")
    paths = sorted(list_of_paths, key=lambda p: p.stat().st_ctime)
    paths.reverse()
    for path in paths:
        if path.name[0:7] == "predict":
            latest_path = path
            break
    with open(latest_path,"rb") as f:
        predicts = pickle.load(f)
    df = pd.DataFrame(predicts, columns=[f"actual", f"predicted_{tag}"])
    return df
    
def get_training_data(modeldir, tag):
    modeldir = MODEL_PARAMS/modeldir
    list_of_paths = modeldir.glob("*")
    paths = sorted(list_of_paths, key=lambda p: p.stat().st_ctime)
    train_data = []
    for path in paths:
        if path.name[0:4] == "loss":
            with open(path,"rb") as f:
                train_data.extend(pickle.load(f))
    for dict in train_data:
        dict.pop("step",None)
        for k,v in dict.items():
            dict[k] = v.detach().numpy()
    train_df = pd.DataFrame(train_data)
    return (tag, train_df)

def precision_accuracy(df):
    """Yields precision and accuracy for pandas df with columns = ["actual","predicted_TAG"]"""
    columns = df.columns
    for x in columns:
        if x[0:9] == "predicted":
            tag = x.split("_")[-1]

    def round(row):
        if row[f"predicted_{tag}"] >= .5: 
            return 1
        if row[f"predicted_{tag}"] < .5:
            return 0

    df["bool"] = df.apply(lambda row: round(row), axis=1)
    recall = df[df["actual"]==1]["bool"].mean()
    precision = df[df["bool"]==1]["actual"].mean()
    dict = {"NNdesign": tag, "precision": precision, "recall": recall}
    return dict



st.write("Fetching precision and recall values for the different network designs...")
predict_dfs = [get_np_predictions(*x) for x in MODELS]
prec_acc_df = pd.DataFrame([precision_accuracy(x) for x in predict_dfs])
st.write(prec_acc_df)

st.write("Getting training loss and test loss data for different network designs...")
train_dfs = [get_training_data(*x) for x in MODELS]
train_times = pd.DataFrame([[x[0], len(x[1].index)] for x in train_dfs], columns=["NNdesign", "train_steps"])
st.write(train_times)
