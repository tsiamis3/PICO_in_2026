"""
Αναπαραγωγή των Results από τα αποθηκευμένα μοντέλα (best_model_fold*.h5) — χωρίς νέο training.

Ίδιο preprocessing με τα training scripts:
    A1  KFold_CV.py                (Picosec_angelos.root,           μοντέλα Check_threshold_cut_1_q_round2)
    A3  Noizy_train.py             (Picosec_angelos_10mV_clean.root, μοντέλα Noizy_train)
    A2  Analysis_Train_nominal_test_noizy.ipynb, cell 0 (CORRECT_VOL=True, ensemble των 10 μοντέλων A1)

Για A1/A3: το shuffle (random_state=1312) και το KFold (random_state=1312) είναι ντετερμινιστικά, άρα
κάθε event πάει στο ΙΔΙΟ fold με το training και προβλέπεται από το μοντέλο που ΔΕΝ το είδε (out-of-fold).
Το t_pred στα training scripts δεν είχε seed, οπότε εδώ τραβιέται νέο (με seed, για αναπαραγωγιμότητα):
στατιστικά ισοδύναμο, όχι bit-for-bit ίδιο με τα παλιά Results.csv.
Για A2 το t_pred είχε seed (1312), άρα η αναπαραγωγή είναι ακριβής.
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import uproot
import tensorflow as tf
from sklearn.model_selection import KFold

DATA_DIR = "data"
MODELS_DIR = "models"
ROOT_NOMINAL = os.path.join(DATA_DIR, "Picosec_angelos.root")
ROOT_NOISY = os.path.join(DATA_DIR, "Picosec_angelos_10mV_clean.root")
THRES = 1
SEED_TPRED = 2026          # seed για το νέο t_pred των A1/A3
PADS = range(4)


# ---------------- Κοινά ----------------
def Data_norm_min_max(Datta, cols):
    """norm_scenario=2 του pico_utils: min-max ανά στήλη."""
    DD = Datta.copy(deep=True)
    for col in cols:
        DD[col] = (DD[col] - DD[col].min()) / (DD[col].max() - DD[col].min())
    return DD


def apply_threshold(FF, thres=THRES, time_fill=10):
    for i in PADS:
        m = FF[f"qall_idx_{i}"] < thres          # μάσκα πριν μηδενιστεί το qall
        FF.loc[m, [f"qall_idx_{i}", f"epeak_{i}"]] = 0
        FF.loc[m, f"time_fit_{i}"] = time_fill
    return FF


def input_columns(FF):
    vv = [c for c in FF.columns if "vol" in c]
    qq = [c for c in FF.columns if "qall" in c]
    ee = [c for c in FF.columns if "epeak" in c]
    tt = [c for c in FF.columns if "time" in c]
    return ee + qq + vv, tt


def rename_noisy(FF):
    """vol_max_waveform_i -> vol_max_i, sigmoid_timepoint_i -> time_fit_i (τα παλιά πετιούνται)."""
    rename_map = {}
    for i in PADS:
        rename_map[f"vol_max_waveform_{i}"] = f"vol_max_{i}"
        rename_map[f"sigmoid_timepoint_{i}"] = f"time_fit_{i}"
    FF = FF.drop(columns=list(rename_map.values()), errors="ignore").rename(columns=rename_map)
    assert not FF.columns.duplicated().any()
    return FF


def load_models(folder):
    files = sorted(glob.glob(os.path.join(MODELS_DIR, folder, "best_model_fold*.h5")),
                   key=lambda p: int(re.search(r"fold(\d+)", p).group(1)))
    return [tf.keras.models.load_model(f, compile=False) for f in files]


def kfold_predict(FF, cols, models, K=10):
    """Out-of-fold προβλέψεις, ίδια splits με το training."""
    t_net = np.full(len(FF), np.nan)
    s_net = np.full(len(FF), np.nan)
    fold_all = np.full(len(FF), -1, dtype=int)
    kf = KFold(n_splits=K, shuffle=True, random_state=1312)
    for fold, (train_idx, test_idx) in enumerate(kf.split(FF)):
        Y = models[fold].predict(FF.iloc[test_idx][cols].values.astype("float32"), batch_size=8192, verbose=0)
        t_net[test_idx] = Y[:, 0]
        s_net[test_idx] = np.exp(0.5 * Y[:, 1])
        fold_all[test_idx] = fold
        tt = Y[:, 0] - FF["t_pred"].values[test_idx]
        print(f"  fold {fold}: bias = {1e3*tt.mean():+.2f} ps, rms = {1e3*tt.std():.2f} ps")
    Data_test = FF.copy()
    Data_test["t_network"], Data_test["std_network"] = t_net, s_net
    Data_test["tt"] = Data_test["t_network"] - Data_test["t_pred"]
    Data_test["fold"] = fold_all
    return Data_test


def save(Data_test, name):
    out = os.path.join(DATA_DIR, name)
    Data_test.to_parquet(out, compression="zstd")
    print(f"  -> {out} ({len(Data_test)} events)")


# ---------------- A1: nominal (KFold_CV.py) ----------------
def make_A1():
    print("A1: nominal")
    rng = np.random.default_rng(SEED_TPRED)
    FF = uproot.open(ROOT_NOMINAL)["events;1"].arrays(library="pd")
    FF["t_pred"] = rng.uniform(-1, 1, len(FF))
    for i in PADS:
        FF[f"time_fit_{i}"] = FF[f"time_fit_{i}"] + FF["t_pred"]
    FF = apply_threshold(FF)
    norm_cols, tt = input_columns(FF)
    FF = Data_norm_min_max(FF, norm_cols)
    cols = norm_cols + tt
    FF = FF.sample(frac=1, random_state=1312).reset_index(drop=True)
    save(kfold_predict(FF, cols, load_models("Check_threshold_cut_1_q_round2")),
         "Check_threshold_cut_1_q_round2__Results.parquet")


# ---------------- A3: train + test noisy (Noizy_train.py) ----------------
def make_A3():
    print("A3: noisy train")
    rng = np.random.default_rng(SEED_TPRED + 1)
    FF = uproot.open(ROOT_NOISY)["events;1"].arrays(library="pd")
    FF = rename_noisy(FF)
    FF["t_pred"] = rng.uniform(-1, 1, len(FF))
    for i in PADS:
        FF[f"time_fit_{i}"] = FF[f"time_fit_{i}"] + FF["t_pred"]
    FF = apply_threshold(FF)
    norm_cols, tt = input_columns(FF)
    FF = FF.dropna(subset=norm_cols + tt).reset_index(drop=True)
    FF = Data_norm_min_max(FF, norm_cols)
    cols = norm_cols + tt
    FF = FF.sample(frac=1, random_state=1312).reset_index(drop=True)
    save(kfold_predict(FF, cols, load_models("Noizy_train")), "Noizy_train__Results.parquet")


# ---------------- A2: train nominal / test noisy (notebook cell 0) ----------------
def make_A2(correct_vol=True):
    print("A2: train nominal / test noisy")

    def preprocess(FF, seed):
        rng = np.random.default_rng(seed)
        FF["t_pred"] = rng.uniform(-0.5, 0.5, len(FF))
        for i in PADS:
            FF[f"time_fit_{i}"] = FF[f"time_fit_{i}"] + FF["t_pred"]
        return apply_threshold(FF)

    # normalizer από τα training (nominal) δεδομένα: γραμμικό map ανά στήλη
    FF_train = preprocess(uproot.open(ROOT_NOMINAL)["events;1"].arrays(library="pd"), seed=0)
    norm_cols, tt = input_columns(FF_train)
    cols = norm_cols + tt
    lin = {c: (FF_train[c].min(), FF_train[c].max()) for c in norm_cols}

    FF = uproot.open(ROOT_NOISY)["events;1"].arrays(library="pd")
    if correct_vol:
        for i in PADS:
            m = FF[f"qall_idx_{i}"] >= THRES
            off = (FF.loc[m, f"vol_max_waveform_{i}"] - FF.loc[m, f"vol_max_{i}"]).median()
            FF[f"vol_max_waveform_{i}"] = FF[f"vol_max_waveform_{i}"] - off
            print(f"  pad {i}: voltage offset = {off:.5f}")
    FF = rename_noisy(FF)
    FF = preprocess(FF, seed=1312)
    for c in norm_cols:
        mn, mx = lin[c]
        FF[c] = (FF[c] - mn) / (mx - mn)

    X = FF[cols].values.astype("float32")
    mus, vars_ = [], []
    for k, model in enumerate(load_models("Check_threshold_cut_1_q_round2")):
        Y = model.predict(X, batch_size=8192, verbose=0)
        mus.append(Y[:, 0]); vars_.append(np.exp(Y[:, 1]))
        tt_k = Y[:, 0] - FF["t_pred"].values
        print(f"  model {k}: bias = {1e3*np.nanmean(tt_k):+.2f} ps, rms = {1e3*np.nanstd(tt_k):.2f} ps")
    mus, vars_ = np.array(mus), np.array(vars_)
    Data_test = FF.copy()
    Data_test["t_network"] = mus.mean(axis=0)
    Data_test["std_network"] = np.sqrt(vars_.mean(axis=0) + mus.var(axis=0))
    Data_test["tt"] = Data_test["t_network"] - Data_test["t_pred"]
    n_nan = int(Data_test["tt"].isna().sum())
    if n_nan:
        print(f"  {n_nan} events με NaN inputs -> αφαιρούνται")
        Data_test = Data_test.dropna(subset=["tt"])
    save(Data_test, "Check_threshold_cut_1_q_round2__Results_noisy_10mV_volcorr.parquet")


if __name__ == "__main__":
    make_A1()
    make_A2()
    make_A3()
