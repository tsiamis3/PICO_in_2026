"""
Συμπίεση των Results*.csv σε parquet (float32 + zstd), ώστε να ανέβουν εύκολα.
Δεν αλλάζει/πετάει καμία στήλη ή γραμμή — μόνο αλλάζει format.

Τρέξιμο (από τον φάκελο PICO_in_2026):
    pip install pandas pyarrow
    python slim_results.py
Βγάζει τα *.parquet δίπλα σε κάθε csv.
"""
import os
import numpy as np
import pandas as pd

main_path = r"C:\Users\tsiam\OneDrive\Desktop\PICO_in_2026"

files = [
    # 1) Nominal: train/test στο Picosec_angelos.root
    os.path.join(main_path, "Check_threshold_cut_1_q_round2", "Results.csv"),
    # 2) Train nominal / test noisy
    os.path.join(main_path, "Check_threshold_cut_1_q_round2", "Results_noisy_10mV.csv"),
    os.path.join(main_path, "Check_threshold_cut_1_q_round2", "Results_noisy_10mV_volcorr.csv"),
    # 3) Train + test noisy (Picosec_angelos_10mV_clean.root)
    os.path.join(main_path, "Noizy_train", "Results.csv"),
]

for f in files:
    if not os.path.exists(f):
        print("Δεν βρέθηκε:", f)
        continue
    DD = pd.read_csv(f, index_col=0)
    # float64 -> float32 (αρκετή ακρίβεια: ~1e-7 σχετικό, δηλ. << 1 fs στους χρόνους σε ns)
    for c in DD.select_dtypes("float64").columns:
        DD[c] = DD[c].astype(np.float32)
    folder = os.path.basename(os.path.dirname(f))
    out = os.path.join(main_path, f"{folder}__{os.path.splitext(os.path.basename(f))[0]}.parquet")
    DD.to_parquet(out, compression="zstd")
    print(f"{f}: {os.path.getsize(f)/1e6:.0f} MB -> {out}: {os.path.getsize(out)/1e6:.1f} MB")
