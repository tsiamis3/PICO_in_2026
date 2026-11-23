# PICOSEC Micromegas — Timing network results

Αποτελέσματα του timing network (heteroscedastic NN, 10-fold CV) για τον 4-pad PICOSEC Micromegas.
**Σελίδα αποτελεσμάτων:** `index.html` (GitHub Pages).

## Αναλύσεις
| | Training | Test |
|---|---|---|
| A1 | nominal (`Picosec_angelos.root`) | nominal |
| A2 | nominal | 10 mV noise (`Picosec_angelos_10mV_clean.root`) |
| A3 | 10 mV noise | 10 mV noise |

## Περιεχόμενα
- `index.html` — αυτόνομη σελίδα με όλα τα αποτελέσματα
- `plots/` — όλα τα plots (PNG/PDF), `summary.json`, CSV του charge analysis
- `scripts/make_results.py` — ξαναβγάζει τις out-of-fold προβλέψεις από τα `best_model_fold*.h5`
- `scripts/picosec_analysis.py` — resolution, pulls, charge analysis, heatmaps (σε ps, με πραγματικό charge)
- `scripts/hierarchical_bayes.py` — ιεραρχικό Bayesian μοντέλο (NumPyro): bias και βαθμονόμηση του σ_network ανά charge bin και ανά κελί θέσης (`plots/hb_*`, `plots/hierarchical_summary.json`)
- `scripts/build_page.py` — φτιάχνει το `index.html` από τον φάκελο `plots/`
- `scripts/slim_results.py` — μετατροπή των μεγάλων `Results*.csv` σε parquet

Τα ROOT αρχεία, τα Results και τα μοντέλα δεν μπαίνουν στο repo (μέγεθος). Βάλ' τα σε `data/` και `models/` για να ξανατρέξεις:

```
python scripts/make_results.py
python scripts/picosec_analysis.py
python scripts/hierarchical_bayes.py      # ~1–2 ώρες σε CPU· το posterior σώζεται στο data/hb_posterior/
python scripts/build_page.py              # γράφει site/index.html → αντιγραφή στο index.html
```

Το `hierarchical_bayes.py` θέλει επιπλέον `pip install numpyro jax`. Με λίγη RAM τρέξε ένα-ένα: `python scripts/hierarchical_bayes.py A2`, και στο τέλος χωρίς όρισμα (φορτώνει από το cache και βγάζει τα plots).
