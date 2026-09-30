# GameProb Lab

A time-aware sports outcome probability workflow with logistic and gradient-boosting models, Brier score, log loss, ROC-AUC, calibration, and a matchup scenario editor.

The default league is synthetic. Upload a dated binary-outcome CSV with numeric pre-game features for substantive analysis.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Outputs are uncertain model estimates, never betting guarantees.

