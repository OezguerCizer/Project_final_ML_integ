# train_ml.py
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# (nur Kopfteil austauschen)
from pathlib import Path
FEATURES_CSV = "features_for_ml.csv"
if not Path(FEATURES_CSV).exists():
    raise SystemExit("❌ features_for_ml.csv nicht gefunden. Bitte zuerst `python data_prep.py` ausführen.")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def get_feature_lists(df: pd.DataFrame):
    target = "Total losses"
    # Eingabefeatures: alles numerische außer Ziel + (Country/Name)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    drop_cols = [target]
    if "Year" in num_cols:
        # Year bleibt als numerisches Feature drin
        pass
    X_cols = [c for c in num_cols if c not in drop_cols]
    return X_cols, target

def train_one_country(df_c: pd.DataFrame, country_id: str):
    # Zeitlich sortiert
    df_c = df_c.sort_values("Year").copy()

    X_cols, y_col = get_feature_lists(df_c)
    X = df_c[X_cols].values
    y = df_c[y_col].values

    # Minimal-Check: genug Samples?
    if len(df_c) < 8:
        return None  # zu wenig Daten – kein Modell

    # 3-fach TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)

    # Zwei Modelle: Ridge (linear) & RandomForest (nichtlinear)
    ridge = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # robust für spärlich/Nullen
        ("model", Ridge(alpha=1.0, random_state=42))
    ])
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
    )

    # Cross-val MAE mitteln
    def cv_mae(model):
        maes = []
        for tr, te in tscv.split(X):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            maes.append(mean_absolute_error(y_te, pred))
        return float(np.mean(maes))

    ridge_mae = cv_mae(ridge)
    rf_mae = cv_mae(rf)

    # Bestes wählen
    best_model, best_name, best_mae = (ridge, "ridge", ridge_mae) if ridge_mae <= rf_mae else (rf, "random_forest", rf_mae)
    best_model.fit(X, y)

    # speichern
    out_path = MODELS_DIR / f"{country_id}_{best_name}.joblib"
    joblib.dump({"model": best_model, "X_cols": X_cols, "y_col": y_col, "metrics": {"cv_mae": best_mae}}, out_path)

    return {
        "country": country_id,
        "algorithm": best_name,
        "cv_mae": best_mae,
        "model_path": str(out_path),
        "n_samples": len(df_c),
        "n_features": len(X_cols),
    }

def main():
    assert Path(FEATURES_CSV).exists(), f"{FEATURES_CSV} nicht gefunden. Bitte vorher data_prep.py ausführen."
    df = pd.read_csv(FEATURES_CSV)
    # NaN → 0 (keine Imputation)
    df = df.fillna(0)

    results = []
    for cid, grp in df.groupby("Country_ID"):
        res = train_one_country(grp, cid)
        if res is not None:
            results.append(res)

    res_df = pd.DataFrame(results).sort_values("cv_mae")
    res_df.to_csv(MODELS_DIR / "training_summary.csv", index=False)
    print(f"✅ Training fertig. Modelle: {len(res_df)}  | Summary: models/training_summary.csv")
    if len(res_df):
        print(res_df.head(10))

if __name__ == "__main__":
    main()
