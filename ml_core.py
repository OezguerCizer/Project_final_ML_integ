# ml_core.py — Autopilot für Features/Training + Forecast
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

YEAR_CANDIDATES = [Path("Merged_Energy_Losses.csv"), Path("data/Merged_Energy_Losses.csv")]
FEATURES_CSV = Path("features_for_ml.csv")
MODELS_DIR = Path("models")

def _find_first_existing(paths):
    for p in paths:
        if Path(p).exists(): return Path(p)
    return None

def load_year_df() -> pd.DataFrame:
    p = _find_first_existing(YEAR_CANDIDATES)
    if p is None:
        raise FileNotFoundError("Merged_Energy_Losses.csv nicht gefunden (in ./ oder ./data/).")
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    need = {"Country_ID","Year","Total losses"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"Spalten fehlen: {missing}")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)  # Lücken => 0
    return df

def build_features(df_year: pd.DataFrame) -> pd.DataFrame:
    df = df_year.copy().sort_values(["Country_ID","Year"])
    def add_feats(g):
        g=g.copy(); y=g["Total losses"]
        g["loss_lag1"]=y.shift(1).fillna(0); g["loss_lag2"]=y.shift(2).fillna(0); g["loss_lag3"]=y.shift(3).fillna(0)
        g["loss_roll3_mean"]=y.rolling(3,min_periods=1).mean(); g["loss_roll5_mean"]=y.rolling(5,min_periods=1).mean()
        g["year_centered"]=g["Year"]-g["Year"].min(); g["year_sq"]=g["year_centered"]**2
        return g
    df = df.groupby("Country_ID", group_keys=False).apply(add_feats)

    base = ["Year","year_centered","year_sq","loss_lag1","loss_lag2","loss_lag3","loss_roll3_mean","loss_roll5_mean"]
    exog = [c for c in df.columns if c.startswith(("PEC","FEC"))]
    keep = ["Country_ID"] + (["Country_name"] if "Country_name" in df.columns else []) + base + exog + ["Total losses"]
    feat = df[keep].copy()
    feat.to_csv(FEATURES_CSV, index=False)
    return feat

def _feature_list(df: pd.DataFrame):
    y = "Total losses"
    num_cols = df.select_dtypes(include="number").columns.tolist()
    X_cols = [c for c in num_cols if c!=y]
    return X_cols, y

def train_models_per_country(df_feat: pd.DataFrame) -> pd.DataFrame:
    MODELS_DIR.mkdir(exist_ok=True)
    rows=[]
    for cid,g in df_feat.groupby("Country_ID"):
        g=g.sort_values("Year")
        if len(g) < 8: continue
        X_cols,y_col = _feature_list(g)
        X=g[X_cols].values; y=g[y_col].values

        ridge = Pipeline([("s",StandardScaler(with_mean=False)),("m",Ridge(alpha=1.0, random_state=42))])
        rf    = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)

        def cv_mae(m):
            tscv=TimeSeriesSplit(n_splits=3); maes=[]
            for tr,te in tscv.split(X):
                m.fit(X[tr],y[tr]); p=m.predict(X[te]); maes.append(mean_absolute_error(y[te],p))
            return float(np.mean(maes)) if maes else float("inf")

        r_mae=cv_mae(ridge); f_mae=cv_mae(rf)
        best, name, mae = (ridge,"ridge",r_mae) if r_mae<=f_mae else (rf,"random_forest",f_mae)
        best.fit(X,y)
        out = MODELS_DIR / f"{cid}_{name}.joblib"
        joblib.dump({"model":best,"X_cols":X_cols,"y_col":y_col,"metrics":{"cv_mae":mae}}, out)
        rows.append({"Country_ID":cid,"algorithm":name,"cv_mae":mae,"model_path":str(out),"n":len(g)})

    res = pd.DataFrame(rows).sort_values("cv_mae") if rows else pd.DataFrame(columns=["Country_ID","algorithm","cv_mae","model_path","n"])
    (MODELS_DIR/"training_summary.csv").write_text(res.to_csv(index=False))
    return res

def load_best_model_for(cid: str):
    files = sorted(MODELS_DIR.glob(f"{cid}_*.joblib"))
    return joblib.load(files[0]) if files else None

def forecast_country(df_feat: pd.DataFrame, cid: str, model_pack: dict, horizon:int=5, scenario:str="Konstant (letzte Werte)") -> pd.DataFrame:
    model=model_pack["model"]; X_cols=model_pack["X_cols"]; y_col=model_pack["y_col"]
    dfc = df_feat[df_feat["Country_ID"]==cid].sort_values("Year").fillna(0).copy()
    if dfc.empty: return pd.DataFrame(columns=["Year","forecast"])
    last_year=int(dfc["Year"].max()); year_min=int(dfc["Year"].min())
    exog_cols=[c for c in X_cols if c.startswith(("PEC","FEC"))]
    preds=[]; hist=dfc.copy()

    for step in range(1,horizon+1):
        year_next=last_year+step
        base={"Year":year_next,"year_centered":year_next-year_min,"year_sq":(year_next-year_min)**2}
        vals=hist[y_col].values
        base["loss_lag1"]=float(vals[-1]) if len(vals)>=1 else 0.0
        base["loss_lag2"]=float(vals[-2]) if len(vals)>=2 else 0.0
        base["loss_lag3"]=float(vals[-3]) if len(vals)>=3 else 0.0
        base["loss_roll3_mean"]=float(np.mean(vals[-3:])) if len(vals) else 0.0
        base["loss_roll5_mean"]=float(np.mean(vals[-5:])) if len(vals) else 0.0
        exog={}
        for c in exog_cols:
            base_v=float(hist.iloc[-1].get(c,0.0))
            exog[c]=base_v if scenario=="Konstant (letzte Werte)" else (base_v*(1.02**step) if scenario=="+2%/Jahr" else base_v*(0.98**step))
        row={**base,**exog}
        X_next=np.array([[row.get(col,0.0) for col in X_cols]])
        y_pred=float(model.predict(X_next)[0])
        new_entry={**row,y_col:y_pred}
        for col in hist.columns:
            if col not in new_entry: new_entry[col]=hist.iloc[-1].get(col,0.0)
        hist=pd.concat([hist,pd.DataFrame([new_entry])], ignore_index=True)
        preds.append({"Year":year_next,"forecast":y_pred})
    return pd.DataFrame(preds)

# ---------- Autopilot ----------
def ensure_ready(verbose: bool = False):
    """Sorgt dafür, dass Features & Modelle existieren.
    Gibt (df_features, training_summary_df) zurück."""
    if verbose: print("→ Prüfe Features/Modelle …")
    if not FEATURES_CSV.exists():
        if verbose: print("   • Features fehlen → baue …")
        df = load_year_df()
        build_features(df)
    df_feat = pd.read_csv(FEATURES_CSV)
    MODELS_DIR.mkdir(exist_ok=True)
    any_model = any(MODELS_DIR.glob("*_*.joblib"))
    if not any_model:
        if verbose: print("   • Modelle fehlen → trainiere …")
        summary = train_models_per_country(df_feat)
    else:
        summary_path = MODELS_DIR / "training_summary.csv"
        summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    return df_feat, summary
