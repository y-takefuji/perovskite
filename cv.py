import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from scipy.stats import spearmanr
import shap

# ============================================================
# 1. Load & Prepare Data
# ============================================================
df = pd.read_csv('Perovskite_Stability_with_features.csv')
df = df.drop(columns=['X site'])

string_cols = df.select_dtypes(include='object').columns.tolist()
target_col = 'energy_above_hull (meV/atom)'
df = df.dropna(subset=[target_col])

for col in string_cols:
    if col != target_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

mean_val = df[target_col].mean()
df['target_binary'] = (df[target_col] > mean_val).astype(int)

feature_cols = [c for c in df.columns if c not in [target_col, 'target_binary']]
X = df[feature_cols].copy()
y = df['target_binary'].copy()
print(f"Features ({len(feature_cols)}): {feature_cols}")

# Dataset shape
print(f"\nDataset shape: {df.shape}")

# Target distribution
print(f"\nTarget distribution:")
print(df[target_col].describe())

# Binary classification using mean threshold
mean_val = df[target_col].mean()
print(f"\nMean threshold: {mean_val:.4f}")
df['target_binary'] = (df[target_col] > mean_val).astype(int)
print(f"Class distribution:\n{df['target_binary'].value_counts()}")



# ============================================================
# 2. Helpers
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def get_cv_accuracy(model, X_sel, y):
    scores = cross_val_score(model, X_sel, y, cv=cv, scoring='accuracy')
    return round(scores.mean(), 4)

def extract_shap_importance(shap_values, feat_cols):
    if isinstance(shap_values, list):
        arr = shap_values[1]
    else:
        arr = shap_values
    arr = np.array(arr)
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    return pd.Series(np.abs(arr).mean(axis=0), index=feat_cols)

def fa_importance(X_input, cols):
    """
    Pure FA-based feature scoring — NO transform, NO RF, NO mean.
    1. Fit FeatureAgglomeration — assigns each feature to a cluster via labels_
    2. Score each feature by its own variance
    3. Rank all features globally — top N selected across ALL clusters
    """
    fa = FeatureAgglomeration(n_clusters=5)
    fa.fit(X_input)

    # Each feature scored by its own variance — globally ranked, no cluster constraint
    imp = pd.Series(X_input.var().values, index=cols)
    return imp, fa

# ============================================================
# 3. Full Dataset: importance scores for ALL methods
# ============================================================

# RF
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
rf_imp = pd.Series(rf_model.feature_importances_, index=feature_cols)

# RF-SHAP
np.random.seed(42)
idx100 = np.random.choice(len(X), 100, replace=False)
shap_values_rf = shap.TreeExplainer(rf_model).shap_values(X.iloc[idx100])
rf_shap_imp = extract_shap_importance(shap_values_rf, feature_cols)

# XGB
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X, y)
xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols)

# XGB-SHAP
np.random.seed(42)
idx100_xgb = np.random.choice(len(X), 100, replace=False)
shap_values_xgb = shap.TreeExplainer(xgb_model).shap_values(X.iloc[idx100_xgb])
xgb_shap_imp = extract_shap_importance(shap_values_xgb, feature_cols)

# LR
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X, y)
lr_imp = pd.Series(np.abs(lr_model.coef_[0]), index=feature_cols)

# FA — pure FA only, no transform, no RF, no mean
fa_imp, fa_full = fa_importance(X, feature_cols)
print(f"\nFA full cluster assignments : {dict(zip(feature_cols, fa_full.labels_))}")
print(f"FA full feature scores      :\n{fa_imp.sort_values(ascending=False)}")

# HVGS
hvgs_imp = pd.Series(X[feature_cols].var().values, index=feature_cols)

# Spearman
spearman_imp = pd.Series(
    {col: abs(spearmanr(X[col], y)[0]) for col in feature_cols}
)

# ============================================================
# 4. Full Dataset Top 5 + CV Accuracy
# ============================================================
print("\n========== FULL DATASET: TOP 5 ==========")

def top5_and_cv(imp_series, model_fn, label):
    top5 = imp_series.nlargest(5).index.tolist()
    acc  = get_cv_accuracy(model_fn(), X[top5], y)
    print(f"{label}: top5={top5}  CV5={acc}")
    return top5, acc

rf_top5,       cv_rf       = top5_and_cv(rf_imp,       lambda: RandomForestClassifier(random_state=42),                                        'RF')
rf_shap_top5,  cv_rf_shap  = top5_and_cv(rf_shap_imp,  lambda: RandomForestClassifier(random_state=42),                                        'RF-SHAP')
xgb_top5,      cv_xgb      = top5_and_cv(xgb_imp,      lambda: XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False), 'XGB')
xgb_shap_top5, cv_xgb_shap = top5_and_cv(xgb_shap_imp, lambda: XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False), 'XGB-SHAP')
lr_top5,       cv_lr       = top5_and_cv(lr_imp,        lambda: LogisticRegression(random_state=42, max_iter=1000),                             'LR')
fa_top5,       cv_fa       = top5_and_cv(fa_imp,        lambda: RandomForestClassifier(random_state=42),                                        'FA')
hvgs_top5,     cv_hvgs     = top5_and_cv(hvgs_imp,      lambda: RandomForestClassifier(random_state=42),                                        'HVGS')
spearman_top5, cv_spearman = top5_and_cv(spearman_imp,  lambda: RandomForestClassifier(random_state=42),                                        'Spearman')

results = {
    'RF':       {'cv5_acc': cv_rf,       'top5': rf_top5},
    'RF-SHAP':  {'cv5_acc': cv_rf_shap,  'top5': rf_shap_top5},
    'XGB':      {'cv5_acc': cv_xgb,      'top5': xgb_top5},
    'XGB-SHAP': {'cv5_acc': cv_xgb_shap, 'top5': xgb_shap_top5},
    'LR':       {'cv5_acc': cv_lr,       'top5': lr_top5},
    'FA':       {'cv5_acc': cv_fa,       'top5': fa_top5},
    'HVGS':     {'cv5_acc': cv_hvgs,     'top5': hvgs_top5},
    'Spearman': {'cv5_acc': cv_spearman, 'top5': spearman_top5},
}

# ============================================================
# 5. Identify highest feature per algorithm
# ============================================================
print("\n========== HIGHEST FEATURE PER ALGORITHM ==========")

highest_per_method = {
    'RF':       rf_imp.idxmax(),
    'RF-SHAP':  rf_shap_imp.idxmax(),
    'XGB':      xgb_imp.idxmax(),
    'XGB-SHAP': xgb_shap_imp.idxmax(),
    'LR':       lr_imp.idxmax(),
    'FA':       fa_imp.idxmax(),
    'HVGS':     hvgs_imp.idxmax(),
    'Spearman': spearman_imp.idxmax(),
}

for m, f in highest_per_method.items():
    print(f"  {m}: highest feature = '{f}'")

# ============================================================
# 6. Per algorithm: remove highest feature -> reduced set
#    -> re-compute importance on reduced set -> Top 4
# ============================================================
print("\n========== REDUCED DATASET: TOP 4 ==========")

top4_results = {}

for method in ['RF', 'RF-SHAP', 'XGB', 'XGB-SHAP', 'LR', 'FA', 'HVGS', 'Spearman']:

    hf = highest_per_method[method]

    # Build reduced feature list and dataset by removing hf
    reduced_cols = [f for f in feature_cols if f != hf]
    X_red = X[reduced_cols].copy()

    # Hard checks
    assert hf not in reduced_cols,           f"BUG [{method}]: '{hf}' in reduced_cols!"
    assert hf not in X_red.columns.tolist(), f"BUG [{method}]: '{hf}' in X_red!"
    print(f"\n[{method}] highest='{hf}' removed | X_red shape={X_red.shape}")

    if method == 'RF':
        m_r = RandomForestClassifier(random_state=42)
        m_r.fit(X_red, y)
        imp_r = pd.Series(m_r.feature_importances_, index=reduced_cols)

    elif method == 'RF-SHAP':
        m_r = RandomForestClassifier(random_state=42)
        m_r.fit(X_red, y)
        np.random.seed(42)
        idx_r = np.random.choice(len(X_red), 100, replace=False)
        sv_r  = shap.TreeExplainer(m_r).shap_values(X_red.iloc[idx_r])
        imp_r = extract_shap_importance(sv_r, reduced_cols)

    elif method == 'XGB':
        m_r = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        m_r.fit(X_red, y)
        imp_r = pd.Series(m_r.feature_importances_, index=reduced_cols)

    elif method == 'XGB-SHAP':
        m_r = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        m_r.fit(X_red, y)
        np.random.seed(42)
        idx_r = np.random.choice(len(X_red), 100, replace=False)
        sv_r  = shap.TreeExplainer(m_r).shap_values(X_red.iloc[idx_r])
        imp_r = extract_shap_importance(sv_r, reduced_cols)

    elif method == 'LR':
        m_r = LogisticRegression(random_state=42, max_iter=1000)
        m_r.fit(X_red, y)
        imp_r = pd.Series(np.abs(m_r.coef_[0]), index=reduced_cols)

    elif method == 'FA':
        # Pure FA only — NO transform, NO RF, NO mean
        imp_r, fa_r = fa_importance(X_red, reduced_cols)
        print(f"  FA cluster assignments : {dict(zip(reduced_cols, fa_r.labels_))}")
        print(f"  FA feature scores      :\n{imp_r.sort_values(ascending=False)}")

    elif method == 'HVGS':
        imp_r = pd.Series(X_red.var().values, index=reduced_cols)

    elif method == 'Spearman':
        imp_r = pd.Series(
            {col: abs(spearmanr(X_red[col], y)[0]) for col in reduced_cols}
        )

    # Hard check: hf must NOT be in imp_r
    assert hf not in imp_r.index.tolist(), \
        f"BUG [{method}]: '{hf}' in imp_r index!"

    # Select Top 4 globally
    top4 = imp_r.nlargest(4).index.tolist()

    # Hard check: hf must NOT be in top4
    assert hf not in top4, \
        f"BUG [{method}]: '{hf}' in top4={top4}!"

    top4_results[method] = top4
    print(f"  [{method}] Top4: {top4}")

# ============================================================
# 7. Final Verification
# ============================================================
print("\n========== FINAL VERIFICATION ==========")
for method, t4 in top4_results.items():
    hf     = highest_per_method[method]
    status = 'PASS' if hf not in t4 else 'FAIL'
    print(f"[{status}] {method}: highest='{hf}' | top4={t4}")

# ============================================================
# 8. Summary Table & Save
# ============================================================
print("\n========== SUMMARY TABLE ==========")
summary_rows = []
for method in ['RF', 'RF-SHAP', 'XGB', 'XGB-SHAP', 'LR', 'FA', 'HVGS', 'Spearman']:
    summary_rows.append({
        'Method':          method,
        'CV5_Accuracy':    f"{results[method]['cv5_acc']:.4f}",
        'Top5_Features':   ', '.join(results[method]['top5']),
        'Top4_Features':   ', '.join(top4_results[method]),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv('result.csv', index=False, encoding='utf-8-sig')
print("\nSaved to result.csv")