Got it, Sho. Let’s break the script down from top to bottom so you know exactly what each piece is doing and why it’s there.

What this script is

A hybrid classifier for three labels — Complaint, Concern, Neutral — that combines:
	•	semantic text embeddings,
	•	numeric/rule features (loudness, AWS sentiment, keyword hits, similarity),
	•	a multinomial Logistic Regression,
	•	and an optional “rule blending” step that nudges model probabilities toward your rules.

It works with CSV or Excel, saves all artifacts, and provides an inference function.

⸻

Imports, config, and columns
	•	INPUT_FILE, SHEET_NAME, OUTPUT_DIR: where to read/write data and artifacts.
	•	Column names: COL_TEXT_CUST, COL_TEXT_FULL, COL_LABEL, COL_LOUDNESS, COL_SENTIMENT. The model expects these to exist.
	•	Model & thresholds:
	•	EMBED_MODEL: sentence-transformers model (mpnet-base-v2, 768-dim).
	•	MAX_CHARS_CHUNK: transcripts are chunked to this many characters before embedding.
	•	Rule thresholds:
	•	LOUD_HIGH_TH (≥ 96) → “loud” call,
	•	SENT_STRONG_NEG (≤ –3) → strong negative sentiment,
	•	SENT_MOD_NEG_MAX (≤ 0) → weak/moderate negative,
	•	NEU_POS_TH (≥ 0.5) → reasonably positive.
	•	Keyword discovery parameters: n-grams, TF-IDF cap, how many candidates, how many per class, and a similarity floor.
	•	ALPHA_RULE_BLEND: weight used to mix rule one-hots with ML probabilities.
	•	Label maps: LABEL2ID and reverse.

⸻

I/O helpers

read_table(path, sheet)
	•	Auto-detects CSV vs Excel.
	•	For Excel: if you don’t pass a sheet name, it picks the first sheet so you get a DataFrame (not a dict).
	•	Returns a pandas DataFrame.

write_table(df, path)
	•	Writes .xlsx via openpyxl or .csv. Falls back to CSV if no extension.

⸻

Minimal text helpers

chunk_text(text, max_chars)

Splits long transcripts into ~max_chars chunks. We embed chunks individually and then mean-pool. This keeps memory stable and gives better coverage for long calls.

text_series(df)

Chooses the text column in priority order: customer_light then full_light. Ensures we actually have text to embed.

normalize_labels_inplace(df)

Normalizes label strings so the model sees only NEUTRAL, CONCERN, COMPLAINT. It also folds plural variants (“CONCERNS”, “COMPLAINTS”) and null-y values into the canonical set.

prepare_labels(df)

Same normalization, but returns the numeric classes using LABEL2ID.

⸻

Keyword discovery (train-only to avoid leakage)

build_candidate_phrases(texts)
	•	Builds a TfidfVectorizer over the train texts, with 1–3 n-grams, min_df=3, English stopword removal.
	•	Ranks features by total TF-IDF mass across the corpus.
	•	Keeps the top TOP_K_CANDIDATES, stripping pure-stopword items.
	•	Output: a list of high-signal phrases (unigram→trigram) from your domain.

expand_keywords_by_similarity(model, candidates, seed_phrases, top_k)
	•	Encodes seed phrases with the sentence-transformer and averages them → seed centroid.
	•	Encodes all TF-IDF candidates.
	•	Because embeddings are normalized, dot product is cosine similarity.
	•	Takes the top_k candidates closest to the centroid.
	•	Output: (expanded phrases for that class, their embeddings). You do this once for complaints and once for concerns.

This is how you move from a small hand-crafted seed list to a richer, data-driven vocabulary, biased by semantics instead of literal string matching.

⸻

Embeddings & features

embed_transcript(model, text)
	•	Chunk → embed each chunk → mean-pool → 768-dim vector (normalized by the model).
	•	If text is empty, returns zeros.

build_feature_matrix(...)

This is the core feature builder. For a given DataFrame:
	1.	Embedding (E)
Build a 768-dim vector per transcript via embed_transcript. Shape (N, d).
	2.	Keyword counts
For each transcript, count substring hits from the expanded complaint and concern keyword lists. Two floats: comp_counts, con_counts.
	3.	Semantic similarity to keyword centroids
	•	Compute complaint centroid = mean(embeddings of complaint keywords). Same for concern.
	•	For each transcript embedding, compute cosine similarity to each centroid → sims_comp, sims_con. These measure “how complaint-like” or “how concern-like” the whole call is, semantically.
	4.	Numeric inputs
Pull loudnessscore and Max negative customer score into arrays.
	5.	Rule flags (booleans → floats)
	•	rule_complaint is 1 if: loudness high AND sentiment ≤ strong negative AND (has complaint keyword hit OR similarity ≥ SIM_THRESHOLD).
	•	rule_concern is 1 if: sentiment > strong negative and ≤ 0 AND (concern keyword hit OR similarity ≥ SIM_THRESHOLD).
	•	rule_neutral is 1 if: sentiment ≥ NEU_POS_TH AND there are no keyword hits AND similarities are below threshold.
	6.	Scale numeric block
The six numeric features [loud, sent, comp_counts, con_counts, sims_comp, sims_con] get MinMax scaled (fit on train only). Embeddings are not scaled.
	7.	Fuse features
Final design matrix X = [Embedding (d) | 6 scaled numeric | rule_complaint | rule_concern | rule_neutral].
Shape is (N, d + 9), with d=768 for mpnet → (N, 777).

Also returns meta_cols (names for the 9 non-embedding features) and the scaler.

⸻

Rule one-hot and blending

build_rule_onehot(rule_neu, rule_con, rule_comp)

Builds a 3-column array where each row is [neutral_flag, concern_flag, complaint_flag]. Multiple flags can be 1 if multiple rule conditions fired.

blended_predict_proba(clf, X, rule_flags, alpha)
	•	Get proba from the logistic regression (shape (N, 3)).
	•	Add alpha * rule_flags.
	•	Renormalize rows to sum to 1.
Effect: rules “push” probability mass toward classes where rules fired. alpha controls the push. 0 = no blending; larger alpha = stronger rule influence.

⸻

Training / evaluation (main)
	1.	Load data via read_table, including first-sheet default for Excel.
	2.	Check required columns and normalize labels.
	3.	Stratified 70/15/15 split on labels into train/val/test.
	4.	Show class distributions (quick sanity check).
	5.	Load embedding model (mpnet).
	6.	Build seed lists (you already fed in domain phrases from your screenshots), then:
	•	build TF-IDF candidates from train only,
	•	expand to complaint/concern keyword lists by semantic similarity.
	•	write both lists to JSON artifacts (so inference can reuse the vocabulary).
	7.	Build features for train/val/test with the same scaler (fit on train).
	8.	Map labels to ints.
	9.	Train classifier
Multinomial Logistic Regression with class_weight="balanced" to compensate for imbalance.
	10.	Evaluation A – Pure ML
Reports on val/test using only the model (no blending). This reflects the value of embeddings+features where rules are inputs, not decision overrides.
	11.	Evaluation B – Hybrid blending
Builds rule one-hots from the last three columns of X and blends them with alpha. Reports again.
	12.	Save artifacts
	•	Model (hybrid_logreg.joblib)
	•	Scaler (numeric_scaler.joblib)
	•	Keyword lists (*_keywords.json)
	•	Embedding model name (embed_model.txt)
	•	Label mapping (label_mapping.json)
	•	Meta column names (meta_cols.json)

You now have everything you need to score new files.

⸻

Inference

inference(input_path, artifacts_dir, output_path, blend_with_rules, alpha, sheet)
	1.	Read the new table (CSV/Excel). Normalize labels if present (ok if not).
	2.	Load artifacts, embedding model name, and keyword lists.
	3.	Recompute keyword centroid embeddings (fast, avoids saving extra arrays).
	4.	Build features with the trained scaler.
	5.	Get probabilities from the model. If blend_with_rules=True, apply the same blending.
	6.	Write out predictions and class probabilities to Excel/CSV.

⸻

Why this design works
	•	No leakage: all keyword discovery (TF-IDF → candidates → semantic expansion) happens on train only.
	•	Context + rules: embeddings capture meaning; numeric/rule features bring in pragmatic signals (volume, sentiment, domain tags).
	•	Interpretability: you can inspect counts, similarities, and rule flags per record to see why a call went to a class.
	•	Switchable enforcement: when you need recall, lower ALPHA_RULE_BLEND (or skip blending); when you need precision, raise it.

⸻

Where to tune (fast wins)
	•	ALPHA_RULE_BLEND: lower to regain complaint recall; sweep on the validation set.
	•	Thresholds: LOUD_HIGH_TH, SENT_STRONG_NEG, SIM_THRESHOLD (lower similarity threshold → more rule hits).
	•	Oversampling: add RandomOverSampler() on X_train to boost complaint recall.
	•	Classifier: try XGBoost/LightGBM on the same X if LR tops out.

That’s the full picture. If you want, I can annotate the code with inline comments at every step or drop in a tiny alpha-sweep block you can paste under main() to print the best blend weight for complaint F1/recall.


First Run:
Nice — thanks for the screenshots. Here’s a clear read on what the numbers mean, what’s working, what’s not, and exactly what to try next (with small code snippets you can paste and run).

⸻

Quick summary of the results you posted
	•	Overall accuracy ~ 0.70. Reasonable for a first hybrid system on a small 3k dataset.
	•	Neutral: excellent recall (1.00) and good F1 (~0.85–0.88). The model is conservative and captures neutral cases reliably.
	•	Concern: solid overall (precision ~0.69–0.81, recall ~0.65–0.87, F1 ~0.72–0.77). Good performance.
	•	Complaint: main pain point.
	•	Pure-ML: complaint recall ≈ 0.74, precision ≈ 0.60, F1 ≈ 0.66–0.67.
	•	Hybrid (rules blended): complaint recall drops to ~0.35–0.37, precision rises to ~0.66–0.71, F1 drops to ~0.46–0.49.

Interpretation: your rule blending is making complaint predictions more precise but far less sensitive. Rules are filtering out many complaints the pure ML would have caught. That’s a precision↔recall tradeoff — rules are conservative.

⸻

Which outcome is better?

Depends on business priorities:
	•	If recall (finding as many complaints as possible) is critical (e.g., regulatory detection), the hybrid blending is hurting you — revert toward the ML or loosen rules.
	•	If precision (minimize false alarms) matters more, hybrid is helping.

Most production setups prefer high recall for complaints, then triage false positives downstream. So I recommend restore/improve complaint recall while keeping precision reasonable.

⸻

Concrete steps to improve (ordered by impact)

1) Tune the rule blend weight (quick, high ROI)

Lower ALPHA_RULE_BLEND so ML gets more say. Or find alpha that maximizes complaint recall / F1 via grid search.

# quick alpha sweep (after training clf and preparing X_val, rule_onehot_val)
import numpy as np
from sklearn.metrics import f1_score

alphas = np.linspace(0.0, 0.9, 10)
best = None
for a in alphas:
    proba = blended_predict_proba(clf, X_val, rule_onehot_val, alpha=a)
    ypred = proba.argmax(axis=1)
    f1_compl = f1_score(y_val, ypred, labels=[2], average='macro')  # label 2 = COMPLAINT
    print(a, f1_compl)
    if best is None or f1_compl > best[0]:
        best = (f1_compl, a)
print("best alpha for complaint f1:", best)

If you want complaint recall specifically, replace f1_score with recall_score.

2) Make rules less strict (easy and effective)
	•	Reduce LOUD_HIGH_TH (e.g., from 96 → 92) or reduce SENT_STRONG_NEG strictness (e.g., -3 → -2.5).
	•	Reduce SIM_THRESHOLD so semantic matches are looser.

Small adjustments -> big recovery in recall.

3) Use rules as features only (don’t override/boost probabilities)

Instead of adding rule-onehot to final probabilities, leave rules in the feature vector and rely on the classifier to weight them. You already include them; the harmful step was blended_predict_proba. Consider dropping the blending and use classifier output (or use minimal blend).

4) Rebalance / oversample Complaint class

Complaint support ~100 vs Concern ~150. Try oversampling complaints in training (SMOTE or simple random oversample) — helps recall.

Example using imbalanced-learn (works on dense X):

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
clf.fit(X_res, y_res)

If SMOTE on high-dim embeddings is unstable, use random oversampling:

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

5) Try a stronger classifier (XGBoost / LightGBM)

Trees often handle heterogeneous features and class imbalance better than LR.

import xgboost as xgb
clf_x = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
clf_x.fit(X_train, y_train)

Tune scale_pos_weight for complaint class (or use class_weight parameter in LightGBM).

6) Error analysis — inspect false negatives & false positives

Look at a sample of complaints that were missed and complaints that were predicted but wrong. This tells whether rules missed synonyms or embeddings are weak for short transcripts.

import pandas as pd
proba_test = clf.predict_proba(X_test)  # or blended
preds = np.argmax(proba_test, axis=1)
df_test = test_df.copy().reset_index(drop=True)
df_test['pred'] = preds
df_test['pred_label'] = df_test['pred'].map(ID2LABEL)
df_test['true_label'] = y_test.map(lambda x: ID2LABEL[x])
# false negatives for complaint:
fn = df_test[(df_test['true_label']=='COMPLAINT') & (df_test['pred_label']!='COMPLAINT')]
fn[['customer_light','true_label','pred_label']].sample(10, random_state=42)

This will quickly show what your rules are missing.

7) Expand/clean keyword seeds (data-driven)

You already generate keywords from TF-IDF + embedding similarity. Manually review the top 100 complaint keywords, add domain phrases (refund, escalation wording), and re-run. That often improves semantic coverage.

8) Calibrate probabilities (if you use thresholds)

If you tune thresholds for class assignment, calibrate classifier with CalibratedClassifierCV.

from sklearn.calibration import CalibratedClassifierCV
cal = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
cal.fit(X_val, y_val)
proba_cal = cal.predict_proba(X_test)

9) Cross-validate & grid-search

Grid over hyperparameters (alpha, SIM_THRESHOLD, loudness threshold, classifier params) using CV to avoid overfitting to a single train/val split.

⸻

Diagnostics to run now (copy/paste)
	1.	Confusion matrix and class-wise errors:

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_tst_blend, labels=[0,1,2])
print(cm)

	2.	Per-class precision/recall/threshold sweep for complaint:

from sklearn.metrics import precision_recall_curve
proba_comp = proba_tst_blend[:, 2]  # complaint prob
prec, rec, th = precision_recall_curve((y_test==2).astype(int), proba_comp)
# find threshold for recall >= 0.8
idx = np.where(rec >= 0.8)[0]
if len(idx): print("threshold for recall>=0.8:", th[idx[0]])

	3.	Alpha sweep (already above) — pick alpha that gives acceptable recall-F1 tradeoff.

⸻

Recommended prioritized plan (what to try in this order)
	1.	Alpha sweep and pick alpha that recovers complaint F1 to an acceptable level.
	2.	Loosen rule thresholds slightly (LOUD_HIGH_TH → 92, SENT_STRONG_NEG → -2.5, SIM_THRESHOLD → 0.30).
	3.	Stop blending (use classifier raw probs) and/or use small alpha (0.1–0.2).
	4.	Run oversampling (RandomOverSampler) for complaint class and retrain.
	5.	If still low recall: inspect false negatives, expand complaint seed phrases and regenerate keywords.
	6.	Try XGBoost or LightGBM on fused features.
	7.	Add calibrated thresholds if you require a deterministic recall target.

⸻

Final take

Your current baseline (~0.70 accuracy) is solid. The hybrid system improved precision for complaints but at the cost of halving recall — that’s reversible and tunable. The fastest, highest-value moves: reduce the blending weight, relax rule thresholds, and/or oversample complaints. Then run a focused error analysis and expand complaint keywords.

If you want, I can:
	•	produce a small script that does the alpha sweep + thresholding and prints the best setting for complaint recall/F1,
	•	or implement automatic oversampling and re-run training in your existing code and return the new reports.

Which one should I run (I’ll generate the code for you now)?


Second Run:

Got it, Sho. Let’s break the script down from top to bottom so you know exactly what each piece is doing and why it’s there.

What this script is

A hybrid classifier for three labels — Complaint, Concern, Neutral — that combines:
	•	semantic text embeddings,
	•	numeric/rule features (loudness, AWS sentiment, keyword hits, similarity),
	•	a multinomial Logistic Regression,
	•	and an optional “rule blending” step that nudges model probabilities toward your rules.

It works with CSV or Excel, saves all artifacts, and provides an inference function.

⸻

Imports, config, and columns
	•	INPUT_FILE, SHEET_NAME, OUTPUT_DIR: where to read/write data and artifacts.
	•	Column names: COL_TEXT_CUST, COL_TEXT_FULL, COL_LABEL, COL_LOUDNESS, COL_SENTIMENT. The model expects these to exist.
	•	Model & thresholds:
	•	EMBED_MODEL: sentence-transformers model (mpnet-base-v2, 768-dim).
	•	MAX_CHARS_CHUNK: transcripts are chunked to this many characters before embedding.
	•	Rule thresholds:
	•	LOUD_HIGH_TH (≥ 96) → “loud” call,
	•	SENT_STRONG_NEG (≤ –3) → strong negative sentiment,
	•	SENT_MOD_NEG_MAX (≤ 0) → weak/moderate negative,
	•	NEU_POS_TH (≥ 0.5) → reasonably positive.
	•	Keyword discovery parameters: n-grams, TF-IDF cap, how many candidates, how many per class, and a similarity floor.
	•	ALPHA_RULE_BLEND: weight used to mix rule one-hots with ML probabilities.
	•	Label maps: LABEL2ID and reverse.

⸻

I/O helpers

read_table(path, sheet)
	•	Auto-detects CSV vs Excel.
	•	For Excel: if you don’t pass a sheet name, it picks the first sheet so you get a DataFrame (not a dict).
	•	Returns a pandas DataFrame.

write_table(df, path)
	•	Writes .xlsx via openpyxl or .csv. Falls back to CSV if no extension.

⸻

Minimal text helpers

chunk_text(text, max_chars)

Splits long transcripts into ~max_chars chunks. We embed chunks individually and then mean-pool. This keeps memory stable and gives better coverage for long calls.

text_series(df)

Chooses the text column in priority order: customer_light then full_light. Ensures we actually have text to embed.

normalize_labels_inplace(df)

Normalizes label strings so the model sees only NEUTRAL, CONCERN, COMPLAINT. It also folds plural variants (“CONCERNS”, “COMPLAINTS”) and null-y values into the canonical set.

prepare_labels(df)

Same normalization, but returns the numeric classes using LABEL2ID.

⸻

Keyword discovery (train-only to avoid leakage)

build_candidate_phrases(texts)
	•	Builds a TfidfVectorizer over the train texts, with 1–3 n-grams, min_df=3, English stopword removal.
	•	Ranks features by total TF-IDF mass across the corpus.
	•	Keeps the top TOP_K_CANDIDATES, stripping pure-stopword items.
	•	Output: a list of high-signal phrases (unigram→trigram) from your domain.

expand_keywords_by_similarity(model, candidates, seed_phrases, top_k)
	•	Encodes seed phrases with the sentence-transformer and averages them → seed centroid.
	•	Encodes all TF-IDF candidates.
	•	Because embeddings are normalized, dot product is cosine similarity.
	•	Takes the top_k candidates closest to the centroid.
	•	Output: (expanded phrases for that class, their embeddings). You do this once for complaints and once for concerns.

This is how you move from a small hand-crafted seed list to a richer, data-driven vocabulary, biased by semantics instead of literal string matching.

⸻

Embeddings & features

embed_transcript(model, text)
	•	Chunk → embed each chunk → mean-pool → 768-dim vector (normalized by the model).
	•	If text is empty, returns zeros.

build_feature_matrix(...)

This is the core feature builder. For a given DataFrame:
	1.	Embedding (E)
Build a 768-dim vector per transcript via embed_transcript. Shape (N, d).
	2.	Keyword counts
For each transcript, count substring hits from the expanded complaint and concern keyword lists. Two floats: comp_counts, con_counts.
	3.	Semantic similarity to keyword centroids
	•	Compute complaint centroid = mean(embeddings of complaint keywords). Same for concern.
	•	For each transcript embedding, compute cosine similarity to each centroid → sims_comp, sims_con. These measure “how complaint-like” or “how concern-like” the whole call is, semantically.
	4.	Numeric inputs
Pull loudnessscore and Max negative customer score into arrays.
	5.	Rule flags (booleans → floats)
	•	rule_complaint is 1 if: loudness high AND sentiment ≤ strong negative AND (has complaint keyword hit OR similarity ≥ SIM_THRESHOLD).
	•	rule_concern is 1 if: sentiment > strong negative and ≤ 0 AND (concern keyword hit OR similarity ≥ SIM_THRESHOLD).
	•	rule_neutral is 1 if: sentiment ≥ NEU_POS_TH AND there are no keyword hits AND similarities are below threshold.
	6.	Scale numeric block
The six numeric features [loud, sent, comp_counts, con_counts, sims_comp, sims_con] get MinMax scaled (fit on train only). Embeddings are not scaled.
	7.	Fuse features
Final design matrix X = [Embedding (d) | 6 scaled numeric | rule_complaint | rule_concern | rule_neutral].
Shape is (N, d + 9), with d=768 for mpnet → (N, 777).

Also returns meta_cols (names for the 9 non-embedding features) and the scaler.

⸻

Rule one-hot and blending

build_rule_onehot(rule_neu, rule_con, rule_comp)

Builds a 3-column array where each row is [neutral_flag, concern_flag, complaint_flag]. Multiple flags can be 1 if multiple rule conditions fired.

blended_predict_proba(clf, X, rule_flags, alpha)
	•	Get proba from the logistic regression (shape (N, 3)).
	•	Add alpha * rule_flags.
	•	Renormalize rows to sum to 1.
Effect: rules “push” probability mass toward classes where rules fired. alpha controls the push. 0 = no blending; larger alpha = stronger rule influence.

⸻

Training / evaluation (main)
	1.	Load data via read_table, including first-sheet default for Excel.
	2.	Check required columns and normalize labels.
	3.	Stratified 70/15/15 split on labels into train/val/test.
	4.	Show class distributions (quick sanity check).
	5.	Load embedding model (mpnet).
	6.	Build seed lists (you already fed in domain phrases from your screenshots), then:
	•	build TF-IDF candidates from train only,
	•	expand to complaint/concern keyword lists by semantic similarity.
	•	write both lists to JSON artifacts (so inference can reuse the vocabulary).
	7.	Build features for train/val/test with the same scaler (fit on train).
	8.	Map labels to ints.
	9.	Train classifier
Multinomial Logistic Regression with class_weight="balanced" to compensate for imbalance.
	10.	Evaluation A – Pure ML
Reports on val/test using only the model (no blending). This reflects the value of embeddings+features where rules are inputs, not decision overrides.
	11.	Evaluation B – Hybrid blending
Builds rule one-hots from the last three columns of X and blends them with alpha. Reports again.
	12.	Save artifacts
	•	Model (hybrid_logreg.joblib)
	•	Scaler (numeric_scaler.joblib)
	•	Keyword lists (*_keywords.json)
	•	Embedding model name (embed_model.txt)
	•	Label mapping (label_mapping.json)
	•	Meta column names (meta_cols.json)

You now have everything you need to score new files.

⸻

Inference

inference(input_path, artifacts_dir, output_path, blend_with_rules, alpha, sheet)
	1.	Read the new table (CSV/Excel). Normalize labels if present (ok if not).
	2.	Load artifacts, embedding model name, and keyword lists.
	3.	Recompute keyword centroid embeddings (fast, avoids saving extra arrays).
	4.	Build features with the trained scaler.
	5.	Get probabilities from the model. If blend_with_rules=True, apply the same blending.
	6.	Write out predictions and class probabilities to Excel/CSV.

⸻

Why this design works
	•	No leakage: all keyword discovery (TF-IDF → candidates → semantic expansion) happens on train only.
	•	Context + rules: embeddings capture meaning; numeric/rule features bring in pragmatic signals (volume, sentiment, domain tags).
	•	Interpretability: you can inspect counts, similarities, and rule flags per record to see why a call went to a class.
	•	Switchable enforcement: when you need recall, lower ALPHA_RULE_BLEND (or skip blending); when you need precision, raise it.

⸻

Where to tune (fast wins)
	•	ALPHA_RULE_BLEND: lower to regain complaint recall; sweep on the validation set.
	•	Thresholds: LOUD_HIGH_TH, SENT_STRONG_NEG, SIM_THRESHOLD (lower similarity threshold → more rule hits).
	•	Oversampling: add RandomOverSampler() on X_train to boost complaint recall.
	•	Classifier: try XGBoost/LightGBM on the same X if LR tops out.

That’s the full picture. If you want, I can annotate the code with inline comments at every step or drop in a tiny alpha-sweep block you can paste under main() to print the best blend weight for complaint F1/recall.


Results:

Nice screenshot — let’s walk through that classification-report table line-by-line and make the numbers concrete so you can act on them.

I’ll first define each metric, then interpret the values in your table, and finally give an exact, approximate count breakdown (how many true/false predictions that implies) and what each tells you to do next.

⸻

What the columns mean
	•	precision = TP / (TP + FP)
Of all examples the model predicted this class, how many were actually that class. High precision → few false alarms.
	•	recall = TP / (TP + FN)
Of all actual examples of that class, how many did the model find. High recall → few misses.
	•	f1-score = harmonic mean(precision, recall) = 2·(P·R)/(P+R).
Single-number balance of precision and recall.
	•	support = number of ground-truth examples of that class in the evaluated set.

Below the class rows you also see:
	•	accuracy = (total correct) / (total samples).
	•	macro avg = unweighted average of the three class metrics (treats each class equally).
	•	weighted avg = average weighted by each class’s support (reflects dataset imbalance).

⸻

Your table — plain English interpretation

Classes and supports (from the table):
	•	NEUTRAL: support = 23
	•	CONCERN: support = 153
	•	COMPLAINT: support = 100
Total = 276 samples.

Per-class summary (from your table):
	1.	NEUTRAL — precision 0.74, recall 1.00, f1 0.85, support 23
	•	The model found every neutral example (recall 1.00).
	•	But ~26% of what it labeled “neutral” were actually other classes (precision 0.74) → some false positives.
	2.	CONCERN — precision 0.81, recall 0.65, f1 0.72, support 153
	•	Good precision: of predictions labeled “concern,” ~81% are correct.
	•	But recall is 0.65: it misses ~35% of true concerns.
	3.	COMPLAINT — precision 0.60, recall 0.74, f1 0.66, support 100
	•	The model finds ~74% of true complaints (recall 0.74).
	•	But precision is only 0.60: many predicted complaints are actually something else (false alarms).

Overall accuracy 0.71 — the model made correct predictions for ~71% of the 276 samples (≈196 correct, ≈80 incorrect).

Macro averages (treat each class equally) are about precision 0.72, recall 0.80, f1 0.75.
Weighted averages (accounting for the 153 concerns dominating the set) are precision ~0.73, recall ~0.71, f1 ~0.71.

⸻

Convert those ratios into actual counts (approximate)

I’ll use the definitions P = TP/(TP+FP) and R = TP/(TP+FN) to derive approximate counts.

Calculations (rounded)
	•	Neutral (support 23)
	•	TP ≈ recall × support = 1.00 × 23 = 23 (so FN = 0)
	•	Predicted as Neutral (PP) ≈ TP / precision = 23 / 0.74 ≈ 31 → so FP ≈ 31 − 23 = 8
	•	Concern (support 153)
	•	TP ≈ 0.65 × 153 ≈ 99
	•	FN = 153 − 99 ≈ 54 (missed concerns)
	•	Predicted as Concern (PP) ≈ 99 / 0.81 ≈ 123 → FP ≈ 123 − 99 = 24
	•	Complaint (support 100)
	•	TP ≈ 0.74 × 100 = 74
	•	FN = 100 − 74 = 26 (missed complaints)
	•	Predicted as Complaint (PP) ≈ 74 / 0.60 ≈ 123 → FP ≈ 123 − 74 = 49

Double-check totals:
	•	Sum of TP ≈ 23 + 99 + 74 = 196 correct predictions (matches accuracy ≈ 0.71 × 276 = 195.96).
	•	Sum of FP ≈ 8 + 24 + 49 = 81 false positives (≈ total incorrect because FP across classes = FN across classes).

Takeaway from counts:
	•	The model never misses Neutral (good), but it overpredicts Neutral a bit (8 false positives).
	•	For Concerns, it misses ~54 of 153 (so 35% of concerns are not labeled as concern).
	•	For Complaints, it misses 26 of 100 — so complaint recall is decent (74%), but precision is low: about 49 predicted complaints are false alarms.

⸻

What these patterns mean, practically
	•	If your priority is not missing complaints (high recall): current recall = 0.74. You’re catching most but not all — roughly 1 in 4 complaints is missed. Change needed: favor recall (reduce blending weight, oversample complaint, loosen rules).
	•	If your priority is not raising false complaint alerts (high precision): current precision = 0.60, meaning ~40% of predicted complaints are false positives — that’s high noise for an operations team.
	•	Concerns show an asymmetry: high precision, low recall. The model is cautious calling something a concern (fewer false alarms) but therefore misses many real concerns.

⸻

Quick actions suggested from the matrix
	•	If you want higher complaint recall (catch more complaints):
	•	Lower rule blending (ALPHA_RULE_BLEND) so ML has more influence, or remove blending and use the model alone (rules are already features).
	•	Oversample complaints in training (RandomOverSampler or SMOTE) so model learns complaint patterns better.
	•	Loosen complaint-related rule thresholds (e.g., SIM_THRESHOLD, SENT_STRONG_NEG) so the rules don’t overly filter candidates.
	•	If you want higher complaint precision (fewer false alarms):
	•	Keep/increase blending or tighten rules (but that will reduce recall).
	•	Add secondary heuristics to filter predicted complaints (for example require complaint keyword and low sentiment).
	•	For concerns, increase recall by adding more concern seed phrases, expanding TF-IDF candidates and retraining, or using oversampling for concern class.

⸻

One final note about averages
	•	Macro avg is useful when you care about all classes equally (it shows the model is relatively balanced across classes).
	•	Weighted avg reflects expected performance on your actual dataset mix (here dominated by concerns).

⸻

If you want, I’ll:
	•	compute and print the exact confusion matrix from your test fold (needs the true/pred arrays), or
	•	run an alpha sweep on blending to show the precision/recall tradeoff for complaints and pick an alpha that hits a target recall (say 0.85) — pasteable code I can generate for you. Which one should I do next?