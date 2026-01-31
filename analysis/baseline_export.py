#!/usr/bin/env python
# coding: utf-8

# In[1]:


# A better way to read the same file, handling BOM and end-of-line truncation

import re
import pandas as pd
from pathlib import Path

path = "arrets_ben.csv"
out_parquet = "arrets_ben.parquet"

# --- 1) Read header + dash line, remove BOM automatically ---
with open(path, encoding="utf-8-sig") as f:
    header_line = f.readline().rstrip("\n")
    dash_line   = f.readline().rstrip("\n")

# --- 2) Infer column spans from runs of dashes ---
colspecs = [(m.start(), m.end()) for m in re.finditer(r"-+", dash_line)]

# Make the last column go to end-of-line to avoid truncation
colspecs[-1] = (colspecs[-1][0], None)

# --- 3) Slice column names from the (BOM-stripped) header ---
raw_names = [header_line[s:] if e is None else header_line[s:e] for s, e in colspecs]
names = []
seen = {}
for nm in map(str.strip, raw_names):
    seen[nm] = seen.get(nm, -1) + 1
    names.append(nm if seen[nm] == 0 else f"{nm}_{seen[nm]}")

print("Detected columns:", len(names))
print(names[:10], "...")

# --- 4) Read the data as fixed-width (skip header + dashes) ---
df = pd.read_fwf(
    path,
    colspecs=colspecs,
    names=names,
    skiprows=2,
    na_values=["NULL"],
    encoding="utf-8-sig",
)
print(df.shape)
print(df.head(3))


# ### Build Segment

# In[2]:


# =========================
# Build `seg` (prev stop -> current stop) from df
# - single-source targets:
#     link_s  := TempsInterArretRealise               (actual link runtime)
#     dwell_s := DTSortieFenetreArretReal - DTEntreeFenetreArretReal (window-based dwell)
# - punctuality signals (if present):
#     E_i, E_prev, dE := EcartDepart and its lag
# - time anchor for features:
#     link_start_time := previous stop's window-exit (fallback to ATP enter, then schedule depart)
# =========================
import numpy as np
import pandas as pd

seg_src = df.copy()

# --- 0) Parse datetimes (only if columns exist) ---
time_cols = [
    "DTDepartTheo","DTArriveeTheo",
    "DTEntreeFenetreArretReal","DTSortieFenetreArretReal",
    "DTEntreeArretAtp","DTSortieArretAtp",
    "DTMarquageArretTheo","DTMarquageArretReal",
    "HOuverturePortesReal","HFermetureportesReal"
]
for c in time_cols:
    if c in seg_src.columns:
        seg_src[c] = pd.to_datetime(seg_src[c], errors="coerce")

# --- 1) Basic casting / trimming ---
# IDs & ordering
if "RangArretAsc" in seg_src.columns:
    seg_src["RangArretAsc"] = pd.to_numeric(seg_src["RangArretAsc"], errors="coerce")
if "C_Ligne" in seg_src.columns:
    seg_src["C_Ligne"] = pd.to_numeric(seg_src["C_Ligne"], errors="coerce")

# stop codes / direction as clean strings
seg_src["CodeLong"] = seg_src.get("CodeLong", "").astype(str).str.strip()
if "C_SensAppl" in seg_src.columns:
    seg_src["C_SensAppl"] = seg_src["C_SensAppl"].astype(str).str.strip()

# distances & counts (if exist)
for c in ["DistanceInterArret","TempsInterArretRealise","EcartDepart","NbMontees","NbDescentes"]:
    if c in seg_src.columns:
        seg_src[c] = pd.to_numeric(seg_src[c], errors="coerce")

# --- 2) Sort and build previous-stop columns (within a trip/course) ---
by = ["IdCourse","RangArretAsc"] if "IdCourse" in seg_src.columns else ["C_Ligne","DateCourse","RangArretAsc"]
seg_src = seg_src.sort_values(by, kind="mergesort")

grp = seg_src.groupby("IdCourse") if "IdCourse" in seg_src.columns else seg_src.groupby(["C_Ligne","DateCourse"])

seg_src["prev_CodeLong"] = grp["CodeLong"].shift(1)

# previous-window exit (our preferred "actual depart" at upstream stop)
if {"DTEntreeFenetreArretReal","DTSortieFenetreArretReal"}.issubset(seg_src.columns):
    seg_src["prev_DT_win_out"] = grp["DTSortieFenetreArretReal"].shift(1)
    seg_src["prev_DT_win_in"]  = grp["DTEntreeFenetreArretReal"].shift(1)
else:
    seg_src["prev_DT_win_out"] = np.nan
    seg_src["prev_DT_win_in"]  = np.nan

# fallbacks if window missing: ATP enter, then scheduled depart
seg_src["prev_DT_atp_in"]   = grp["DTEntreeArretAtp"].shift(1) if "DTEntreeArretAtp" in seg_src.columns else np.nan
seg_src["prev_DT_sched_dep"]= grp["DTDepartTheo"].shift(1)     if "DTDepartTheo" in seg_src.columns else np.nan

# upstream demand / punctuality
for c in ["NbMontees","NbDescentes","EcartDepart"]:
    if c in seg_src.columns:
        seg_src[f"prev_{c}"] = grp[c].shift(1)

# --- 3) Keep only rows that have a previous stop (i.e., valid link) ---
seg = seg_src.dropna(subset=["prev_CodeLong"]).copy()
seg["from_stop"] = seg["prev_CodeLong"].astype(str).str.strip()
seg["to_stop"]   = seg["CodeLong"].astype(str).str.strip()
seg["SegmentKey"] = seg["from_stop"] + "→" + seg["to_stop"]

# --- 4) Targets from single sources ---
# 4a) Actual link runtime: Realise (single source; no cross-source mixing)
seg["link_s"] = seg.get("TempsInterArretRealise")
seg.loc[(seg["link_s"]<=0) | (seg["link_s"]>1800), "link_s"] = np.nan  # basic QC (0/neg, >30min)

# 4b) Dwell time from window timestamps (single source)
if {"DTEntreeFenetreArretReal","DTSortieFenetreArretReal"}.issubset(seg.columns):
    seg["dwell_s"] = (seg["DTSortieFenetreArretReal"] - seg["DTEntreeFenetreArretReal"]).dt.total_seconds()
    seg.loc[(seg["dwell_s"]<0) | (seg["dwell_s"]>900), "dwell_s"] = np.nan  # QC: <0 or >15min
else:
    seg["dwell_s"] = np.nan

# 4c) Punctuality signals (if present)
seg["E_i"]    = pd.to_numeric(seg.get("EcartDepart"), errors="coerce")
seg["E_prev"] = pd.to_numeric(seg.get("prev_EcartDepart"), errors="coerce")
seg["dE"]     = seg["E_i"] - seg["E_prev"]

# --- 5) Upstream dwell used as a feature (window-based) ---
seg["dwell_prev_s"] = (seg["prev_DT_win_out"] - seg["prev_DT_win_in"]).dt.total_seconds()
seg.loc[(seg["dwell_prev_s"]<0) | (seg["dwell_prev_s"]>900), "dwell_prev_s"] = np.nan

# --- 6) Distance feature (static meta; keep raw to avoid mixing assumptions) ---
seg["distance_m"] = pd.to_numeric(seg.get("DistanceInterArret"), errors="coerce")
seg.loc[seg["distance_m"]<0, "distance_m"] = np.nan

# --- 7) Time anchor for features: previous stop's window-exit (fallbacks) ---
seg["link_start_time"] = seg["prev_DT_win_out"]
seg.loc[seg["link_start_time"].isna(), "link_start_time"] = seg["prev_DT_atp_in"]
seg.loc[seg["link_start_time"].isna(), "link_start_time"] = seg["prev_DT_sched_dep"]

# Drop rows without any anchor time
seg = seg[pd.notna(seg["link_start_time"])].copy()

# Calendar & cyclical features
seg["hour"] = seg["link_start_time"].dt.hour
seg["dow"]  = seg["link_start_time"].dt.dayofweek   # 0=Mon .. 6=Sun
seg["is_weekend"] = seg["dow"].isin([5,6]).astype(int)
seg["hour_sin"] = np.sin(2*np.pi*seg["hour"]/24.0)
seg["hour_cos"] = np.cos(2*np.pi*seg["hour"]/24.0)
seg["period168"] = seg["dow"]*24 + seg["hour"]      # 0..167

# Optional coarse label (handy for plots)
def assign_period(dt):
    h, d = dt.hour, dt.dayofweek
    if d==5: return "Sat"
    if d==6: return "Sun"
    if 7<=h<9:   return "AM"
    if 9<=h<16:  return "Day"
    if 16<=h<19: return "PM"
    if 19<=h<23: return "Eve"
    return "Other"
seg["period"] = seg["link_start_time"].map(assign_period)

# --- 8) IDs for grouping/filters ---
seg["line"] = seg.get("C_Ligne").astype("Int64").astype(str) if "C_Ligne" in seg.columns else "NA"
seg["dir"]  = seg.get("C_SensAppl", "NA").astype(str)

# demand features from upstream stop (if present)
seg["board_prev"]  = pd.to_numeric(seg.get("prev_NbMontees"),   errors="coerce")
seg["alight_prev"] = pd.to_numeric(seg.get("prev_NbDescentes"), errors="coerce")

# --- 9) Feature list for GBM (E/ΔE) ---
feat_cols = [
    "distance_m","dwell_prev_s","E_prev",
    "board_prev","alight_prev",
    "hour_sin","hour_cos","dow","is_weekend",
    "from_stop","to_stop","line","dir"
]

print(f"[seg] rows: {len(seg):,}")
print("Targets available:",
      f"link_s={seg['link_s'].notna().mean():.2%}",
      f"dwell_s={seg['dwell_s'].notna().mean():.2%}",
      f"E_i={seg['E_i'].notna().mean():.2%}",
      f"E_prev={seg['E_prev'].notna().mean():.2%}",
      f"dE={seg['dE'].notna().mean():.2%}")
print("Feature columns:", feat_cols)


# In[3]:


# Assumes you already have `seg` with:
# - line, from_stop, to_stop
# - link_start_time (datetime)
# - TempsInterArretRealise (actual link runtime)
# - DTEntreeFenetreArretReal, DTSortieFenetreArretReal (window timestamps)

# 1) segment key (include line to avoid cross-line collisions)
seg = seg.copy()
seg["seg3"] = seg["line"].astype(str) + "|" + seg["from_stop"] + "→" + seg["to_stop"]

# 2) actual link runtime (single source: Realise)
seg["link_s"] = pd.to_numeric(seg.get("TempsInterArretRealise"), errors="coerce")

# 3) dwell (single source: window)
if {"DTEntreeFenetreArretReal","DTSortieFenetreArretReal"}.issubset(seg.columns):
    seg["dwell_s"] = (
        seg["DTSortieFenetreArretReal"] - seg["DTEntreeFenetreArretReal"]
    ).dt.total_seconds()
else:
    seg["dwell_s"] = np.nan

# 4) basic QC (leave NaN if missing; remove impossible/outlier tails conservatively)
seg.loc[seg["link_s"]<=0, "link_s"] = np.nan
seg.loc[seg["link_s"]>1800, "link_s"] = np.nan  # 30 min hard cap (tune)
seg.loc[seg["dwell_s"]<0, "dwell_s"] = np.nan
seg.loc[seg["dwell_s"]>900, "dwell_s"] = np.nan  # 15 min hard cap (tune)

# 5) time bins: hour-of-day × day-of-week -> 168 buckets
ts = seg["link_start_time"]
seg["dow"]   = ts.dt.dayofweek      # 0=Mon..6=Sun
seg["hour"]  = ts.dt.hour           # 0..23
seg["period168"] = seg["dow"]*24 + seg["hour"]  # 0..167


# In[4]:


# =========================
# Empirical baselines for link_s / dwell_s
# - train window → fit quantile maps (by levels)
# - test window  → apply with hierarchical fallback
# =========================
import numpy as np
import pandas as pd

# 1) 時間切分（沿用你的 split）
TRAIN_FROM = "2024-10-01"
TRAIN_TO   = "2024-10-31"
TEST_FROM  = "2024-11-01"
TEST_TO    = "2024-12-31"

mask_tr = (seg["link_start_time"]>=pd.to_datetime(TRAIN_FROM)) & (seg["link_start_time"]<=pd.to_datetime(TRAIN_TO))
mask_te = (seg["link_start_time"]>=pd.to_datetime(TEST_FROM))  & (seg["link_start_time"]<=pd.to_datetime(TEST_TO))
train = seg[mask_tr].copy()
test  = seg[mask_te].copy()

# 2) 定義分層層級（由細到粗）
#    你可以依資料量調整；樣本不足時會往下一層回退
LEVELS = [
    ["SegmentKey","period168"],           # 每段×每週168小時
#    ["SegmentKey","is_weekend","hour"],   # 每段×是否週末×小時
#    ["SegmentKey"],                       # 只看段
]
MIN_N = 10  # 每群至少樣本數；不夠就回退

def _fit_level_quantile(train_df, target, by, q):
    g = (train_df
         .dropna(subset=[target])
         .groupby(by)[target]
         .agg(n="size", q=lambda s: float(s.quantile(q))))
    # 只留 n >= MIN_N 的群
    g = g[g["n"] >= MIN_N].reset_index()
    return g  # columns: by... + ["n","q"]

def fit_maps(train_df, target, levels, q):
    """回傳每一層的 quantile 對照表（含 n）與全域保底 quantile。"""
    maps = [ _fit_level_quantile(train_df, target, by, q) for by in levels ]
    global_q = float(train_df[target].dropna().quantile(q)) if train_df[target].notna().any() else 0.0
    return maps, global_q

def apply_maps(apply_df, levels, maps, global_q):
    """依序回填：第一層沒命中 → 第二層 → ... → 全域。也回傳用到哪一層。"""
    pred  = pd.Series(np.nan, index=apply_df.index, dtype=float)
    used  = pd.Series(np.nan, index=apply_df.index)  # 紀錄命中層級索引（0=最細）
    remain = pred.isna()

    for i, by in enumerate(levels):
        if len(maps[i]) == 0 or not remain.any():
            continue
        tmp = (apply_df.loc[remain, by]
               .merge(maps[i], on=by, how="left")["q"])
        hit = tmp.notna().values
        pred.loc[remain[remain].index[hit]] = tmp[hit].values
        used.loc[remain[remain].index[hit]] = i
        remain = pred.isna()

    # 全域保底
    pred.loc[remain] = global_q
    used.loc[remain] = len(levels)  # 全域
    return pred.values, used.values

def make_empirical_baseline(train_df, test_df, target, q, levels=LEVELS):
    maps, gq = fit_maps(train_df, target, levels, q)
    pred, used = apply_maps(test_df, levels, maps, gq)
    return pred, used, maps, gq

# 3) 對 link_s / dwell_s 做 p50 與 p85 baseline
# Link (段間實測行車時間)
b_link50, usedL50, mapsL50, gL50 = make_empirical_baseline(train, test, target="link_s", q=0.50, levels=LEVELS)
b_link85, usedL85, mapsL85, gL85 = make_empirical_baseline(train, test, target="link_s", q=0.85, levels=LEVELS)

# Dwell（站內停留；若 window 不完整可能較稀疏）
b_dwell50, usedD50, mapsD50, gD50 = make_empirical_baseline(train, test, target="dwell_s", q=0.50, levels=LEVELS)
b_dwell85, usedD85, mapsD85, gD85 = make_empirical_baseline(train, test, target="dwell_s", q=0.85, levels=LEVELS)

# 4) 簡單評估（p50 當點估、p85 當覆蓋）
def mae(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan

def coverage(y, qhat):
    m = np.isfinite(y) & np.isfinite(qhat)
    return float(np.mean(y[m] <= qhat[m])) if m.any() else np.nan

print("=== Link_s baselines ===")
print("MAE vs p50:", mae(test["link_s"].to_numpy(float), b_link50))
print("Cov@p85  :", coverage(test["link_s"].to_numpy(float), b_link85))
print("Hit-rate by level (p50):", pd.Series(usedL50).value_counts(normalize=True).sort_index().to_dict())

print("\n=== Dwell_s baselines ===")
print("MAE vs p50:", mae(test["dwell_s"].to_numpy(float), b_dwell50))
print("Cov@p85  :", coverage(test["dwell_s"].to_numpy(float), b_dwell85))
print("Hit-rate by level (p50):", pd.Series(usedD50).value_counts(normalize=True).sort_index().to_dict())

# 5) 產生「empirical deviation」可供後續分析/預測（完全不碰理論）
test = test.copy()
test["link_dev_emp_p50"]  = test["link_s"]  - b_link50   # 正值 = 比常態慢（延）
test["dwell_dev_emp_p50"] = test["dwell_s"] - b_dwell50

# （可選）把 baseline 附回 test，之後做圖或當 ML feature 皆可
test["base_link_p50"]  = b_link50
test["base_link_p85"]  = b_link85
test["base_dwell_p50"] = b_dwell50
test["base_dwell_p85"] = b_dwell85


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# --- helpers ---
def _ensure_link_s(seg):
    if "link_s" not in seg.columns:
        cand = None
        for c in ["TempsInterArretRealise","act_link_s"]:
            if c in seg.columns: cand = c; break
        if cand is None:
            raise ValueError("need link_s / TempsInterArretRealise / act_link_s")
        seg = seg.copy()
        seg["link_s"] = pd.to_numeric(seg[cand], errors="coerce")
    return seg

def _segment_spread_table(s):
    g = s.groupby("SegmentKey")["link_s"]
    tbl = pd.DataFrame({
        "n":      g.size(),
        "median": g.median(),
        "p10":    g.quantile(0.10),
        "p25":    g.quantile(0.25),
        "p75":    g.quantile(0.75),
        "p90":    g.quantile(0.90),
    })
    tbl["IQR"] = tbl["p75"] - tbl["p25"]
    tbl["p90-p10"] = tbl["p90"] - tbl["p10"]
    return tbl

def _count_breaks(ordered_keys):
    """連續性檢查：相鄰段是否 to==next.from"""
    def ends(k): 
        a,b = k.split("→"); return a,b
    br = 0
    for k1, k2 in zip(ordered_keys, ordered_keys[1:]):
        if ends(k1)[1] != ends(k2)[0]:
            br += 1
    return br

def _top_patterns(seg_line_dir, top_k=2, min_share=0.10):
    """
    以每個班次的「完整停靠序列」當作 pattern：
    path = [該班次第一筆 seg 的 from_stop] + list(to_stop)
    回傳 [(pattern_tuple, count, share), ...] (最多 top_k，且 share >= min_share)
    """
    s = seg_line_dir.copy()

    # 用班次內的停靠順序排序：優先用 RangArretAsc；沒有就退回 link_start_time
    if "RangArretAsc" in s.columns:
        s = s.sort_values(["IdCourse", "RangArretAsc"], kind="mergesort")
    else:
        s = s.sort_values(["IdCourse", "link_start_time"], kind="mergesort")

    # 乾淨的 stop id
    if "from_stop" not in s.columns or "to_stop" not in s.columns:
        # 萬一沒有 from/to_stop，就從 SegmentKey 還原
        a = s["SegmentKey"].str.split("→", expand=True)
        s["from_stop"] = a[0].astype(str).str.strip()
        s["to_stop"]   = a[1].astype(str).str.strip()
    else:
        s["from_stop"] = s["from_stop"].astype(str).str.strip()
        s["to_stop"]   = s["to_stop"].astype(str).str.strip()

    # 組出每個班次的完整路徑（含起點）
    def _trip_full_path(g):
        first_from = g["from_stop"].iloc[0]
        to_seq     = g["to_stop"].tolist()
        return tuple([first_from] + to_seq)

    paths = s.groupby("IdCourse", sort=False).apply(_trip_full_path)

    cnt = Counter(paths)
    total = sum(cnt.values()) if cnt else 0
    ranked = [(p, n, n/total) for p, n in cnt.most_common() if total > 0 and (n/total) >= min_share]
    return ranked[:top_k]

def plot_line_by_patterns(seg, seg_pat, line, dir_=None, top_k=2, min_share=0.10,
                          min_n_per_seg=20, per_fig_max_segments=24, figsize=(16,6)):
    # seg 用於畫值（包含 link_s 正值與 QC）
    s_pos = _ensure_link_s(seg)
    s_pos = s_pos[s_pos["line"].astype(str) == str(line)].copy()
    if dir_ is not None:
        s_pos = s_pos[s_pos["dir"].astype(str) == str(dir_)].copy()
    s_pos = s_pos[pd.to_numeric(s_pos["link_s"], errors="coerce") > 0].copy()
    if s_pos.empty:
        print("no data with positive link_s"); return

    # y 軸範圍用 s_pos
    y_lo = float(np.nanpercentile(s_pos["link_s"], 1))
    y_hi = float(np.nanpercentile(s_pos["link_s"], 99))

    # 🔑 pattern 用 seg_pat（不會漏掉起站的第一段）
    s_all = seg_pat[(seg_pat["line"].astype(str)==str(line)) & (seg_pat["from_stop"].notna())].copy()
    if dir_ is not None:
        s_all = s_all[s_all["dir"].astype(str) == str(dir_)].copy()

    pats = _top_patterns(s_all, top_k=top_k, min_share=min_share)
    if not pats:
        print("no dominant patterns; try lowering min_share"); return

    results = {}
    for pi,(pat, n_pat, share) in enumerate(pats, start=1):
        ordered_keys = [f"{a}→{b}" for a,b in zip(pat[:-1], pat[1:])]

        # 只拿來畫值的資料（用有正值 link_s 的 seg）
        sp = s_pos[s_pos["SegmentKey"].isin(set(ordered_keys))].copy()

        # 允許前 1–2 段樣本少也保留（避免起站被 min_n 吃掉）
        vc = sp["SegmentKey"].value_counts()
        head_keep = set(ordered_keys[:2])
        ok = set(vc[vc >= min_n_per_seg].index) | (head_keep & set(vc.index))
        ordered_keys = [k for k in ordered_keys if k in ok]
        sp = sp[sp["SegmentKey"].isin(ordered_keys)]

        if not ordered_keys:
            print(f"pattern#{pi}: all segments < min_n; skip"); continue

        # 斷點統計
        breaks = _count_breaks(ordered_keys)

        # 畫圖（同你原本）
        def _chunk(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        print(f"\nPattern #{pi}: share={share:.1%}, trips={n_pat}, segments_kept={len(ordered_keys)}, breaks={breaks}")

        spread = _segment_spread_table(sp)
        page = 1
        for keys in _chunk(ordered_keys, per_fig_max_segments):
            data = [sp.loc[sp["SegmentKey"]==k, "link_s"].values for k in keys]
            fig, ax = plt.subplots(figsize=figsize)
            ax.boxplot(data, vert=True, showfliers=False, widths=0.6, labels=keys)
            ax.set_ylim(y_lo, y_hi)
            ttl = f"Line {line}"
            if dir_ is not None: ttl += f" | dir={dir_}"
            ttl += f" — pattern #{pi} (page {page})"
            ax.set_title(ttl)
            ax.set_ylabel("link_s (s)")
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            ax.tick_params(axis='x', rotation=75)
            plt.tight_layout(); plt.show()
            page += 1

        results[f"pattern_{pi}"] = {
            "share": share, "trips": n_pat, "ordered_keys": ordered_keys,
            "breaks": breaks, "spread": spread.loc[ordered_keys]
        }
    return results



# ---- 使用例 ----
# 只畫 18 號線 A 向，前 2 個主流 pattern，每個 pattern 只保留樣本數 >= 20 的段
#res18A = plot_line_by_patterns(seg=seg, seg_pat=seg_pat,
#                               line=18, dir_="A",
#                               top_k=2, min_share=0.15,
#                               min_n_per_seg=20, per_fig_max_segments=20, figsize=(17,6))
# 看 pattern #1 的集中/分散表
# res18A["pattern_1"]["spread"]

def extract_patterns(seg_pat, line, dir_=None, min_share=0.10):
    """
    Return a table of dominant stop-order patterns (>= min_share) for a given line[/dir].
    seg_pat must contain: IdCourse, line, dir, from_stop, to_stop (as in your code).
    """
    s = seg_pat.copy()
    s = s[s["line"].astype(str) == str(line)]
    if dir_ is not None:
        s = s[s["dir"].astype(str) == str(dir_)]
    if s.empty:
        return pd.DataFrame(columns=["pattern_id","trips","share","n_segments","start","end","pattern_str","preview"])

    # Order rows within trip: prefer RangArretAsc, fallback to link_start_time
    if "RangArretAsc" in s.columns:
        s = s.sort_values(["IdCourse", "RangArretAsc"], kind="mergesort")
    else:
        s = s.sort_values(["IdCourse", "link_start_time"], kind="mergesort")

    # Ensure from/to exist
    if "from_stop" not in s.columns or "to_stop" not in s.columns:
        a = s["SegmentKey"].str.split("→", expand=True)
        s["from_stop"] = a[0].astype(str).str.strip()
        s["to_stop"]   = a[1].astype(str).str.strip()
    else:
        s["from_stop"] = s["from_stop"].astype(str).str.strip()
        s["to_stop"]   = s["to_stop"].astype(str).str.strip()

    # Build full path per trip: [first_from] + list(to_stop)
    def _trip_full_path(g):
        first_from = g["from_stop"].iloc[0]
        to_seq = g["to_stop"].tolist()
        return tuple([first_from] + to_seq)

    paths = s.groupby("IdCourse", sort=False).apply(_trip_full_path)

    # Count and compute share
    cnt = Counter(paths)
    total = sum(cnt.values())
    if total == 0:
        return pd.DataFrame(columns=["pattern_id","trips","share","n_segments","start","end","pattern_str","preview"])

    rows = []
    # Keep only patterns >= min_share
    ranked = [(p, n, n/total) for p, n in cnt.most_common() if (n/total) >= min_share]
    for i, (pat, n, share) in enumerate(ranked, start=1):
        pat_list = list(pat)
        n_segments = max(len(pat_list)-1, 0)
        start = pat_list[0] if pat_list else ""
        end   = pat_list[-1] if pat_list else ""
        # full string version
        pattern_str = " → ".join(pat_list)
        # short preview (first 3 … last 3, adaptable)
        if len(pat_list) <= 8:
            preview = pattern_str
        else:
            preview = " → ".join(pat_list[:3]) + " → … → " + " → ".join(pat_list[-3:])
        rows.append({
            "pattern_id": i,
            "trips": n,
            "share": round(share, 4),
            "n_segments": n_segments,
            "start": start,
            "end": end,
            "pattern_str": pattern_str,
            "preview": preview
        })

    df = pd.DataFrame(rows).sort_values(["share","trips"], ascending=False, kind="mergesort").reset_index(drop=True)
    return df

# 先到「Drop rows without any anchor time」之前為止，複製一份：
seg_pat = seg_src.dropna(subset=["prev_CodeLong"]).copy()
seg_pat["from_stop"] = seg_pat["prev_CodeLong"].astype(str).str.strip()
seg_pat["to_stop"]   = seg_pat["CodeLong"].astype(str).str.strip()
seg_pat["SegmentKey"] = seg_pat["from_stop"] + "→" + seg_pat["to_stop"]
seg_pat["line"] = seg_src.get("C_Ligne").astype("Int64").astype(str)
seg_pat["dir"]  = seg_src.get("C_SensAppl", "NA").astype(str)


# In[6]:


res18A = plot_line_by_patterns(seg=seg, seg_pat=seg_pat,
                              line=18, dir_="A",
                              top_k=2, min_share=0.15,
                              min_n_per_seg=20, per_fig_max_segments=50, figsize=(17,6))
res18A["pattern_1"]["spread"]


# In[118]:


df_pat = extract_patterns(seg_pat, line=80, dir_="A", min_share=0)
display(df_pat)  # Jupyter, or print(df_pat.to_string(index=False))


# In[149]:


# ==== 0) 只看 2024/10 的視圖 ====
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import Counter

OCT_START = pd.Timestamp("2024-10-01")
NOV_START = pd.Timestamp("2024-10-31")

def make_october_views(seg, seg_src):
    # seg 已有 link_start_time，可直接切
    S = seg[(seg["link_start_time"] >= OCT_START) & (seg["link_start_time"] < NOV_START)].copy()

    # seg_pat 需從 seg_src 做（不看 anchor），時間用「能代表實際日期」的欄位來切
    # 依優先順序挑一個存在的時間欄位
    for tc in ["DTEntreeFenetreArretReal","DTSortieFenetreArretReal",
               "DTEntreeArretAtp","DTSortieArretAtp","DTDepartTheo","DTArriveeTheo"]:
        if tc in seg_src.columns:
            base_time = seg_src[tc]
            break
    else:
        # 最後退：若只有 DateCourse（字串）就轉成日期
        if "DateCourse" in seg_src.columns:
            base_time = pd.to_datetime(seg_src["DateCourse"], errors="coerce")
        else:
            raise ValueError("找不到能用來切 10 月的時間欄位")

    SS = seg_src[(base_time >= OCT_START) & (base_time < NOV_START)].copy()

    # 形成 seg_pat（保持你之前的邏輯）
    SS = SS.sort_values(["IdCourse","RangArretAsc"], kind="mergesort")
    SS["prev_CodeLong"] = SS.groupby("IdCourse")["CodeLong"].shift(1)
    seg_pat = SS.dropna(subset=["prev_CodeLong"]).copy()
    seg_pat["from_stop"] = seg_pat["prev_CodeLong"].astype(str).str.strip()
    seg_pat["to_stop"]   = seg_pat["CodeLong"].astype(str).str.strip()
    seg_pat["SegmentKey"] = seg_pat["from_stop"] + "→" + seg_pat["to_stop"]
    seg_pat["line"] = SS.get("C_Ligne").astype("Int64").astype(str)
    seg_pat["dir"]  = SS.get("C_SensAppl","NA").astype(str)

    return S, seg_pat

# ==== 1) 用 10 月資料找到最常見完整路徑（含起點） ====
def order_keys_by_top_pattern(seg_pat_oct, line, dir_=None, min_share=0.10):
    s = seg_pat_oct.copy()
    s = s[s["line"].astype(str) == str(line)]
    if dir_ is not None:
        s = s[s["dir"].astype(str) == str(dir_)]
    if s.empty:
        return []

    # 以每個班次的完整序列（[first_from] + list(to)）當 pattern
    s = s.sort_values(["IdCourse","RangArretAsc"], kind="mergesort")
    def _trip_full_path(g):
        first_from = g["from_stop"].iloc[0]
        return tuple([first_from] + g["to_stop"].tolist())
    paths = s.groupby("IdCourse", sort=False).apply(_trip_full_path)

    cnt = Counter(paths)
    total = sum(cnt.values())
    if total == 0:
        return []

    pat, n = cnt.most_common(1)[0]
    share = n / total
    if share < min_share:
        # 分享率不夠也先回傳（只是提醒）
        print(f"[warn] top pattern share only {share:.1%} (<{min_share:.0%})")

    ordered_keys = [f"{a}→{b}" for a,b in zip(pat[:-1], pat[1:])]
    return ordered_keys

# ==== 2A) 單一 period168 的 boxplot（看該時段的離散程度） ====
def plot_box_by_period(S_oct, ordered_keys, period168, line, dir_=None,
                       min_n_per_seg=15, figsize=(17,6)):
    s = S_oct.copy()
    s = s[(s["line"].astype(str)==str(line)) & (s["period168"]==period168)]
    if dir_ is not None:
        s = s[s["dir"].astype(str)==str(dir_)]

    # 只留 >0 的 link_s
    s = s[pd.to_numeric(s["link_s"], errors="coerce") > 0]
    if s.empty:
        print("no data in this period"); return

    # 篩掉樣本太少的段（但保留頭兩段，以免起站被吃掉）
    vc = s["SegmentKey"].value_counts()
    head = set(ordered_keys[:2])
    keep = set(vc[vc>=min_n_per_seg].index) | (head & set(vc.index))
    keys = [k for k in ordered_keys if k in keep]
    if not keys:
        print("all segments < min_n in this period"); return

    data = [s.loc[s["SegmentKey"]==k, "link_s"].values for k in keys]

    # y 軸用該 period 的 1–99 百分位，避免離群值拉扯
    y_lo = float(np.nanpercentile(s["link_s"], 1))
    y_hi = float(np.nanpercentile(s["link_s"], 99))

    plt.figure(figsize=figsize)
    bp = plt.boxplot(data, vert=True, showfliers=False, widths=0.6, labels=keys)
    plt.ylim(y_lo, y_hi)
    ttl = f"Line {line}"
    if dir_ is not None: ttl += f" | dir={dir_}"
    ttl += f" — period168={period168}"
    plt.title(ttl, fontsize=16)
    plt.ylabel("link_s (seconds)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.xticks(rotation=70, ha="right")

    # 在 x 標籤下方標註每段樣本數
    for i, k in enumerate(keys, start=1):
        n_i = len(data[i-1])
        plt.text(i, y_lo + 0.02*(y_hi-y_lo), f"n={n_i}", ha="center", va="bottom", fontsize=9, rotation=90)

    plt.tight_layout(); plt.show()

# ==== 2B) 168×segments 熱圖：變異度 (p90 - p10) ====
def plot_heatmap_spread(S_oct, ordered_keys, line, dir_=None, min_n=15, vmax=None, figsize=(18,6)):
    s = S_oct.copy()
    s = s[s["line"].astype(str)==str(line)]
    if dir_ is not None:
        s = s[s["dir"].astype(str)==str(dir_)]
    s = s[pd.to_numeric(s["link_s"], errors="coerce") > 0]
    s = s[s["SegmentKey"].isin(set(ordered_keys))]

    g = s.groupby(["SegmentKey","period168"])["link_s"]
    agg = g.agg(n="size",
                p10=lambda x: np.nanpercentile(x,10),
                p90=lambda x: np.nanpercentile(x,90)).reset_index()
    agg.loc[agg["n"]<min_n, ["p10","p90"]] = np.nan
    agg["spread"] = agg["p90"] - agg["p10"]

    mat = (agg.pivot(index="period168", columns="SegmentKey", values="spread")
              .reindex(index=range(168), columns=ordered_keys))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.to_numpy().T, aspect="auto", origin="upper",  # ← 上在上
                   interpolation="nearest", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="spread (p90 - p10) [s]")
    ax.set_yticks(range(len(ordered_keys)))
    ax.set_yticklabels(ordered_keys, fontsize=9)
    ax.set_xticks(range(0,168,6)); ax.set_xticklabels(range(0,168,6))
    ttl = f"Line {line}" + (f" | dir={dir_}" if dir_ is not None else "") + " — variability heatmap (p90-p10)"
    ax.set_title(ttl, fontsize=16)
    ax.set_xlabel("period168 (Mon00=0 … Sun23=167)")
    ax.set_ylabel("segments (top → bottom)")
    fig.tight_layout(); plt.show()

def plot_heatmap_spread_norm_per100m(S_oct, ordered_keys, line, dir_=None, min_n=15, vmax=None, figsize=(18,6)):
    s = S_oct.copy()
    s = s[(s["line"].astype(str)==str(line)) & (pd.to_numeric(s["link_s"], errors="coerce")>0)]
    if dir_ is not None:
        s = s[s["dir"].astype(str)==str(dir_)]
    s = s[s["SegmentKey"].isin(set(ordered_keys))]
    s = s[pd.to_numeric(s["distance_m"], errors="coerce")>0].copy()

    s["sec_per_100m"] = s["link_s"] / (s["distance_m"]/100.0)

    g = s.groupby(["SegmentKey","period168"])["sec_per_100m"]
    agg = g.agg(n="size",
                p10=lambda x: np.nanpercentile(x,10),
                p90=lambda x: np.nanpercentile(x,90)).reset_index()
    agg.loc[agg["n"]<min_n, ["p10","p90"]] = np.nan
    agg["spread"] = agg["p90"] - agg["p10"]   # 單位：秒/100m

    mat = (agg.pivot(index="period168", columns="SegmentKey", values="spread")
              .reindex(index=range(168), columns=ordered_keys))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.to_numpy().T, aspect="auto", origin="upper",
                   interpolation="nearest", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="spread (p90 - p10) [sec/100m]")
    ax.set_yticks(range(len(ordered_keys))); ax.set_yticklabels(ordered_keys, fontsize=9)
    ax.set_xticks(range(0,168,6)); ax.set_xticklabels(range(0,168,6))
    ax.set_title(f"Line {line}" + (f" | dir={dir_}" if dir_ else "") + " — variability per distance", fontsize=16)
    ax.set_xlabel("period168"); ax.set_ylabel("segments (top → bottom)")
    fig.tight_layout(); plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import colors

# ---- 1) 找「完整停靠序列」的 top patterns（同你現在的 seg_pat 流程）----
def _top_patterns(seg_line_dir, top_k=1, min_share=0.10):
    s = seg_line_dir.copy()
    if "RangArretAsc" in s.columns:
        s = s.sort_values(["IdCourse","RangArretAsc"], kind="mergesort")
    else:
        s = s.sort_values(["IdCourse","link_start_time"], kind="mergesort")
    if ("from_stop" not in s.columns) or ("to_stop" not in s.columns):
        a = s["SegmentKey"].str.split("→", expand=True)
        s["from_stop"] = a[0].astype(str).str.strip()
        s["to_stop"]   = a[1].astype(str).str.strip()
    else:
        s["from_stop"] = s["from_stop"].astype(str).str.strip()
        s["to_stop"]   = s["to_stop"].astype(str).str.strip()

    def _trip_full_path(g):
        first_from = g["from_stop"].iloc[0]
        to_seq     = g["to_stop"].tolist()
        return tuple([first_from] + to_seq)

    paths = s.groupby("IdCourse", sort=False).apply(_trip_full_path)
    cnt = Counter(paths); total = sum(cnt.values()) if cnt else 0
    ranked = [(p, n, n/total) for p, n in cnt.most_common() if total>0 and (n/total)>=min_share]
    return ranked[:top_k]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# --- 跟你現有的一樣：找最常見的完整停靠序列（含起點） ---
def _top_patterns(seg_line_dir, top_k=1, min_share=0.10):
    s = seg_line_dir.copy()
    if "RangArretAsc" in s.columns:
        s = s.sort_values(["IdCourse","RangArretAsc"], kind="mergesort")
    else:
        s = s.sort_values(["IdCourse","link_start_time"], kind="mergesort")
    if ("from_stop" not in s.columns) or ("to_stop" not in s.columns):
        a = s["SegmentKey"].str.split("→", expand=True)
        s["from_stop"] = a[0].astype(str).str.strip()
        s["to_stop"]   = a[1].astype(str).str.strip()
    else:
        s["from_stop"] = s["from_stop"].astype(str).str.strip()
        s["to_stop"]   = s["to_stop"].astype(str).str.strip()

    def _trip_full_path(g):
        first_from = g["from_stop"].iloc[0]
        to_seq     = g["to_stop"].tolist()
        return tuple([first_from] + to_seq)

    paths = s.groupby("IdCourse", sort=False).apply(_trip_full_path)
    cnt = Counter(paths); total = sum(cnt.values()) if cnt else 0
    ranked = [(p, n, n/total) for p, n in cnt.most_common() if total>0 and (n/total)>=min_share]
    return ranked[:top_k]

# --- p50 熱圖（含距離校正版本） ---
def plot_p50_heatmap_with_distance(
    seg, seg_pat, line, dir_="A",
    month_from="2024-10-01", month_to="2024-10-31",
    min_n=10, norm_unit_m=100,
    show_raw=True, show_norm=True,
    cmap_raw="magma", cmap_norm="cividis",
    figsize=(18,6)
):
    # 1) 用 seg_pat 決定 row 順序（不漏掉起站第一段）
    s_pat = seg_pat[(seg_pat["line"].astype(str)==str(line)) & (seg_pat["dir"].astype(str)==str(dir_))].copy()
    pats = _top_patterns(s_pat, top_k=1, min_share=0.05)
    if not pats:
        print("No dominant pattern for this line/dir."); return
    pat, trips, share = pats[0]
    ordered_keys = [f"{a}→{b}" for a,b in zip(pat[:-1], pat[1:])]

    # 2) 取指定月份 + 有效 link_s 的資料
    s = seg[(seg["line"].astype(str)==str(line)) & (seg["dir"].astype(str)==str(dir_))].copy()
    s = s[(s["link_start_time"]>=pd.to_datetime(month_from)) & (s["link_start_time"]<=pd.to_datetime(month_to))]
    s = s[pd.to_numeric(s["link_s"], errors="coerce") > 0]
    if s.empty:
        print("No positive link_s data in given month range."); return

    # 3) 各段×period168 的 p50 與樣本數
    g = (s.groupby(["SegmentKey","period168"])["link_s"]
           .agg(n="size", p50=lambda x: float(np.nanmedian(x)))).reset_index()
    g.loc[g["n"] < min_n, "p50"] = np.nan

    mat_raw = (g.pivot(index="SegmentKey", columns="period168", values="p50")
                 .reindex(index=ordered_keys))
    keep_rows = mat_raw.index[mat_raw.notna().any(axis=1)].tolist()
    mat_raw = mat_raw.loc[keep_rows]

    # 4) 距離校正：拿同一段在 seg 中的距離（中位數），計算 秒/100m
    #    缺距離或<=0 的段一律留白
    dist_per_seg = (s.groupby("SegmentKey")["distance_m"]
                      .median().reindex(mat_raw.index))
    dist_vec = dist_per_seg.to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        mat_norm_vals = mat_raw.to_numpy() / (dist_vec[:, None] / float(norm_unit_m))
    mat_norm = pd.DataFrame(mat_norm_vals, index=mat_raw.index, columns=mat_raw.columns)
    mat_norm[(~np.isfinite(mat_norm)) | (dist_vec[:,None] <= 0)] = np.nan  # 無距離→留白

    # 5) 畫圖：按需求顯示 raw / norm
    n_panels = int(show_raw) + int(show_norm)
    fig, axs = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axs = axs[0]

    # x 軸輔助
    def style_xaxis(ax):
        ax.set_xlim(-0.5, 167.5)
        for d in range(1,7):
            ax.axvline(d*24-0.5, color="k", lw=1, alpha=0.25)
        ax.set_xticks(np.arange(0, 168, 6))
        ax.set_xlabel("period168 (Mon00=0 … Sun23=167)")

    # y 軸（上→下=行駛方向）
    def style_yaxis(ax, idx):
        ax.set_yticks(np.arange(len(idx)))
        ax.set_yticklabels(idx)
        ax.set_ylabel("segments (top → bottom)")

    # 色階範圍：用 1–99 分位，避免極端值
    vmin_raw = np.nanpercentile(mat_raw.values, 1) if show_raw else None
    vmax_raw = np.nanpercentile(mat_raw.values, 99) if show_raw else None
    vmin_norm = np.nanpercentile(mat_norm.values, 1) if show_norm else None
    vmax_norm = np.nanpercentile(mat_norm.values, 99) if show_norm else None

    pane = 0
    if show_raw:
        cmap = plt.get_cmap(cmap_raw).copy(); cmap.set_bad("white")
        im = axs[pane].imshow(mat_raw.values, aspect="auto", origin="upper",
                              interpolation="nearest", cmap=cmap,
                              vmin=vmin_raw, vmax=vmax_raw)
        style_xaxis(axs[pane]); style_yaxis(axs[pane], mat_raw.index)
        cbar = fig.colorbar(im, ax=axs[pane])
        cbar.set_label("median link_s (p50) [s]")
        axs[pane].set_title(f"Line {line} | dir={dir_} — p50 (seconds)  [{month_from[:7]}]")
        pane += 1

    if show_norm:
        cmap = plt.get_cmap(cmap_norm).copy(); cmap.set_bad("white")
        im = axs[pane].imshow(mat_norm.values, aspect="auto", origin="upper",
                              interpolation="nearest", cmap=cmap,
                              vmin=vmin_norm, vmax=vmax_norm)
        style_xaxis(axs[pane]); style_yaxis(axs[pane], mat_norm.index)
        cbar = fig.colorbar(im, ax=axs[pane])
        cbar.set_label(f"median link_s per {norm_unit_m} m (p50) [s/{norm_unit_m}m]")
        axs[pane].set_title(f"Line {line} | dir={dir_} — p50 (sec/{norm_unit_m}m)  [{month_from[:7]}]")

    plt.tight_layout()
    plt.show()

    return {
        "ordered_keys": ordered_keys,
        "pattern_share": share,
        "trips_in_pattern": trips,
        "mat_raw": mat_raw,
        "mat_norm": mat_norm,
        "dist_per_seg": dist_per_seg
    }


# In[152]:


# 只取 2024/10 視圖
S_oct, seg_pat_oct = make_october_views(seg, seg_src)

# 先決定 Line / Dir 的順序（用 10 月 top pattern）
ordered = order_keys_by_top_pattern(seg_pat_oct, line=18, dir_="A", min_share=0.10)
print(len(ordered), ordered[:5], "...")

# (A) 指定某個 period168 看「該時段」的分布
plot_box_by_period(S_oct, ordered, period168= (2*24 + 8),  # 例：Wed 08:00 → 2*24+8=56
                   line=80, dir_="A", min_n_per_seg=10)

# (B) 看整個 168×segments 的變異熱圖
plot_heatmap_spread(S_oct, ordered, line=18, dir_="A", min_n=10, vmax=120)


# In[153]:


plot_heatmap_spread_norm_per100m(S_oct, ordered, line=18, dir_="A", min_n=10, vmax=40.0)


# In[154]:


plot_p50_heatmap_with_distance(
        seg=seg, seg_pat=seg_pat, line=18, dir_="A",
        month_from="2024-10-01", month_to="2024-10-31",
        min_n=10, norm_unit_m=100,
        show_raw=True, show_norm=True,
        cmap_raw="magma", cmap_norm="cividis",
        figsize=(20,6)
)


# In[73]:


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def _top_patterns(seg_line_dir, top_k=1, min_share=0.10):
    s = seg_line_dir.copy()
    if "RangArretAsc" in s.columns:
        s = s.sort_values(["IdCourse","RangArretAsc"], kind="mergesort")
    else:
        s = s.sort_values(["IdCourse","link_start_time"], kind="mergesort")
    if "from_stop" not in s or "to_stop" not in s:
        a = s["SegmentKey"].str.split("→", expand=True)
        s["from_stop"] = a[0].astype(str).str.strip()
        s["to_stop"]   = a[1].astype(str).str.strip()

    def _trip_full_path(g):
        return tuple([g["from_stop"].iloc[0]] + g["to_stop"].tolist())

    paths = s.groupby("IdCourse", sort=False).apply(_trip_full_path)
    from collections import Counter
    cnt = Counter(paths); tot = sum(cnt.values())
    ranked = [(p,n,n/tot) for p,n in cnt.most_common() if tot>0 and n/tot>=min_share]
    return ranked[:top_k]

def plot_p50_heatmap_with_distance(seg, seg_pat, line, dir_="A",
                                   month_from="2024-10-01", month_to="2024-10-31",
                                   min_n=10, norm_unit_m=100,
                                   show_raw=True, show_norm=True,
                                   cmap_raw="magma", cmap_norm="cividis",
                                   figsize=(20,6)):
    # ---- 篩資料（只看該月）----
    s = seg[(seg["line"].astype(str)==str(line)) & (seg["dir"].astype(str)==str(dir_))].copy()
    s = s[(s["link_start_time"]>=pd.to_datetime(month_from)) &
          (s["link_start_time"]<=pd.to_datetime(month_to))].copy()

    # period168 一定要是 int，且補齊 0..167
    s["period168"] = (s["period168"].astype("Int64")).astype(int)

    # 依最常見 pattern 取順序（用 seg_pat，避免起站掉）
    sp0 = seg_pat[(seg_pat["line"].astype(str)==str(line)) & (seg_pat["dir"].astype(str)==str(dir_))].copy()
    pats = _top_patterns(sp0, top_k=1, min_share=0.05)
    if not pats:
        print("No dominant pattern."); return
    pat,_n,_share = pats[0][0], pats[0][1], pats[0][2]
    ordered_keys = [f"{a}→{b}" for a,b in zip(pat[:-1], pat[1:])]

    # 只保留在 seg 有測得 link_s 的段，且樣本數 >= min_n（但前兩段強制保留）
    s = s[s["SegmentKey"].isin(set(ordered_keys))].copy()
    vc = s["SegmentKey"].value_counts()
    head_keep = set(ordered_keys[:2])
    keep = set(vc[vc>=min_n].index) | (head_keep & set(vc.index))
    ordered_keys = [k for k in ordered_keys if k in keep]
    s = s[s["SegmentKey"].isin(ordered_keys)].copy()

    # 距離
    s["distance_m"] = pd.to_numeric(s["distance_m"], errors="coerce")

    # ---- 聚合：每段 × period168 的 p50（秒），與距離校正（秒/100m）----
    def _agg_one(df):
        out = {"p50": float(np.nanquantile(df["link_s"], 0.5))}
        # per 100m：用「秒/100m」的中位數（中位數在此作用像 robust 比率）
        m = (df["distance_m"]>0) & df["link_s"].notna()
        out["p50_per100m"] = float(np.nanquantile(df.loc[m,"link_s"] / (df.loc[m,"distance_m"]/norm_unit_m), 0.5)) if m.any() else np.nan
        return pd.Series(out)

    g = s.groupby(["SegmentKey","period168"]).apply(_agg_one).reset_index()

    # 轉寬 + 補齊 0..167 欄
    def _pivot(col):
        pv = g.pivot(index="SegmentKey", columns="period168", values=col)
        pv = pv.reindex(index=ordered_keys, columns=range(168))   # 補齊欄、確保順序
        return pv

    H_raw  = _pivot("p50")
    H_norm = _pivot("p50_per100m")

    # ---- 畫圖（用 extent 對齊 cell，vline 在邊界 -0.5）----
    nseg = len(ordered_keys)
    ncols = (show_raw and show_norm) + 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)

    def _imshow(ax, M, ttl, cmap, cbar_label):
        im = ax.imshow(M.values, aspect="auto", origin="upper",
                       cmap=cmap, interpolation="nearest",
                       extent=[-0.5, 167.5, -0.5, nseg-0.5])
        # y tick
        ax.set_yticks(np.arange(nseg))
        ax.set_yticklabels(ordered_keys)
        ax.set_ylabel("segments (top → bottom)")
        # x tick與分隔
        ax.set_xlabel("period168 (Mon00=0 ... Sun23=167)")
        for d in range(1,7):
            ax.axvline(d*24-0.5, color="white", lw=2, alpha=0.8)
        ax.set_title(ttl)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        return ax

    if show_raw and show_norm:
        _imshow(axes[0], H_raw,  f"Line {line} | dir={dir_} — p50 (seconds)   [{month_from[:7]}]",
                cmap_raw,  "median link_s (p50) [s]")
        _imshow(axes[1], H_norm, f"Line {line} | dir={dir_} — p50 (sec/{norm_unit_m}m)   [{month_from[:7]}]",
                cmap_norm, f"median link_s per {norm_unit_m} m (p50) [s/{norm_unit_m}m]")
    elif show_raw:
        _imshow(axes, H_raw,  f"Line {line} | dir={dir_} — p50 (seconds)   [{month_from[:7]}]",
                cmap_raw,  "median link_s (p50) [s]")
    else:
        _imshow(axes, H_norm, f"Line {line} | dir={dir_} — p50 (sec/{norm_unit_m}m)   [{month_from[:7]}]",
                cmap_norm, f"median link_s per {norm_unit_m} m (p50) [s/{norm_unit_m}m]")

    plt.show()


# In[74]:


plot_p50_heatmap_with_distance(
        seg=seg, seg_pat=seg_pat, line=80, dir_="A",
        month_from="2024-10-01", month_to="2024-10-31",
        min_n=10, norm_unit_m=100,
        show_raw=True, show_norm=True,
        cmap_raw="magma", cmap_norm="cividis",
        figsize=(20,6)
)


# In[75]:


import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error

# ---- 你已經有的：LEVELS、_fit_level_quantile、fit_maps、apply_maps、make_empirical_baseline ----
# (略) 直接沿用你現有版本

def pinball_loss(y, qhat, alpha):
    m = np.isfinite(y) & np.isfinite(qhat)
    if not m.any(): return np.nan
    e = y[m] - qhat[m]
    return float(np.mean(np.maximum(alpha*e, (alpha-1)*e)))

def mae(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(np.abs(y[m]-yhat[m]))) if m.any() else np.nan

def coverage(y, qhat):
    m = np.isfinite(y) & np.isfinite(qhat)
    return float(np.mean(y[m] <= qhat[m])) if m.any() else np.nan

# 1) 準備日期切點（以週為例；要改月也很簡單）
def weekly_cutpoints(s, min_weeks_train=4):
    dt = pd.to_datetime(s["link_start_time"])
    wk = (dt.dt.to_period("W").apply(lambda p: p.start_time)).rename("week")
    s = s.assign(_week=wk)
    weeks = sorted(s["_week"].unique())
    cuts = []
    for i in range(min_weeks_train, len(weeks)):
        train_to = weeks[i-1]            # 包含這一週
        test_week = weeks[i]             # 預測下一週
        cuts.append((train_to, test_week))
    return s, cuts

# 2) 單一回合：產生 p50/p85 baseline 並計算指標（含距離校正）
def eval_one_round(train_df, test_df, target="link_s", levels=None, q_main=0.50, q_hi=0.85):
    # fit maps
    b50, _, _, _ = make_empirical_baseline(train_df, test_df, target=target, q=q_main, levels=levels)
    b85, _, _, _ = make_empirical_baseline(train_df, test_df, target=target, q=q_hi,   levels=levels)

    y = test_df[target].to_numpy(dtype=float)
    d = test_df["distance_m"].to_numpy(dtype=float)

    out = {
        "MAE_p50": mae(y, b50),
        "MedAE_p50": float(np.nanmedian(np.abs(y - b50))),
        "Pinball@0.85": pinball_loss(y, b85, 0.85),
        "Coverage@0.85": coverage(y, b85),
    }
    # 距離校正
    m = np.isfinite(y) & np.isfinite(b50) & np.isfinite(d) & (d>0)
    out["MAE_p50_per100m"] = float(np.mean(np.abs(y[m]-b50[m]) / (d[m]/100))) if m.any() else np.nan
    out["MedAE_p50_per100m"] = float(np.nanmedian(np.abs(y[m]-b50[m]) / (d[m]/100))) if m.any() else np.nan
    return out, b50, b85

# 3) 週度滾動回測
def rolling_backtest(seg, target="link_s", levels=None, min_weeks_train=4):
    s, cuts = weekly_cutpoints(seg.dropna(subset=[target, "link_start_time"]), min_weeks_train)
    rows = []
    all_preds = []
    for train_to, test_week in cuts:
        tr = s[s["_week"] <= train_to].copy()
        te = s[s["_week"] == test_week].copy()
        if len(te)==0 or len(tr)==0: 
            continue
        met, p50, p85 = eval_one_round(tr, te, target=target, levels=levels)
        met["train_to"] = train_to; met["test_week"] = test_week; met["n_test"] = len(te)
        rows.append(met)
    res = pd.DataFrame(rows).sort_values("test_week")
    return res

# ---- 跑起來（link_s）----
LEVELS = [
    ["SegmentKey","period168"],
    ["SegmentKey","is_weekend","hour"],
    ["SegmentKey"],
]
bt = rolling_backtest(seg, target="link_s", levels=LEVELS, min_weeks_train=4)

# 整體回答（平均 & 中位數）
summary = bt.agg({
    "MAE_p50":["mean","median"],
    "MedAE_p50":["mean","median"],
    "Coverage@0.85":["mean","median"],
    "Pinball@0.85":["mean","median"],
    "MAE_p50_per100m":["mean","median"],
    "MedAE_p50_per100m":["mean","median"],
    "n_test":"sum",
})
print(summary)


# In[ ]:





# ### Dwell/Runtime segments

# In[119]:


import numpy as np
import pandas as pd
from collections import Counter

# ------------------------------
# Build window-based micro-segments (dwell + runtime)
# ------------------------------
def build_window_microsegments(seg_src, window_total_m=70):
    """
    Create a dataframe 'winseg' with dwell and runtime segments defined by window timestamps.
    Nodes are 'CodeLong|Entree' and 'CodeLong|Sortie'.
    Returns columns:
      - IdCourse, line, dir, trip_seq (ordering index)
      - from_node, to_node, type ('dwell'|'run'), duration_s, start_time, end_time
      - from_stop, to_stop, SegmentKey_win
      - distance_m (dwell=window_total_m; runtime=max(DistanceInterArret - window_total_m, 0) if available)
    """
    s = seg_src.copy()

    # ensure datetimes
    for c in ["DTEntreeFenetreArretReal","DTSortieFenetreArretReal","DTEntreeArretAtp","DTSortieArretAtp",
              "DTDepartTheo","DTArriveeTheo"]:
        if c in s.columns:
            s[c] = pd.to_datetime(s[c], errors="coerce")

    # basic ids
    if "RangArretAsc" in s.columns:
        s["RangArretAsc"] = pd.to_numeric(s["RangArretAsc"], errors="coerce")
    if "C_Ligne" in s.columns:
        s["C_Ligne"] = pd.to_numeric(s["C_Ligne"], errors="coerce")

    # sort trip order
    if "IdCourse" in s.columns:
        s = s.sort_values(["IdCourse","RangArretAsc"], kind="mergesort")
        g = s.groupby("IdCourse", sort=False)
    else:
        # fallback (line/date)
        s = s.sort_values(["C_Ligne","DateCourse","RangArretAsc"], kind="mergesort")
        g = s.groupby(["C_Ligne","DateCourse"], sort=False)

    # clean stop ids
    s["CodeLong"] = s.get("CodeLong","").astype(str).str.strip()
    s["line"] = s.get("C_Ligne").astype("Int64").astype(str) if "C_Ligne" in s.columns else "NA"
    s["dir"]  = s.get("C_SensAppl","NA").astype(str)

    # distances
    dist = pd.to_numeric(s.get("DistanceInterArret"), errors="coerce")

    # shift next stop's Entree for runtime construction
    s["next_Entree"]  = g["DTEntreeFenetreArretReal"].shift(-1)
    s["next_CodeLong"] = g["CodeLong"].shift(-1)

    # build dwell rows (where both window times exist)
    dwell_mask = s["DTEntreeFenetreArretReal"].notna() & s["DTSortieFenetreArretReal"].notna()
    dwell = s.loc[dwell_mask, [
        "IdCourse","line","dir","CodeLong","DTEntreeFenetreArretReal","DTSortieFenetreArretReal","RangArretAsc"
    ]].copy()
    dwell["from_node"] = dwell["CodeLong"] + "|Entree"
    dwell["to_node"]   = dwell["CodeLong"] + "|Sortie"
    dwell["type"] = "dwell"
    dwell["start_time"] = dwell["DTEntreeFenetreArretReal"]
    dwell["end_time"]   = dwell["DTSortieFenetreArretReal"]
    dwell["duration_s"] = (dwell["end_time"] - dwell["start_time"]).dt.total_seconds()
    dwell["from_stop"]  = dwell["CodeLong"]
    dwell["to_stop"]    = dwell["CodeLong"]
    dwell["SegmentKey_win"] = dwell["from_node"] + "→" + dwell["to_node"]
    dwell["distance_m"] = float(window_total_m)

    # QC dwell
    dwell.loc[(dwell["duration_s"]<=0) | (dwell["duration_s"]>900), "duration_s"] = np.nan

    # build runtime rows (need current Sortie and next Entree)
    run_mask = s["DTSortieFenetreArretReal"].notna() & s["next_Entree"].notna()
    run = s.loc[run_mask, [
        "IdCourse","line","dir","CodeLong","DTSortieFenetreArretReal","next_Entree","next_CodeLong","RangArretAsc"
    ]].copy()
    run["from_node"] = run["CodeLong"] + "|Sortie"
    run["to_node"]   = run["next_CodeLong"] + "|Entree"
    run["type"] = "run"
    run["start_time"] = run["DTSortieFenetreArretReal"]
    run["end_time"]   = run["next_Entree"]
    run["duration_s"] = (run["end_time"] - run["start_time"]).dt.total_seconds()
    run["from_stop"]  = run["CodeLong"]
    run["to_stop"]    = run["next_CodeLong"]
    run["SegmentKey_win"] = run["from_node"] + "→" + run["to_node"]

    # runtime distance: stop-to-stop minus window length (>=0). If DistanceInterArret missing, leave NaN.
    dist_runtime = pd.to_numeric(s.get("DistanceInterArret"), errors="coerce") - float(window_total_m)
    run["distance_m"] = dist_runtime.loc[run.index].clip(lower=0)

    # QC runtime
    run.loc[(run["duration_s"]<=0) | (run["duration_s"]>1800), "duration_s"] = np.nan

    # concatenate
    keep_cols = ["IdCourse","line","dir","RangArretAsc","from_node","to_node","type",
                 "start_time","end_time","duration_s","from_stop","to_stop","SegmentKey_win","distance_m"]
    winseg = pd.concat([dwell[keep_cols], run[keep_cols]], ignore_index=True)

    # an ordering index across micro-segments inside the trip
    # dwell at k gets seq=2*k, runtime k->k+1 gets seq=2*k+1 (preserves stop order)
    winseg["trip_seq"] = winseg.groupby("IdCourse")["start_time"].rank(method="first").astype(int)

    # convenience time bins
    winseg["period168"] = winseg["start_time"].dt.dayofweek * 24 + winseg["start_time"].dt.hour
    winseg["hour"] = winseg["start_time"].dt.hour
    winseg["dow"]  = winseg["start_time"].dt.dayofweek

    return winseg

# ------------------------------
# Pattern extraction on window-nodes
# ------------------------------
def extract_window_patterns(winseg, line, dir_=None, min_share=0.10):
    """
    Build full node sequence per trip: [A|Entree, A|Sortie, B|Entree, B|Sortie, ...]
    Count identical sequences; return dominant ones (>= min_share).
    """
    s = winseg[winseg["line"].astype(str)==str(line)].copy()
    if dir_ is not None:
        s = s[s["dir"].astype(str)==str(dir_)].copy()
    if s.empty:
        return pd.DataFrame(columns=["pattern_id","trips","share","n_nodes","start_node","end_node","pattern_str","preview"])

    s = s.sort_values(["IdCourse","trip_seq"], kind="mergesort")

    def _trip_node_seq(g):
        # reconstruct node sequence from from_node and to_node in order
        nodes = [g["from_node"].iloc[0]] + g["to_node"].tolist()
        return tuple(nodes)

    paths = s.groupby("IdCourse", sort=False).apply(_trip_node_seq)
    cnt = Counter(paths); total = sum(cnt.values())
    if total == 0:
        return pd.DataFrame(columns=["pattern_id","trips","share","n_nodes","start_node","end_node","pattern_str","preview"])

    rows = []
    ranked = [(p, n, n/total) for p, n in cnt.most_common() if (n/total) >= min_share]
    for i,(pat,n,share) in enumerate(ranked, start=1):
        pat_list = list(pat)
        preview = " → ".join(pat_list[:5]) + (" → … → " + " → ".join(pat_list[-4:]) if len(pat_list)>12 else "")
        rows.append({
            "pattern_id": i,
            "trips": n,
            "share": round(share, 4),
            "n_nodes": len(pat_list),
            "start_node": pat_list[0],
            "end_node": pat_list[-1],
            "pattern_str": " → ".join(pat_list),
            "preview": preview
        })
    return pd.DataFrame(rows)

# ------------------------------
# Example usage
# ------------------------------
# winseg = build_window_microsegments(seg_src, window_total_m=70)
# df_winpat = extract_window_patterns(winseg, line=80, dir_="A", min_share=0.05)
# display(df_winpat.head(10))


# In[123]:


winseg = build_window_microsegments(seg_src, window_total_m=70)
df_winpat = extract_window_patterns(winseg, line=18, dir_="A", min_share=0.05)


# In[124]:


display(df_winpat.head(10))


# In[142]:


# ========= Imports =========
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import Counter

# ========= 0) Month slice for window segments =========
def month_slice_win(winseg, start="2024-10-01", end="2024-10-31"):
    S = winseg[(winseg["start_time"] >= pd.to_datetime(start)) &
               (winseg["start_time"] <  pd.to_datetime(end))].copy()
    return S

# ========= 1) Order keys by top *node* pattern, then convert to edge keys =========
def order_keys_by_top_window_pattern(
    S_month, line, dir_=None, min_share=0.05, type_filter="both"
):
    """
    returns ordered list of SegmentKey_win for the dominant node-sequence pattern.
    type_filter: 'both' | 'run' | 'dwell'
    """
    s = S_month[S_month["line"].astype(str)==str(line)].copy()
    if dir_ is not None: s = s[s["dir"].astype(str)==str(dir_)]
    if s.empty: return []

    # keep trip order
    s = s.sort_values(["IdCourse","trip_seq"], kind="mergesort")

    # full node sequence per trip (start node then successive to_nodes)
    def _trip_nodes(g):
        return tuple([g["from_node"].iloc[0]] + g["to_node"].tolist())
    paths = s.groupby("IdCourse", sort=False, group_keys=False).apply(_trip_nodes)

    cnt = Counter(paths); total = sum(cnt.values())
    if total == 0: return []

    # top pattern
    pat, n = cnt.most_common(1)[0]
    share = n/total
    if share < min_share:
        print(f"[warn] top pattern share only {share:.1%} (<{min_share:.0%})")

    # Convert the node pattern to *edge* keys and optionally filter by type
    # Recreate the trip rows that follow this exact node sequence
    # Build an edge-key list in order of appearance within that pattern
    nodes = list(pat)
    wanted_edges = set([f"{a}→{b}" for a, b in zip(nodes[:-1], nodes[1:])])

    sp = s[s["SegmentKey_win"].isin(wanted_edges)].copy()
    # enforce pattern order strictly
    sp["__ord"] = sp["from_node"].map({n:i for i,n in enumerate(nodes)})
    sp = sp.sort_values(["__ord"], kind="mergesort")

    if type_filter in ("run","dwell"):
        sp = sp[sp["type"] == type_filter]

    ordered_keys = sp["SegmentKey_win"].tolist()
    # remove consecutive duplicates (defensive)
    ordered_keys = [k for i,k in enumerate(ordered_keys) if i==0 or k != ordered_keys[i-1]]
    return ordered_keys

# ========= 2A) Boxplot for a single period168 =========
def plot_box_by_period_win(
    S_month, ordered_keys, period168, line, dir_=None, type_filter="run",
    min_n_per_seg=15, figsize=(17,6)
):
    s = S_month.copy()
    s = s[(s["line"].astype(str)==str(line)) & (s["period168"]==period168)]
    if dir_ is not None: s = s[s["dir"].astype(str)==str(dir_)]
    if type_filter in ("run","dwell"): s = s[s["type"]==type_filter]

    s = s[pd.to_numeric(s["duration_s"], errors="coerce") > 0]
    if s.empty: 
        print("no data in this period"); return

    vc = s["SegmentKey_win"].value_counts()
    head = set(ordered_keys[:2])
    keep = set(vc[vc>=min_n_per_seg].index) | (head & set(vc.index))
    keys = [k for k in ordered_keys if k in keep]
    if not keys:
        print("all segments < min_n in this period"); return

    data = [s.loc[s["SegmentKey_win"]==k, "duration_s"].values for k in keys]

    y_lo = float(np.nanpercentile(s["duration_s"], 1))
    y_hi = float(np.nanpercentile(s["duration_s"], 99))

    plt.figure(figsize=figsize)
    plt.boxplot(data, vert=True, showfliers=False, widths=0.6, labels=keys)
    plt.ylim(y_lo, y_hi)
    ttl = f"Line {line}" + (f" | dir={dir_}" if dir_ else "")
    ttl += f" — period168={period168} — {type_filter if type_filter!='both' else 'both types'}"
    plt.title(ttl, fontsize=16)
    plt.ylabel("duration_s (seconds)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.xticks(rotation=70, ha="right")

    for i, k in enumerate(keys, start=1):
        n_i = len(data[i-1])
        plt.text(i, y_lo + 0.02*(y_hi-y_lo), f"n={n_i}", ha="center", va="bottom", fontsize=9, rotation=90)

    plt.tight_layout(); plt.show()

# ========= 2B) Heatmap of variability (p90 - p10) =========
def plot_heatmap_spread_win(
    S_month, ordered_keys, line, dir_=None, type_filter="run",
    min_n=15, vmax=None, figsize=(18,6)
):
    s = S_month.copy()
    s = s[s["line"].astype(str)==str(line)]
    if dir_ is not None: s = s[s["dir"].astype(str)==str(dir_)]
    if type_filter in ("run","dwell"): s = s[s["type"]==type_filter]
    s = s[pd.to_numeric(s["duration_s"], errors="coerce") > 0]
    s = s[s["SegmentKey_win"].isin(set(ordered_keys))]

    g = s.groupby(["SegmentKey_win","period168"])["duration_s"]
    agg = g.agg(n="size",
                p10=lambda x: np.nanpercentile(x,10),
                p90=lambda x: np.nanpercentile(x,90)).reset_index()
    agg.loc[agg["n"]<min_n, ["p10","p90"]] = np.nan
    agg["spread"] = agg["p90"] - agg["p10"]

    mat = (agg.pivot(index="period168", columns="SegmentKey_win", values="spread")
              .reindex(index=range(168), columns=ordered_keys))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.to_numpy().T, aspect="auto", origin="upper",
                   interpolation="nearest", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="spread (p90 - p10) [s]")
    ax.set_yticks(range(len(ordered_keys))); ax.set_yticklabels(ordered_keys, fontsize=9)
    ax.set_xticks(range(0,168,6)); ax.set_xticklabels(range(0,168,6))
    ttl = f"Line {line}" + (f" | dir={dir_}" if dir_ else "") + f" — variability heatmap (p90-p10) — {type_filter}"
    ax.set_title(ttl, fontsize=16)
    ax.set_xlabel("period168 (Mon00=0 … Sun23=167)")
    ax.set_ylabel("segments (top → bottom)")
    fig.tight_layout(); plt.show()

# ========= 2C) p50 heatmap (+ distance-normalized) =========
def plot_p50_heatmap_with_distance_win(
    S_month, ordered_keys, line, dir_=None, type_filter="run",
    min_n=10, norm_unit_m=100, show_raw=True, show_norm=True,
    cmap_raw="magma", cmap_norm="cividis", figsize=(20,6)
):
    s = S_month.copy()
    s = s[s["line"].astype(str)==str(line)]
    if dir_ is not None: s = s[s["dir"].astype(str)==str(dir_)]
    if type_filter in ("run","dwell"): s = s[s["type"]==type_filter]

    s = s[pd.to_numeric(s["duration_s"], errors="coerce") > 0]
    s = s[s["SegmentKey_win"].isin(set(ordered_keys))]
    if s.empty:
        print("No positive duration data in month range."); return

    g = (s.groupby(["SegmentKey_win","period168"])["duration_s"]
           .agg(n="size", p50=lambda x: float(np.nanmedian(x)))).reset_index()
    g.loc[g["n"]<min_n, "p50"] = np.nan

    mat_raw = (g.pivot(index="SegmentKey_win", columns="period168", values="p50")
                 .reindex(index=ordered_keys))
    keep_rows = mat_raw.index[mat_raw.notna().any(axis=1)].tolist()
    mat_raw = mat_raw.loc[keep_rows]

    # distance-normalized (sec per norm_unit_m)
    dist_per_seg = (s.groupby("SegmentKey_win")["distance_m"].median().reindex(mat_raw.index))
    dist_vec = dist_per_seg.to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        mat_norm_vals = mat_raw.to_numpy() / (dist_vec[:, None] / float(norm_unit_m))
    mat_norm = pd.DataFrame(mat_norm_vals, index=mat_raw.index, columns=mat_raw.columns)
    mat_norm[(~np.isfinite(mat_norm)) | (dist_vec[:,None] <= 0)] = np.nan

    n_panels = int(show_raw) + int(show_norm)
    fig, axs = plt.subplots(1, n_panels, figsize=figsize, squeeze=False); axs = axs[0]

    def style_x(ax):
        ax.set_xlim(-0.5, 167.5)
        for d in range(1,7): ax.axvline(d*24-0.5, color="k", lw=1, alpha=0.25)
        ax.set_xticks(np.arange(0,168,6)); ax.set_xlabel("period168 (Mon00 … Sun23)")

    def style_y(ax, idx):
        ax.set_yticks(np.arange(len(idx))); ax.set_yticklabels(idx); ax.set_ylabel("segments (top → bottom)")

    pane = 0
    if show_raw:
        cmap = plt.get_cmap(cmap_raw).copy(); cmap.set_bad("white")
        vmin_raw = np.nanpercentile(mat_raw.values, 1); vmax_raw = np.nanpercentile(mat_raw.values, 95)
        im = axs[pane].imshow(mat_raw.values, aspect="auto", origin="upper",
                              interpolation="nearest", cmap=cmap, vmin=vmin_raw, vmax=vmax_raw)
        style_x(axs[pane]); style_y(axs[pane], mat_raw.index)
        fig.colorbar(im, ax=axs[pane]).set_label("median duration (p50) [s]")
        axs[pane].set_title(f"Line {line}" + (f" | dir={dir_}" if dir_ else "") + f" — p50 [s] — {type_filter}")
        pane += 1

    if show_norm:
        cmap = plt.get_cmap(cmap_norm).copy(); cmap.set_bad("white")
        vmin_norm = np.nanpercentile(mat_norm.values, 1); vmax_norm = np.nanpercentile(mat_norm.values, 95)
        im = axs[pane].imshow(mat_norm.values, aspect="auto", origin="upper",
                              interpolation="nearest", cmap=cmap, vmin=vmin_norm, vmax=vmax_norm)
        style_x(axs[pane]); style_y(axs[pane], mat_norm.index)
        fig.colorbar(im, ax=axs[pane]).set_label(f"median per {norm_unit_m} m (p50) [s/{norm_unit_m}m]")
        axs[pane].set_title(f"Line {line}" + (f" | dir={dir_}" if dir_ else "") + f" — p50 [s/{norm_unit_m}m] — {type_filter}")

    plt.tight_layout(); plt.show()


# In[131]:


# 1) month view (Oct 2024)
W_oct = month_slice_win(winseg, start="2024-10-01", end="2024-11-01")

# 2) decide row order by the dominant node pattern, then choose which micro-segments to display
ordered_all  = order_keys_by_top_window_pattern(W_oct, line=18, dir_="A", min_share=0.05, type_filter="both")
ordered_run  = order_keys_by_top_window_pattern(W_oct, line=18, dir_="A", min_share=0.05, type_filter="run")
ordered_dwel = order_keys_by_top_window_pattern(W_oct, line=18, dir_="A", min_share=0.05, type_filter="dwell")

print(len(ordered_run), ordered_run[:5], "...")


# In[138]:


# 3B) Variability heatmap (p90-p10)
plot_heatmap_spread_win(W_oct, ordered_run, line=18, dir_="A",
                        type_filter="run", min_n=10, vmax=100)


# In[144]:


plot_heatmap_spread_win(W_oct, ordered_dwel, line=18, dir_="A",
                        type_filter="dwell", min_n=10, vmax=100)


# In[155]:


# 3C) p50 heatmap (and per-distance)
plot_p50_heatmap_with_distance_win(W_oct, ordered_run, line=18, dir_="A",
                                   type_filter="run", min_n=10, norm_unit_m=100,
                                   show_raw=True, show_norm=True,
                                   cmap_raw="magma", cmap_norm="cividis", figsize=(20,6))

