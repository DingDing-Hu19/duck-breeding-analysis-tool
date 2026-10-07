# -*- coding: utf-8 -*-
"""
肉鸭智能化采食行为分析工具 V5.0 —— 核心算法层
四川农业大学 张雅晨团队 · 大挑项目 · 留种决策模块
Python 迁移版
"""
import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. 统一配色（红紫蓝绿）
# ============================================================
PAL_RPBG = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]

# ============================================================
# 1. 基础辅助
# ============================================================

def clean_names_simple(cols):
    return [re.sub(r"[\r\n\t]", "", str(c).replace("\u00a0", " ")).strip() for c in cols]


def find_col(df, patterns, required=False):
    """按正则列表查找第一个匹配列名（忽略大小写）"""
    cols = list(df.columns)
    for p in patterns:
        for c in cols:
            if re.search(p, str(c), flags=re.IGNORECASE):
                return c
    if required:
        raise ValueError(f"无法找到需要的字段。候选关键词：{patterns}\n当前字段：{cols}")
    return None


def safe_numeric(x):
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce")
    s = (x.astype(str)
           .str.replace(",", "", regex=False)
           .str.strip()
           .replace({"": np.nan, "NA": np.nan, "NaN": np.nan,
                     "NULL": np.nan, "null": np.nan, "-": np.nan}))
    return pd.to_numeric(s, errors="coerce")


def parse_datetime_flexible(x):
    """兼容 Excel 序列号 / 多种字符串格式"""
    if pd.api.types.is_datetime64_any_dtype(x):
        return pd.to_datetime(x, errors="coerce")
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_datetime(x, unit="D", origin="1899-12-30", errors="coerce")
    s = x.astype(str).str.strip().replace(
        {"": np.nan, "NA": np.nan, "NaN": np.nan, "NULL": np.nan, "null": np.nan})
    return pd.to_datetime(s, errors="coerce", dayfirst=False, format="mixed")


def parse_duration_seconds(x):
    """解析 '1.02:03:04' / '01:02:03' / 'HH:MM' / 数字（天）"""
    if pd.api.types.is_timedelta64_dtype(x):
        return x.dt.total_seconds()
    if pd.api.types.is_numeric_dtype(x):
        return x.astype(float) * 86400.0
    s = x.astype(str).str.strip()
    out = pd.Series(np.nan, index=s.index, dtype=float)

    m = s.str.match(r"^\d+\.\d{1,2}:\d{2}:\d{2}$")
    if m.any():
        parts = s[m].str.extract(r"^(\d+)\.(\d{1,2}):(\d{2}):(\d{2})$").astype(float)
        out[m] = parts[0] * 86400 + parts[1] * 3600 + parts[2] * 60 + parts[3]
    m = out.isna() & s.str.match(r"^\d+:\d{2}:\d{2}$")
    if m.any():
        parts = s[m].str.extract(r"^(\d+):(\d{2}):(\d{2})$").astype(float)
        out[m] = parts[0] * 3600 + parts[1] * 60 + parts[2]
    m = out.isna() & s.str.match(r"^\d+:\d{2}$")
    if m.any():
        parts = s[m].str.extract(r"^(\d+):(\d{2})$").astype(float)
        out[m] = parts[0] * 3600 + parts[1] * 60
    m = out.isna() & s.str.match(r"^\d+(\.\d+)?$")
    if m.any():
        out[m] = pd.to_numeric(s[m], errors="coerce")
    return out


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def safe_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def safe_sd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan


def format_p(p):
    if p is None or pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ============================================================
# 2. 读取 Excel（全部 sheet，自动匹配字段结构）
# ============================================================
FEED_PATTERNS = {
    "animal":  [r"^耳标$", r"耳标", r"animal.*id", r"tag"],
    "time1":   [r"^记录时间1$", r"记录.*时间.*1", r"time.*1", r"start.*time"],
    "time2":   [r"^记录时间2$", r"记录.*时间.*2", r"time.*2", r"end.*time"],
    "intake":  [r"^处理后采食量\s*\(?g\)?$", r"^处理后采食量", r"处理后.*采食量"],
    "duration":[r"^采食时长$", r"采食.*时长", r"feeding.*duration", r"duration"],
}
BW_PATTERNS = {
    "animal":  [r"^耳标$", r"耳标", r"animal.*id", r"tag"],
    "time1":   [r"^记录时间1$", r"记录.*时间.*1", r"time.*1"],
    "time2":   [r"^记录时间2$", r"记录.*时间.*2", r"time.*2"],
    "bw1":     [r"^体重1\s*\(?kg\)?$", r"^体重1", r"weight1"],
    "bw2":     [r"^体重2\s*\(?kg\)?$", r"^体重2", r"weight2"],
    "bw3":     [r"^体重3\s*\(?kg\)?$", r"^体重3", r"weight3"],
    "mean_bw": [r"^平均体重\s*\(?kg\)?$", r"^平均体重", r"mean.*weight"],
}


def _sheet_matches(header_df, patterns):
    if header_df is None or header_df.shape[1] == 0:
        return False
    h = header_df.copy()
    h.columns = clean_names_simple(h.columns)
    return all(find_col(h, p, False) is not None for p in patterns.values())


def read_matching_excel_files(files, dataset_type):
    """
    files: Streamlit UploadedFile 列表 或 [(name, path/bytes)] 列表
    返回 (合并后 DataFrame, QC 表 DataFrame)
    """
    patterns = FEED_PATTERNS if dataset_type == "feed" else BW_PATTERNS
    result_list, qc_list = [], []

    for f in files:
        name = f.name if hasattr(f, "name") else f[0]
        src = f if hasattr(f, "read") else f[1]
        try:
            xls = pd.ExcelFile(src)
        except Exception as e:
            qc_list.append(dict(数据集=dataset_type, Source_File=name,
                                Source_Sheet="-", 状态="读取失败", 读取记录数=0,
                                说明=str(e)))
            continue

        for sheet in xls.sheet_names:
            try:
                header = xls.parse(sheet, nrows=5)
            except Exception as e:
                qc_list.append(dict(数据集=dataset_type, Source_File=name,
                                    Source_Sheet=sheet, 状态="跳过：表头读取失败",
                                    读取记录数=0, 说明=str(e)))
                continue

            if not _sheet_matches(header, patterns):
                qc_list.append(dict(数据集=dataset_type, Source_File=name,
                                    Source_Sheet=sheet, 状态="跳过：字段结构不匹配",
                                    读取记录数=0, 说明=""))
                continue

            try:
                df = xls.parse(sheet)
            except Exception as e:
                qc_list.append(dict(数据集=dataset_type, Source_File=name,
                                    Source_Sheet=sheet, 状态="跳过：读取失败",
                                    读取记录数=0, 说明=str(e)))
                continue

            df = df.dropna(how="all").dropna(axis=1, how="all")
            df.columns = clean_names_simple(df.columns)
            if df.empty:
                qc_list.append(dict(数据集=dataset_type, Source_File=name,
                                    Source_Sheet=sheet, 状态="跳过：工作表为空",
                                    读取记录数=0, 说明="字段匹配但无有效记录"))
                continue

            df["_Source_File"] = name
            df["_Source_Sheet"] = sheet
            result_list.append(df)
            qc_list.append(dict(数据集=dataset_type, Source_File=name,
                                Source_Sheet=sheet, 状态="已读取",
                                读取记录数=len(df), 说明="包含完整数据字段"))

    if not result_list:
        raise ValueError(f"{dataset_type} 数据中没有识别到包含完整字段的工作表。")
    return pd.concat(result_list, ignore_index=True), pd.DataFrame(qc_list)


def read_pedigree(path):
    """系谱：翅号 / 父本笼号 / 母本笼号 / 性别"""
    if path is None:
        return pd.DataFrame(columns=["ID", "Sire_Cage", "Dam_Cage", "Sex"])
    ped = pd.read_excel(path)
    ped.columns = clean_names_simple(ped.columns)
    id_col  = find_col(ped, [r"^翅号$", r"翅号", r"id", r"animal"], required=True)
    sire_col = find_col(ped, [r"^父本笼号$", r"父本", r"sire", r"father"])
    dam_col  = find_col(ped, [r"^母本笼号$", r"母本", r"dam", r"mother"])
    sex_col  = find_col(ped, [r"^性别$", r"性别", r"sex"])
    out = pd.DataFrame({
        "ID": ped[id_col].astype(str),
        "Sire_Cage": ped[sire_col].astype(str) if sire_col else np.nan,
        "Dam_Cage":  ped[dam_col].astype(str)  if dam_col  else np.nan,
        "Sex":       ped[sex_col].astype(str)  if sex_col  else np.nan,
    })
    out = out[out["ID"].str.strip().ne("") & out["ID"].ne("nan")]
    return out.drop_duplicates("ID", keep="first").reset_index(drop=True)


def read_idmap(path):
    """eID - ID 对照表"""
    if path is None:
        return pd.DataFrame(columns=["eID", "ID"])
    m = pd.read_excel(path)
    m.columns = clean_names_simple(m.columns)
    eid_col = find_col(m, [r"^eID$", r"eid", r"耳标", r"电子耳标", r"rfid"], required=True)
    id_col  = find_col(m, [r"^ID$", r"id", r"翅号", r"个体号"], required=True)
    out = pd.DataFrame({"eID": m[eid_col].astype(str), "ID": m[id_col].astype(str)})
    out = out[out["eID"].str.strip().ne("") & out["ID"].str.strip().ne("")]
    return out.drop_duplicates("eID", keep="first").reset_index(drop=True)


def link_animals(feed, bw, ped, idmap):
    """eID → ID(翅号) → 系谱"""
    feed_ids = set(feed["Animal_ID"].astype(str))
    bw_ids   = set(bw["Animal_ID"].astype(str))
    both = sorted(feed_ids & bw_ids)

    if idmap is None or len(idmap) == 0 or ped is None or len(ped) == 0:
        return pd.DataFrame({
            "eID": both, "ID": both,
            "Sire_Cage": np.nan, "Dam_Cage": np.nan, "Sex": np.nan,
            "Has_Pedigree": False, "Has_Both_Data": True,
        })

    mapped = idmap[idmap["eID"].isin(both)].copy()
    mapped = mapped.merge(ped, on="ID", how="left")
    mapped["Has_Pedigree"] = mapped["Sire_Cage"].notna() | mapped["Dam_Cage"].notna()
    mapped["Has_Both_Data"] = True
    return mapped.reset_index(drop=True)


# ============================================================
# 3. 清洗与指标计算
# ============================================================

def make_experimental_week(dt_series, start_dt, week_length):
    dt = pd.to_datetime(dt_series)
    day_num = np.floor((dt - pd.to_datetime(start_dt)).dt.total_seconds() / 86400).astype(int) + 1
    week_num = np.floor((day_num - 1) / week_length).astype(int) + 1
    return pd.DataFrame({
        "Experimental_Day": day_num.values,
        "Week": ["Week " + str(w) for w in week_num.values],
        "Week_Number": week_num.values,
    }, index=dt_series.index)


def remove_weekly_outliers(dat, value_col, week_col="Week_Number", sd_multiplier=3):
    g = dat.groupby(week_col)[value_col]
    stats_df = g.agg(["count", "mean", "std"]).reset_index()
    stats_df.columns = [week_col, "N", "Mean", "SD"]
    stats_df["Lower"] = np.where(stats_df["SD"].isna(), -np.inf,
                                 np.where(stats_df["SD"] == 0, stats_df["Mean"],
                                          stats_df["Mean"] - sd_multiplier * stats_df["SD"]))
    stats_df["Upper"] = np.where(stats_df["SD"].isna(), np.inf,
                                 np.where(stats_df["SD"] == 0, stats_df["Mean"],
                                          stats_df["Mean"] + sd_multiplier * stats_df["SD"]))
    out = dat.merge(stats_df[[week_col, "Lower", "Upper", "N", "Mean", "SD"]],
                    on=week_col, how="left")
    v = out[value_col]
    out["QC_Outlier"] = v.notna() & np.isfinite(v) & ((v < out["Lower"]) | (v > out["Upper"]))
    return out, stats_df


def run_clean_v5(feed_raw, bw_raw, cfg):
    """返回 dict: feed, bw, datetime_qc, na_qc, feed_week_stats, bw_week_stats"""
    # ---------- 采食 ----------
    feed = feed_raw.copy()
    feed.columns = clean_names_simple(feed.columns)
    col_animal = find_col(feed, [r"^耳标$", r"耳标"], required=True)
    col_t1     = find_col(feed, [r"^记录时间1$", r"记录时间1"], required=True)
    col_t2     = find_col(feed, [r"^记录时间2$", r"记录时间2"])
    col_intake = find_col(feed, [r"^处理后采食量\s*\(?g\)?$", r"^处理后采食量"], required=True)
    col_dur    = find_col(feed, [r"^采食时长$", r"采食.*时长"], required=True)
    col_house  = find_col(feed, [r"^栋舍$", r"栋舍"])
    col_pen    = find_col(feed, [r"^栏圈$", r"栏圈"])
    col_group  = find_col(feed, [r"^群组$", r"群组"])

    feed = feed.assign(
        Animal_ID=feed[col_animal].astype(str),
        Time1=parse_datetime_flexible(feed[col_t1]),
        Time2=parse_datetime_flexible(feed[col_t2]) if col_t2 else parse_datetime_flexible(feed[col_t1]),
        Feed_Intake_g=safe_numeric(feed[col_intake]),
        Feed_Duration_sec=parse_duration_seconds(feed[col_dur]),
        House=feed[col_house].astype(str) if col_house else np.nan,
        Pen=feed[col_pen].astype(str) if col_pen else np.nan,
        Group=feed[col_group].astype(str) if col_group else np.nan,
    )
    feed = feed[feed["Animal_ID"].str.strip().ne("") & feed["Time1"].notna()].reset_index(drop=True)

    t_min, t_max = feed["Time1"].min(), feed["Time1"].max()
    datetime_qc = pd.DataFrame({"Start": [t_min], "End": [t_max], "N": [len(feed)]})

    # 自动时间范围
    if cfg.get("auto_time_range", True):
        cfg["experiment_start"] = t_min.date()
        cfg["training_end"] = t_max.date()
        cfg["training_time"] = t_max.strftime("%H:%M:%S")

    # NA QC（全部字段）
    na_rows = []
    for c in feed.columns:
        v = feed[c]
        n_na = int(v.isna().sum()) if pd.api.types.is_numeric_dtype(v) or pd.api.types.is_datetime64_any_dtype(v) \
               else int((v.isna() | v.astype(str).str.strip().eq("")).sum())
        na_rows.append(dict(字段=c, 类型=str(v.dtype), 缺失数=n_na,
                            记录数=len(feed), 缺失率=round(n_na / max(len(feed), 1) * 100, 2)))
    na_qc = pd.DataFrame(na_rows)

    # 截止时间
    training_end = cfg.get("training_end")
    if training_end is not None and not pd.isna(training_end):
        end_dt = pd.to_datetime(f"{training_end} {cfg.get('training_time', '00:00:00')}")
        feed = feed[feed["Time1"] <= end_dt]

    start_dt = cfg.get("experiment_start")
    if start_dt is not None and not pd.isna(start_dt):
        feed = feed[feed["Time1"].dt.date >= pd.to_datetime(start_dt).date()]
    feed = feed.sort_values("Time1").reset_index(drop=True)

    if feed.empty:
        raise ValueError("数据过滤后为空：请检查『试验开始日期』是否晚于数据最早记录时间。")

    wk = make_experimental_week(feed["Time1"], start_dt, cfg["week_length"])
    feed = pd.concat([feed.reset_index(drop=True), wk.reset_index(drop=True)], axis=1)

    feed, feed_week_stats = remove_weekly_outliers(
        feed, "Feed_Intake_g", "Week_Number", cfg["feed_sd"])
    feed.loc[feed["QC_Outlier"], "Feed_Intake_g"] = np.nan

    # ---------- 体重 ----------
    bw = bw_raw.copy()
    bw.columns = clean_names_simple(bw.columns)
    c_bw_animal = find_col(bw, [r"^耳标$", r"耳标"], required=True)
    c_bw_t1     = find_col(bw, [r"^记录时间1$", r"记录时间1"], required=True)
    c_mean      = find_col(bw, [r"^平均体重\s*\(?kg\)?$", r"^平均体重"], required=True)
    c_b1 = find_col(bw, [r"^体重1\s*\(?kg\)?$", r"^体重1"])
    c_b2 = find_col(bw, [r"^体重2\s*\(?kg\)?$", r"^体重2"])
    c_b3 = find_col(bw, [r"^体重3\s*\(?kg\)?$", r"^体重3"])
    c_bw_house = find_col(bw, [r"^栋舍$", r"栋舍"])
    c_bw_pen   = find_col(bw, [r"^栏圈$", r"栏圈"])
    c_bw_group = find_col(bw, [r"^群组$", r"群组"])

    bw = bw.assign(
        Animal_ID=bw[c_bw_animal].astype(str),
        Time1=parse_datetime_flexible(bw[c_bw_t1]),
        BW_kg=safe_numeric(bw[c_mean]),
        BW1_kg=safe_numeric(bw[c_b1]) if c_b1 else np.nan,
        BW2_kg=safe_numeric(bw[c_b2]) if c_b2 else np.nan,
        BW3_kg=safe_numeric(bw[c_b3]) if c_b3 else np.nan,
        House=bw[c_bw_house].astype(str) if c_bw_house else np.nan,
        Pen=bw[c_bw_pen].astype(str) if c_bw_pen else np.nan,
        Group=bw[c_bw_group].astype(str) if c_bw_group else np.nan,
    )
    bw = bw[bw["Animal_ID"].str.strip().ne("") &
            bw["Time1"].notna() & bw["BW_kg"].notna()].reset_index(drop=True)

    if training_end is not None and not pd.isna(training_end):
        bw = bw[bw["Time1"] <= end_dt]
    if start_dt is not None and not pd.isna(start_dt):
        bw = bw[bw["Time1"].dt.date >= pd.to_datetime(start_dt).date()]

    wk_b = make_experimental_week(bw["Time1"], start_dt, cfg["week_length"])
    bw = pd.concat([bw.reset_index(drop=True), wk_b.reset_index(drop=True)], axis=1)
    bw, bw_week_stats = remove_weekly_outliers(bw, "BW_kg", "Week_Number", cfg["bw_sd"])
    bw.loc[bw["QC_Outlier"], "BW_kg"] = np.nan

    return dict(feed=feed, bw=bw, datetime_qc=datetime_qc, na_qc=na_qc,
                feed_week_stats=feed_week_stats, bw_week_stats=bw_week_stats)


# ---------- bout 划分 ----------
def build_bouts(feed, imi_threshold_sec=300, min_intake_g=1):
    df = feed.sort_values(["Animal_ID", "Time1"]).copy()
    df["Gap_sec"] = df.groupby("Animal_ID")["Time1"].diff().dt.total_seconds()
    df["New_Bout"] = df["Gap_sec"].isna() | (df["Gap_sec"] >= imi_threshold_sec)
    df["Bout_ID"] = df.groupby("Animal_ID")["New_Bout"].cumsum()

    bout = (df.groupby(["Animal_ID", "Bout_ID"])
              .agg(Time1=("Time1", "first"),
                   Time2=("Time2", "last"),
                   Feed_Intake_g=("Feed_Intake_g", "sum"),
                   Feed_Duration_sec=("Feed_Duration_sec", "sum"))
              .reset_index())
    bout["Date"] = bout["Time1"].dt.date
    bout["FR_g_sec"] = np.where(bout["Feed_Duration_sec"] > 0,
                                bout["Feed_Intake_g"] / bout["Feed_Duration_sec"], np.nan)
    bout = bout.sort_values(["Animal_ID", "Time1"])
    bout["IMI_sec"] = bout.groupby("Animal_ID")["Time1"].diff().dt.total_seconds()
    bout = bout[bout["Feed_Intake_g"] >= min_intake_g].reset_index(drop=True)
    return bout


def calc_individual_feeding(bout):
    g = bout.groupby("Animal_ID")
    out = g.agg(
        TFB=("Feed_Intake_g", "size"),
        FI_g=("Feed_Intake_g", "sum"),
        AMS_g=("Feed_Intake_g", "mean"),
        TFD_sec=("Feed_Duration_sec", "sum"),
        AFBD_sec=("Feed_Duration_sec", "mean"),
        IMI_sec=("IMI_sec", "median"),
        FR_g_sec=("FR_g_sec", "mean"),
        N_Days=("Date", "nunique"),
    ).reset_index()
    out["TFB_Day"]  = out["TFB"] / out["N_Days"]
    out["FI_Day_g"] = out["FI_g"] / out["N_Days"]
    return out


def calc_production(feed, bw, individual_feeding):
    bw_terminal = (bw.sort_values(["Animal_ID", "Time1"])
                     .groupby("Animal_ID")
                     .agg(IBW_kg=("BW_kg", "first"), FBW_kg=("BW_kg", "last"),
                          IBW_Day=("Experimental_Day", "first"),
                          FBW_Day=("Experimental_Day", "last"),
                          N_W=("BW_kg", "size"))
                     .reset_index())
    bw_terminal["Gain_kg"] = bw_terminal["FBW_kg"] - bw_terminal["IBW_kg"]
    bw_terminal["Test_Days"] = bw_terminal["FBW_Day"] - bw_terminal["IBW_Day"]
    bw_terminal["ADG_g"] = np.where(bw_terminal["Test_Days"] > 0,
                                    bw_terminal["Gain_kg"] * 1000 / bw_terminal["Test_Days"], np.nan)

    # ADG 稳健过滤（4*MAD）
    adg = bw_terminal["ADG_g"]
    adg_med = adg.median()
    adg_mad = (adg - adg_med).abs().median() * 1.4826
    if pd.notna(adg_med) and pd.notna(adg_mad) and adg_mad > 0:
        mask = (adg - adg_med).abs() > 4 * adg_mad
        bw_terminal.loc[mask, "ADG_g"] = np.nan

    prod = individual_feeding.merge(bw_terminal, on="Animal_ID", how="left")
    prod["ADFI_g"] = prod["FI_g"] / prod["N_Days"]
    prod["MBW"] = (prod["IBW_kg"] + prod["FBW_kg"]) / 2
    prod["FCR"] = np.where(prod["Gain_kg"] > 0, prod["ADFI_g"] / prod["ADG_g"], np.nan)
    prod["Feed_Efficiency"] = 1 / prod["FCR"]

    # RFI
    sub = prod[prod["ADG_g"].notna() & prod["MBW"].notna() & prod["ADFI_g"].notna()].copy()
    if len(sub) >= 20:
        X = np.column_stack([np.ones(len(sub)), sub["ADG_g"], sub["MBW"]])
        y = sub["ADFI_g"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sub["RFI"] = y - X @ beta
        prod = prod.merge(sub[["Animal_ID", "RFI"]], on="Animal_ID", how="left")
    else:
        prod["RFI"] = np.nan
    return prod


def assign_hff_lff(prod, method="median", cutoff=None):
    d = prod[prod["TFB_Day"].notna()].copy()
    if cutoff is None or pd.isna(cutoff):
        cutoff = d["TFB_Day"].median()
    d["Feed_Frequency_Group"] = np.where(d["TFB_Day"] >= cutoff, "HFF", "LFF")
    return d, cutoff


# ---------- 节律 ----------
def calc_rhythm(bout):
    daily = (bout.groupby("Date")
                 .apply(lambda x: len(x) / x["Animal_ID"].nunique())
                 .reset_index(name="Mean_Bouts_Per_Duck"))
    b = bout.copy()
    b["Week_Number"] = make_experimental_week(b["Time1"], b["Time1"].min(), 7)["Week_Number"].values
    weekly = (b.groupby(["Week_Number", "Date"])
                .apply(lambda x: len(x) / x["Animal_ID"].nunique())
                .reset_index(name="Mean_Bouts_Per_Duck"))
    b["Hour_Block"] = b["Time1"].dt.hour
    hourly = (b.groupby(["Week_Number", "Hour_Block"])
                .apply(lambda x: len(x) / x["Animal_ID"].nunique())
                .reset_index(name="Mean_Bouts_Per_Duck"))
    return dict(daily=daily, weekly=weekly, hourly=hourly)


def calc_weekly_fcr(feed, bw, week_length=7):
    fd = (feed.assign(Date=feed["Time1"].dt.date)
              .groupby(["Animal_ID", "Date", "Week_Number"])["Feed_Intake_g"].sum()
              .reset_index(name="FI_day"))
    bd = (bw.assign(Date=bw["Time1"].dt.date)
            .sort_values(["Animal_ID", "Date"])
            .groupby(["Animal_ID", "Date"])["BW_kg"].last()
            .reset_index(name="BW_day"))
    m = fd.merge(bd, on=["Animal_ID", "Date"], how="left").sort_values(["Animal_ID", "Date"])
    m["BW_prev"] = m.groupby("Animal_ID")["BW_day"].shift(1)
    m["Gain_day"] = (m["BW_day"] - m["BW_prev"]) * 1000
    m = m[m["Gain_day"].notna() & m["FI_day"].notna() & (m["Gain_day"] > 0)]
    return (m.groupby(["Animal_ID", "Week_Number"])
             .apply(lambda x: x["FI_day"].sum() / x["Gain_day"].sum())
             .reset_index(name="Weekly_FCR"))


# ---------- 创新行为指标 ----------
def _clock_to_minutes(s):
    parts = [int(p) for p in str(s).split(":")[:2]]
    return parts[0] * 60 + parts[1]


def compute_daynight_v5(bout, day_start="06:00", day_end="18:00"):
    ds, de = _clock_to_minutes(day_start), _clock_to_minutes(day_end)
    mins = bout["Time1"].dt.hour * 60 + bout["Time1"].dt.minute
    if ds < de:
        is_day = (mins >= ds) & (mins < de)
    elif ds > de:
        is_day = (mins >= ds) | (mins < de)
    else:
        is_day = pd.Series(False, index=bout.index)
    b = bout.assign(_Is_Day=is_day)
    out = b.groupby("Animal_ID").apply(lambda x: pd.Series({
        "Day_FI_g":    x.loc[x["_Is_Day"], "Feed_Intake_g"].sum(),
        "Night_FI_g":  x.loc[~x["_Is_Day"], "Feed_Intake_g"].sum(),
        "Day_Bouts":   int(x["_Is_Day"].sum()),
        "Night_Bouts": int((~x["_Is_Day"]).sum()),
    })).reset_index()
    out["Total_FI_g"] = out["Day_FI_g"] + out["Night_FI_g"]
    out["Total_Bouts"] = out["Day_Bouts"] + out["Night_Bouts"]
    out["Day_FI_Ratio"] = np.where(out["Total_FI_g"] > 0,
                                   out["Day_FI_g"] / out["Total_FI_g"], np.nan)
    out["Day_Bout_Ratio"] = np.where(out["Total_Bouts"] > 0,
                                     out["Day_Bouts"] / out["Total_Bouts"], np.nan)
    return out


def compute_behavior_cv_v5(bout, min_bouts_single=20, min_days_daily=3):
    def _cv(g):
        n = len(g); nd = g["Date"].nunique()
        return pd.Series({
            "N_Bouts": n, "N_Days": nd,
            "CV_Duration": g["Feed_Duration_sec"].std(ddof=1) / g["Feed_Duration_sec"].mean()
                            if n >= min_bouts_single and g["Feed_Duration_sec"].mean() > 0 else np.nan,
            "CV_FR":       g["FR_g_sec"].std(ddof=1) / g["FR_g_sec"].mean()
                            if n >= min_bouts_single and g["FR_g_sec"].mean() > 0 else np.nan,
            "Robust_CV_IMI": (g["IMI_sec"].quantile(.75) - g["IMI_sec"].quantile(.25)) / g["IMI_sec"].median()
                              if n >= min_bouts_single and g["IMI_sec"].median() > 0 else np.nan,
        })
    single = bout.groupby("Animal_ID").apply(_cv).reset_index()

    daily = bout.groupby(["Animal_ID", "Date"]).agg(
        Daily_Bouts=("Feed_Intake_g", "size"),
        Daily_FI=("Feed_Intake_g", "sum"),
        Daily_TFD=("Feed_Duration_sec", "sum")).reset_index()

    def _dcv(g):
        n = len(g)
        return pd.Series({
            "N_Days_Daily": n,
            "CV_Daily_Bouts": g["Daily_Bouts"].std(ddof=1) / g["Daily_Bouts"].mean()
                              if n >= min_days_daily and g["Daily_Bouts"].mean() > 0 else np.nan,
            "CV_Daily_FI":    g["Daily_FI"].std(ddof=1) / g["Daily_FI"].mean()
                              if n >= min_days_daily and g["Daily_FI"].mean() > 0 else np.nan,
            "CV_Daily_TFD":   g["Daily_TFD"].std(ddof=1) / g["Daily_TFD"].mean()
                              if n >= min_days_daily and g["Daily_TFD"].mean() > 0 else np.nan,
        })
    daily_cv = daily.groupby("Animal_ID").apply(_dcv).reset_index()
    return single.merge(daily_cv, on="Animal_ID", how="left")


def compute_cosinor_v5(bout, min_bouts=20):
    b = bout.assign(Hour_Block=bout["Time1"].dt.hour)
    hourly = (b.groupby(["Animal_ID", "Hour_Block"]).size()
                .reset_index(name="n"))
    full_idx = pd.MultiIndex.from_product([hourly["Animal_ID"].unique(), range(24)],
                                          names=["Animal_ID", "Hour_Block"])
    hourly = (hourly.set_index(["Animal_ID", "Hour_Block"])
                    .reindex(full_idx, fill_value=0).reset_index())

    rows = []
    for aid, g in hourly.groupby("Animal_ID"):
        t = g["Hour_Block"].values.astype(float)
        y = g["n"].values.astype(float)
        if y.sum() < min_bouts:
            rows.append((aid, np.nan, np.nan, np.nan, np.nan)); continue
        X = np.column_stack([np.ones(24),
                             np.cos(2 * np.pi * t / 24),
                             np.sin(2 * np.pi * t / 24)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        A = np.hypot(beta[1], beta[2])
        peak = (np.arctan2(beta[2], beta[1]) * 24 / (2 * np.pi)) % 24
        rows.append((aid, beta[0], A, peak, r2))
    return pd.DataFrame(rows, columns=["Animal_ID", "Cosinor_M", "Cosinor_A",
                                       "Peak_Hour", "Cosinor_R2"])


def compute_fano_v5(bout):
    b = bout.assign(Hour_Block=bout["Time1"].dt.hour)
    cnt = (b.groupby(["Animal_ID", "Date", "Hour_Block"]).size()
             .reset_index(name="n"))
    full = (cnt.set_index(["Animal_ID", "Date", "Hour_Block"])["n"]
               .groupby(level=[0, 1])
               .apply(lambda s: s.reindex(range(24), fill_value=0))
               .reset_index(name="n"))
    out = full.groupby("Animal_ID")["n"].agg(
        N_Hour_Cells="size",
        Mean_Bouts_Per_Hour="mean",
        Var_Bouts_Per_Hour=lambda x: x.var(ddof=1) if len(x) > 1 else np.nan
    ).reset_index()
    out["Fano"] = np.where(out["Mean_Bouts_Per_Hour"] > 0,
                           out["Var_Bouts_Per_Hour"] / out["Mean_Bouts_Per_Hour"], np.nan)
    return out


# ============================================================
# 4. 遗传评估与留种
# ============================================================

def select_indicator_traits(prod, target_trait, min_r=0.3, max_n=5):
    target_trait = [t for t in target_trait if t and t != "---"]
    if not target_trait:
        return pd.DataFrame(columns=["Indicator", "Target", "N", "Spearman_r", "P", "Abs_r"])

    measured = [t for t in target_trait if t in prod.columns]
    if not measured:
        return select_lit_indicator_traits(target_trait, prod, max_n)

    cand = ["TFB", "TFB_Day", "FI_g", "FI_Day_g", "AMS_g", "TFD_sec", "AFBD_sec",
            "IMI_sec", "FR_g_sec", "IBW_kg", "FBW_kg", "Gain_kg", "ADG_g",
            "MBW", "Day_FI_Ratio", "Day_Bout_Ratio",
            "CV_Duration", "CV_FR", "CV_Daily_Bouts", "CV_Daily_FI", "Cosinor_A", "Fano"]
    exclude_map = {"FCR": "Feed_Efficiency", "RFI": "Feed_Efficiency"}
    for tr in measured:
        ex = exclude_map.get(tr)
        if ex:
            cand = [c for c in cand if c != ex]
    cand = [c for c in cand if c in prod.columns]
    if not cand:
        return pd.DataFrame(columns=["Indicator", "Target", "N", "Spearman_r", "P", "Abs_r"])

    rows = []
    for tr in measured:
        for v in cand:
            if v == tr:
                continue
            tmp = prod[[v, tr]].dropna()
            if len(tmp) >= 5:
                r, p = stats.spearmanr(tmp[v], tmp[tr])
                rows.append(dict(Indicator=v, Target=tr, N=len(tmp),
                                 Spearman_r=r, P=p))
    if not rows:
        return pd.DataFrame(columns=["Indicator", "Target", "N", "Spearman_r", "P", "Abs_r"])

    res = pd.DataFrame(rows)
    res["Abs_r"] = res["Spearman_r"].abs()
    res = res[res["Abs_r"] >= min_r]
    res = (res.sort_values("Abs_r", ascending=False)
              .drop_duplicates("Indicator", keep="first")
              .head(max_n)
              .reset_index(drop=True))
    return res


def build_A_matrix(ped):
    """Henderson 递归法构建亲缘矩阵 A"""
    animal_ids = list(ped["ID"].dropna().astype(str))
    founder = pd.unique(pd.concat([ped["Sire_Cage"], ped["Dam_Cage"]]).dropna().astype(str))
    animal_set = set(animal_ids)
    founder = [f for f in founder if f not in animal_set]
    order = founder + animal_ids
    idx = {aid: i for i, aid in enumerate(order)}
    n = len(order)
    A = np.zeros((n, n))
    ped_map = ped.set_index("ID")

    for i, aid in enumerate(order):
        if aid not in animal_set:
            A[i, i] = 1.0
            continue
        row = ped_map.loc[aid]
        s = idx.get(str(row["Sire_Cage"]), None) if pd.notna(row["Sire_Cage"]) else None
        d = idx.get(str(row["Dam_Cage"]), None) if pd.notna(row["Dam_Cage"]) else None
        if s is not None and d is not None:
            A[i, i] = 1 + 0.5 * A[s, d]
            for j in range(i):
                A[i, j] = A[j, i] = 0.5 * (A[j, s] + A[j, d])
        elif s is not None or d is not None:
            p = s if s is not None else d
            A[i, i] = 1 + 0.5 * A[p, p]
            for j in range(i):
                A[i, j] = A[j, i] = 0.5 * A[j, p]
        else:
            A[i, i] = 1.0
    return pd.DataFrame(A, index=order, columns=order)


def run_pblup(prod, ped, trait, h2=0.3, fixed_effects=("Sex",), id_col="Animal_ID"):
    dat = prod[prod[trait].notna()].copy()
    ped_sub = ped[["ID", "Sex"]] if "Sex" in ped.columns else ped[["ID"]]

    if "Sex" in dat.columns:
        dat = dat.merge(ped_sub, left_on=id_col, right_on="ID",
                        how="left", suffixes=("", "_ped"))
        if "Sex_ped" in dat.columns:
            dat["Sex"] = dat["Sex"].fillna(dat["Sex_ped"])
            dat.drop(columns=["Sex_ped"], inplace=True)
        if "ID" in dat.columns and id_col != "ID":
            dat.drop(columns=["ID"], inplace=True)
    else:
        dat = dat.merge(ped_sub, left_on=id_col, right_on="ID", how="left")

    if len(dat) < 10:
        raise ValueError("PBLUP 需要至少 10 个有表型的个体。")

    A = build_A_matrix(ped)
    animals = [a for a in dat[id_col].astype(str) if a in A.index]
    dat = dat[dat[id_col].astype(str).isin(animals)].reset_index(drop=True)
    if len(dat) < 10:
        raise ValueError("PBLUP 需要至少 10 个同时有表型和系谱的个体。")

    y = dat[trait].values.astype(float)
    n = len(y)

    fe_cols = [c for c in fixed_effects if c in dat.columns]
    if fe_cols:
        X = pd.get_dummies(dat[fe_cols].astype(str), drop_first=True).values.astype(float)
        X = np.column_stack([np.ones(n), X])
    else:
        X = np.ones((n, 1))

    Z = np.zeros((n, len(animals)))
    for i, a in enumerate(animals):
        Z[dat[id_col].astype(str).values == a, i] = 1

    A_sub = A.loc[animals, animals].values
    try:
        Ai = np.linalg.inv(A_sub)
    except np.linalg.LinAlgError:
        Ai = np.linalg.pinv(A_sub)

    lam = (1 - h2) / h2
    LHS = np.block([[X.T @ X, X.T @ Z],
                    [Z.T @ X, Z.T @ Z + lam * Ai]])
    RHS = np.concatenate([X.T @ y, Z.T @ y])
    sol = np.linalg.solve(LHS, RHS)
    b = sol[:X.shape[1]]
    u = sol[X.shape[1]:]

    out = pd.DataFrame({"Animal_ID": animals, "EBV": u})
    out = out.merge(dat[[id_col, trait] + (["Sex"] if "Sex" in dat.columns else [])],
                    left_on="Animal_ID", right_on=id_col, how="left")
    return dict(result=out, h2=h2, lambda_=lam, n_animals=len(animals),
                fixed=fe_cols, A=A)


def run_pblup_multi(prod, ped, traits, h2_by_trait=None, fixed_effects=("Sex",),
                    id_col="Animal_ID"):
    if not traits:
        raise ValueError("未选择参与指数的性状。")
    parts = []
    for tr in traits:
        if tr not in prod.columns:
            continue
        h2i = h2_by_trait.get(tr, 0.3) if h2_by_trait else 0.3
        try:
            pb = run_pblup(prod, ped, tr, h2=h2i, fixed_effects=fixed_effects, id_col=id_col)
        except Exception:
            continue
        r = pb["result"].copy()
        r["Trait"] = tr
        r["H2_Used"] = h2i
        parts.append(r)
    if not parts:
        raise ValueError("所有勾选性状的 PBLUP 均失败。")
    ebv = pd.concat(parts, ignore_index=True)
    return dict(ebv=ebv, n_animals=ebv["Animal_ID"].nunique(),
                traits=list(ebv["Trait"].unique()))


def estimate_h2_reml(prod, ped, trait, fixed_effects=("Sex",), id_col="Animal_ID"):
    """直接最大化 REML 对数似然（对数参数化）"""
    dat = prod[prod[trait].notna()].copy()
    if len(dat) < 10:
        raise ValueError("遗传力估计需要至少 10 个有表型的个体。")

    A = build_A_matrix(ped)
    animals = [a for a in dat[id_col].astype(str) if a in A.index]
    dat = dat[dat[id_col].astype(str).isin(animals)].reset_index(drop=True)
    if len(dat) < 10:
        raise ValueError("遗传力估计需要至少 10 个同时有表型和系谱的个体。")

    y = dat[trait].values.astype(float)
    n = len(y)
    fe_cols = [c for c in fixed_effects if c in dat.columns]
    if fe_cols:
        X = pd.get_dummies(dat[fe_cols].astype(str), drop_first=True).values.astype(float)
        X = np.column_stack([np.ones(n), X])
    else:
        X = np.ones((n, 1))

    Z = np.zeros((n, len(animals)))
    for i, a in enumerate(animals):
        Z[dat[id_col].astype(str).values == a, i] = 1

    A_sub = A.loc[animals, animals].values
    ZA = Z @ A_sub
    var_y = np.var(y, ddof=1)

    def reml_nll(par):
        s2a, s2e = np.exp(par[0]), np.exp(par[1])
        V = ZA @ Z.T * s2a + np.eye(n) * s2e
        try:
            Vi = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            Vi = np.linalg.pinv(V)
        M = X.T @ Vi @ X
        sign_v, logdetV = np.linalg.slogdet(V)
        sign_m, logdetM = np.linalg.slogdet(M)
        if sign_v <= 0 or sign_m <= 0:
            return 1e12
        ytVi = y @ Vi
        ytPy = ytVi @ y - ytVi @ X @ np.linalg.solve(M, X.T @ Vi @ y)
        return 0.5 * (logdetV + logdetM + ytPy)

    par0 = np.log([0.3 * var_y, 0.7 * var_y])
    res = minimize(reml_nll, par0, method="Nelder-Mead",
                   options=dict(maxiter=1000, xatol=1e-8, fatol=1e-8))
    s2a, s2e = np.exp(res.x[0]), np.exp(res.x[1])
    h2 = s2a / (s2a + s2e)
    return dict(trait=trait, h2=h2, sigma2_a=s2a, sigma2_e=s2e,
                n_animals=len(animals), n_obs=n,
                n_iter=int(res.nit), converged=res.success)


def estimate_h2_multi(prod, ped, traits, fixed_effects=("Sex",), id_col="Animal_ID"):
    rows = []
    for tr in traits:
        if tr not in prod.columns:
            continue
        try:
            r = estimate_h2_reml(prod, ped, tr, fixed_effects, id_col)
            rows.append(dict(Trait=r["trait"], h2=r["h2"],
                             sigma2_a=r["sigma2_a"], sigma2_e=r["sigma2_e"],
                             N_Animals=r["n_animals"], N_Obs=r["n_obs"],
                             N_Iter=r["n_iter"], Converged=r["converged"]))
        except Exception:
            continue
    if not rows:
        raise ValueError("所有性状的遗传力估计均失败。")
    return pd.DataFrame(rows)


# ---------- 文献参数 ----------
def literature_params():
    return pd.DataFrame([
        ("FCR", 0.29, "北京鸭 Pekin duck", "Li et al. 2020, Poultry Science 99:2375-2384"),
        ("RFI", 0.41, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("ADG_g", 0.38, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("FBW_kg", 0.39, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("FI_Day_g", 0.31, "北京鸭 Pekin duck", "Li et al. 2020, Poultry Science"),
        ("FR_g_sec", 0.54, "北京鸭 Pekin duck", "Li 2020 / Chapuis 2024"),
        ("TFB_Day", 0.54, "北京鸭 Pekin duck", "Li 2020 / Chapuis 2024"),
        ("AMS_g", 0.54, "北京鸭 Pekin duck", "Li 2020 / Chapuis 2024"),
        ("ADFI_g", 0.31, "北京鸭 Pekin duck", "Li 2020 / Chapuis 2024"),
        ("IBW_kg", 0.39, "北京鸭 Pekin duck", "参考 Zhang 2017 BW42"),
        ("SkinFat_Rate", 0.55, "北京鸭×绿头鸭 F2", "Cai et al. 2023, JASB 14:88"),
        ("AbFat_Rate", 0.56, "北京鸭×绿头鸭 F2", "Cai et al. 2023, JASB 14:88"),
        ("BMP", 0.38, "北京鸭×绿头鸭 F2", "Cai et al. 2023, JASB 14:88"),
        ("AbFat_Wt", 0.63, "北京鸭×绿头鸭 F2", "Cai et al. 2023, JASB 14:88"),
        ("SkinFat_Wt", 0.60, "北京鸭×绿头鸭 F2", "Cai et al. 2023, JASB 14:88"),
    ], columns=["Trait", "h2", "Breed", "Source"])


def literature_corr():
    return pd.DataFrame([
        ("RFI", "FI_Day_g", 0.77, 0.17, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("RFI", "FCR", 0.54, 0.05, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("FCR", "ADG_g", -0.80, 0.11, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("FCR", "FI_Day_g", 0.54, 0.05, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("ADG_g", "FBW_kg", 0.92, 0.08, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("ADG_g", "FI_Day_g", 0.49, 0.15, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("FBW_kg", "ADG_g", 0.92, 0.08, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("FBW_kg", "FCR", -0.64, 0.14, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("FI_Day_g", "RFI", 0.77, 0.17, "北京鸭 Pekin duck", "Zhang et al. 2017, AJAS"),
        ("TFB_Day", "AMS_g", -0.91, 0.05, "北京鸭 Pekin duck", "Li et al. 2021, BMC Genomics"),
        ("TFB_Day", "AFBD_sec", -0.65, 0.13, "北京鸭 Pekin duck", "Li et al. 2021, BMC Genomics"),
        ("AMS_g", "AFBD_sec", 0.73, 0.11, "北京鸭 Pekin duck", "Li et al. 2021, BMC Genomics"),
        ("AFBD_sec", "TFD_sec", 0.60, 0.11, "北京鸭 Pekin duck", "Li et al. 2021, BMC Genomics"),
        ("TFD_sec", "AFBD_sec", 0.60, 0.11, "北京鸭 Pekin duck", "Li et al. 2021, BMC Genomics"),
        ("SkinFat_Rate", "RFI", 0.58, 0.159, "肉鸡 Broiler（替代证据）", "Chen et al. 2021, Poultry Science"),
        ("SkinFat_Rate", "FCR", 0.51, 0.17, "肉鸡 Broiler（替代证据）", "Chen et al. 2021, Poultry Science"),
        ("AbFat_Rate", "RFI", 0.58, 0.159, "肉鸡 Broiler（替代证据）", "Chen et al. 2021, Poultry Science"),
        ("AbFat_Rate", "FCR", 0.51, 0.17, "肉鸡 Broiler（替代证据）", "Chen et al. 2021, Poultry Science"),
    ], columns=["Target", "Indicator", "Genetic_r", "SE", "Species", "Source"])


def select_lit_indicator_traits(targets, prod, max_n=5):
    targets = [t for t in targets if t and t != "---"]
    if not targets:
        return pd.DataFrame(columns=["Indicator", "Target", "N", "Spearman_r",
                                     "P", "Abs_r", "rG_Source"])
    lc = literature_corr()
    rows = []
    for tr in targets:
        sub = lc[(lc["Target"] == tr) & (lc["Indicator"].isin(prod.columns))].copy()
        if len(sub) == 0:
            continue
        sub["Target"] = tr
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["Indicator", "Target", "N", "Spearman_r",
                                     "P", "Abs_r", "rG_Source"])
    res = pd.concat(rows, ignore_index=True)
    res["Abs_r"] = res["Genetic_r"].abs()
    res = (res.sort_values("Abs_r", ascending=False)
              .drop_duplicates("Indicator", keep="first")
              .head(max_n)
              .reset_index(drop=True))
    res["N"] = np.nan
    res["Spearman_r"] = np.nan
    res["P"] = np.nan
    res["rG_Source"] = res["Source"] + "；" + res["Species"]
    return res


def run_literature_index(prod, target_trait, lit, weights=None, traits_subset=None):
    dat = prod.copy()
    targets = [t for t in target_trait if t in prod.columns]
    if targets:
        dat = prod[prod[targets[0]].notna()].copy()

    traits = [t for t in lit["Trait"].tolist() if t in dat.columns]
    if traits_subset is not None:
        traits = [t for t in traits if t in traits_subset]
    if not traits:
        raise ValueError("文献参数表与数据无可匹配性状。")

    if weights is None:
        weights = {t: 1.0 / len(traits) for t in traits}

    ebv_list = []
    for tr in traits:
        h2 = float(lit.loc[lit["Trait"] == tr, "h2"].iloc[0])
        p = dat[tr]
        m, s = p.mean(), p.std(ddof=1)
        if pd.notna(s) and s > 0:
            ebv = h2 * (p - m) / s
        else:
            ebv = pd.Series(np.nan, index=p.index)
        ebv_list.append(pd.DataFrame({"Animal_ID": dat["Animal_ID"].values,
                                      "Trait": tr, "EBV": ebv.values}))
    ebv = pd.concat(ebv_list, ignore_index=True)
    ebv["W"] = ebv["Trait"].map(weights).fillna(0.0)
    idx = (ebv.assign(Contrib=ebv["EBV"] * ebv["W"])
              .groupby("Animal_ID")["Contrib"].sum()
              .reset_index(name="Index"))
    h2_used = lit[lit["Trait"].isin(traits)][["Trait", "h2"]].reset_index(drop=True)
    return dict(index=idx, ebv=ebv, h2_used=h2_used)


# ---------- 选择指数 ----------
TRAIT_META = pd.DataFrame([
    # Trait, Label, Dir
    ("FCR", "饲料转化比", -1), ("RFI", "剩余采食量", -1), ("ADFI_g", "平均日采食量", -1),
    ("FI_Day_g", "日采食量", -1), ("TFD_sec", "总采食时长", -1), ("AFBD_sec", "单次采食时长", -1),
    ("IMI_sec", "餐间间隔", -1), ("FR_g_sec", "采食速率", 1), ("TFB_Day", "日访饲次数", 1),
    ("AMS_g", "单次采食量", 1), ("IBW_kg", "初重", 1), ("FBW_kg", "终重", 1),
    ("Gain_kg", "总增重", 1), ("ADG_g", "日增重", 1), ("MBW", "代谢体重", 1),
    ("Day_FI_Ratio", "白天采食占比", 1), ("Day_Bout_Ratio", "白天访饲占比", 1),
    ("CV_Duration", "采食时长变异", -1), ("CV_FR", "采食速率变异", -1),
    ("CV_Daily_Bouts", "日访饲次数变异", -1), ("CV_Daily_FI", "日采食量变异", -1),
    ("Cosinor_A", "节律振幅", 1), ("Fano", "Fano指数", -1),
    ("Feed_Efficiency", "饲料效率", 1),
    ("SkinFat_Rate", "皮脂率", 1), ("AbFat_Rate", "腹脂率", -1), ("BMP", "胸肌率", 1),
], columns=["Trait", "Label", "Dir"])


def trait_dir(traits):
    m = dict(zip(TRAIT_META["Trait"], TRAIT_META["Dir"]))
    return [m.get(t, 1) for t in traits]


def trait_label(traits):
    m = dict(zip(TRAIT_META["Trait"], TRAIT_META["Label"]))
    return [m.get(t, t) for t in traits]


INDEX_TRAIT_CHOICES = {
    "效率与增重": ["FCR", "RFI", "ADG_g", "FBW_kg", "Gain_kg", "IBW_kg", "MBW"],
    "采食行为":   ["FI_Day_g", "ADFI_g", "TFB_Day", "AMS_g", "FR_g_sec",
                   "TFD_sec", "AFBD_sec", "IMI_sec"],
    "节律与稳定性": ["Day_FI_Ratio", "Day_Bout_Ratio", "CV_Duration", "CV_FR",
                     "CV_Daily_Bouts", "CV_Daily_FI", "Cosinor_A", "Fano"],
}
INDEX_TRAIT_VALUES = [v for vs in INDEX_TRAIT_CHOICES.values() for v in vs]


def selection_index(ebv_result, weights=None):
    if "Trait" in ebv_result.columns:
        traits = list(ebv_result["Trait"].unique())
        if weights is None:
            weights = {t: 1.0 for t in traits}
        d = pd.DataFrame({"Trait": traits,
                          "Dir": trait_dir(traits),
                          "W": [weights.get(t, 0) for t in traits]})
        if (d["W"] == 0).all():
            d["W"] = 1.0
        ebv = ebv_result.merge(d, on="Trait", how="left")
        ebv["EBV_sd"] = ebv.groupby("Trait")["EBV"].transform(lambda x: x.std(ddof=1))
        ebv["EBV_z"] = np.where(ebv["EBV_sd"].notna() & (ebv["EBV_sd"] > 0),
                                ebv["EBV"] / ebv["EBV_sd"], 0.0)
        ebv["Contrib"] = ebv["EBV_z"] * ebv["Dir"] * ebv["W"]
        idx = ebv.groupby("Animal_ID")["Contrib"].sum().reset_index(name="Index")
    else:
        idx = ebv_result[["Animal_ID", "EBV"]].rename(columns={"EBV": "Index"})
    idx = idx.sort_values("Index", ascending=False).reset_index(drop=True)
    idx["Rank"] = idx.index + 1
    return idx


def make_retention_list(index_df, prod, retention_ratio=0.3, sex_balance=False,
                        key_col="Animal_ID", prod_key="Animal_ID",
                        phenotype_cols=("TFB_Day", "FI_Day_g", "ADG_g", "FCR", "RFI", "Sex")):
    idx = index_df.sort_values("Index", ascending=False).reset_index(drop=True)
    n_total = len(idx)
    n_keep = max(1, round(n_total * retention_ratio))

    pheno_cols = [c for c in phenotype_cols if c in prod.columns]
    if pheno_cols:
        prod_sub = prod[[prod_key] + pheno_cols].drop_duplicates(prod_key).copy()
        prod_sub = prod_sub.rename(columns={prod_key: key_col})
        idx = idx.merge(prod_sub, on=key_col, how="left")

    sexes = []
    if "Sex" in idx.columns:
        sexes = [s for s in idx["Sex"].dropna().unique() if str(s).strip() != ""]

    if sex_balance and len(sexes) == 2:
        per = int(np.ceil(n_keep / 2))
        keep = (idx[idx["Sex"].isin(sexes)]
                  .sort_values("Index", ascending=False)
                  .groupby("Sex").head(per)
                  .sort_values("Index", ascending=False)
                  .head(n_keep))
        if len(keep) < n_keep:
            fill = idx[~idx[key_col].isin(keep[key_col])].head(n_keep - len(keep))
            keep = pd.concat([keep, fill], ignore_index=True)
        sex_bal = True
    else:
        keep = idx.head(n_keep).copy()
        sex_bal = False

    keep = keep.copy(); keep["Retained"] = True
    not_keep = idx[~idx[key_col].isin(keep[key_col])].copy(); not_keep["Retained"] = False
    return dict(retained=keep, not_retained=not_keep, n_keep=len(keep),
                n_target=n_keep, total=n_total, ratio=retention_ratio,
                sex_balanced=sex_bal)


def run_retention_module(prod, link, cfg):
    warnings_list = []
    target = cfg.get("target_trait", ["FCR"])
    if isinstance(target, str):
        target = [target]
    target = [t for t in target if t and t != "---"]
    req_traits = cfg.get("index_traits") or target

    not_in_prod = [t for t in req_traits if t not in prod.columns]
    if not_in_prod:
        warnings_list.append(
            f"性状 [{', '.join(not_in_prod)}] 本场未测量（多为屠宰性状），已从指数中剔除。")

    index_traits = [t for t in req_traits if t in prod.columns]
    ind_traits = select_indicator_traits(prod, target,
                                         cfg.get("indicator_min_r", 0.3),
                                         cfg.get("indicator_max_n", 5))
    if not index_traits:
        return dict(genetic=None, retention=None, indicator_traits=ind_traits,
                    key_metrics=None, index_traits=index_traits,
                    warnings=warnings_list,
                    error="参与指数的性状均无本场数据，无法构建选择指数。")

    n_has_ped = int(link["Has_Pedigree"].sum()) if "Has_Pedigree" in link.columns else 0
    use_ped = bool(cfg.get("use_pedigree", False)) and n_has_ped >= 20
    if cfg.get("use_pedigree") and not use_ped:
        warnings_list.append(
            f"系谱可用个体不足 20（当前 {n_has_ped} 只），已自动切换为『文献参数』路线。")

    genetic = dict(mode="PBLUP" if use_ped else "文献参数")
    retention = None    idx = None
    ratio = cfg.get("retention_ratio", 0.3)
    sex_balance = bool(cfg.get("sex_balance", False))

    if use_ped:
        ped_animals = link[link["Has_Pedigree"]][["ID", "Sire_Cage", "Dam_Cage", "Sex"]].copy()
        prod_gen = prod.merge(
            link[["eID", "ID"]].drop_duplicates("eID"),
            left_on="Animal_ID", right_on="eID", how="left")
        prod_gen["Wing_ID"] = prod_gen["ID"].fillna(prod_gen["Animal_ID"])
        prod_gen = prod_gen.drop(columns=["ID"], errors="ignore")

        h2_by_trait = cfg.get("h2_by_trait")
        h2_est = None
        if cfg.get("use_reml_h2"):
            try:
                h2_est = estimate_h2_multi(prod_gen, ped_animals, index_traits,
                                           fixed_effects=cfg.get("fixed_effects", ("Sex",)),
                                           id_col="Wing_ID")
                h2_by_trait = dict(zip(h2_est["Trait"], h2_est["h2"]))
            except Exception:
                h2_est = None
        genetic["h2_est"] = h2_est
        genetic["h2_used"] = h2_by_trait

        try:
            pblup = run_pblup_multi(prod_gen, ped_animals, index_traits,
                                    h2_by_trait=h2_by_trait,
                                    fixed_effects=cfg.get("fixed_effects", ("Sex",)),
                                    id_col="Wing_ID")
        except Exception:
            pblup = None
        genetic["pblup"] = pblup

        if pblup is not None:
            ebv = pblup["ebv"].merge(
                link[["ID", "eID"]].drop_duplicates("ID"),
                left_on="Animal_ID", right_on="ID", how="left")
            idx = selection_index(ebv, cfg.get("weights"))
            idx = idx.merge(link[["eID", "ID"]].drop_duplicates("ID"),
                            left_on="Animal_ID", right_on="ID", how="left")
            retention = make_retention_list(idx, prod_gen, ratio, sex_balance,
                                            key_col="Animal_ID", prod_key="Wing_ID")
            genetic["ebv_tbl"] = ebv
        else:
            warnings_list.append("PBLUP 求解失败，未生成留种名单。")
    else:
        lit = literature_params()
        li = run_literature_index(prod, target, lit,
                                  cfg.get("weights"), traits_subset=index_traits)
        genetic["literature"] = li
        idx = selection_index(li["ebv"], cfg.get("weights"))
        retention = make_retention_list(idx, prod, ratio, sex_balance)
        genetic["ebv_tbl"] = li["ebv"]

    genetic["index"] = idx
    genetic["retention"] = retention
    genetic["index_traits"] = index_traits

    targets_meas = [t for t in target if t in prod.columns]
    if targets_meas:
        n_valid = int(prod[targets_meas[0]].notna().sum())
    elif index_traits:
        n_valid = int(prod[index_traits[0]].notna().sum())
    else:
        n_valid = 0

    if genetic.get("pblup") is not None:
        h2_used = genetic.get("h2_used") or cfg.get("h2_by_trait")
        src = "实测REML" if genetic.get("h2_est") is not None else "文献/输入"
        if h2_used:
            h2_txt = f"[{src}] " + "; ".join(f"{k}={v:.2f}" for k, v in h2_used.items())
        else:
            h2_txt = "-"
    elif genetic.get("literature") is not None:
        lu = genetic["literature"]["h2_used"]
        h2_txt = "[文献] " + "; ".join(f"{r.Trait}={r.h2:.2f}" for r in lu.itertuples())
    else:
        h2_txt = "-"

    key_metrics = pd.DataFrame({
        "指标": ["目标性状", "参与指数性状", "可用个体数", "系谱覆盖",
                 "评估模式", "指示性状数", "遗传力(h²)", "留种数", "留种比例"],
        "数值": [" + ".join(target),
                 " + ".join(index_traits),
                 str(n_valid),
                 f"{n_has_ped} / {len(link)}",
                 genetic["mode"],
                 str(len(ind_traits)),
                 h2_txt,
                 str(retention["n_keep"]) if retention else "-",
                 f"{ratio*100:.0f}%" if retention else "-"],
    })
    return dict(genetic=genetic, retention=retention, indicator_traits=ind_traits,
                key_metrics=key_metrics, index_traits=index_traits,
                warnings=warnings_list, error=None)


# ============================================================
# 5. 主流程
# ============================================================
def run_pipeline_v5(feed_files, bw_files, ped_path, idmap_path, cfg):
    import time
    t0 = time.time()
    feed_raw, feed_qc = read_matching_excel_files(feed_files, "feed")
    bw_raw, bw_qc = read_matching_excel_files(bw_files, "weight")
    ped = read_pedigree(ped_path)
    idmap = read_idmap(idmap_path)

    clean = run_clean_v5(feed_raw, bw_raw, cfg)
    bout = build_bouts(clean["feed"], cfg["imi_threshold"], cfg["min_intake"])
    ind_feeding = calc_individual_feeding(bout)
    prod = calc_production(clean["feed"], clean["bw"], ind_feeding)
    prod, _ = assign_hff_lff(prod, cfg.get("hff_method", "median"), cfg.get("hff_cutoff"))
    prod["HFF"] = (prod["Feed_Frequency_Group"] == "HFF").astype(int)

    prod = (prod.merge(compute_daynight_v5(bout, cfg.get("day_start", "06:00"),
                                            cfg.get("day_end", "18:00")),
                        on="Animal_ID", how="left")
                 .merge(compute_behavior_cv_v5(bout), on="Animal_ID", how="left")
                 .merge(compute_cosinor_v5(bout), on="Animal_ID", how="left")
                 .merge(compute_fano_v5(bout), on="Animal_ID", how="left"))

    rhythm = calc_rhythm(bout)
    weekly_fcr = calc_weekly_fcr(clean["feed"], clean["bw"], cfg["week_length"])

    link = link_animals(clean["feed"], clean["bw"], ped, idmap)
    if "eID" in link.columns and "Sex" in link.columns:
        prod = prod.merge(link[["eID", "Sex"]].drop_duplicates("eID"),
                          left_on="Animal_ID", right_on="eID", how="left")

    ind_traits = select_indicator_traits(prod, cfg.get("target_trait", ["FCR"]),
                                          cfg.get("indicator_min_r", 0.3),
                                          cfg.get("indicator_max_n", 5))
    genetic, retention, key_metrics = None, None, None
    if cfg.get("run_retention"):
        rr = run_retention_module(prod, link, cfg)
        genetic = rr["genetic"]; retention = rr["retention"]
        key_metrics = rr["key_metrics"]; ind_traits = rr["indicator_traits"]

    return dict(feed_raw_qc=feed_qc, bw_raw_qc=bw_qc, clean=clean, bout=bout,
                feeding=ind_feeding, prod=prod, rhythm=rhythm,
                weekly_fcr=weekly_fcr, link=link, indicator_traits=ind_traits,
                genetic=genetic, retention=retention, key_metrics=key_metrics,
                elapsed=time.time() - t0)


# ============================================================
# 6. AI 智能选种探索
# ============================================================
def ml_feature_groups_v5():
    return {
        "基础采食": ["TFB", "TFB_Day", "FI_g", "FI_Day_g", "AMS_g"],
        "单次采食": ["TFD_sec", "AFBD_sec", "IMI_sec", "FR_g_sec"],
        "昼夜分配": ["Day_FI_g", "Night_FI_g", "Day_Bouts", "Night_Bouts",
                     "Total_FI_g", "Total_Bouts", "Day_FI_Ratio", "Day_Bout_Ratio"],
        "行为稳定性": ["CV_Duration", "CV_FR", "Robust_CV_IMI",
                       "CV_Daily_Bouts", "CV_Daily_FI", "CV_Daily_TFD"],
        "节律": ["Cosinor_M", "Cosinor_A", "Cosinor_R2", "Peak_Hour"],
        "聚集度": ["Mean_Bouts_Per_Hour", "Var_Bouts_Per_Hour", "Fano"],
    }


def ml_features_v5(prod):
    feats = [f for grp in ml_feature_groups_v5().values() for f in grp]
    return [f for f in feats if f in prod.columns]


def ml_cluster(prod, seed=123):
    cluster_features = [f for f in ["TFB", "FI_g", "AMS_g", "IMI_sec",
                                     "Robust_CV_IMI", "Fano"] if f in prod.columns]
    if len(cluster_features) < 3:
        return dict(error="用于聚类的行为指标不足（至少 3 个）。")

    d = prod[["Animal_ID"] + cluster_features].copy()
    for c in cluster_features:
        d[c] = d[c].fillna(d[c].median())

    X = d[cluster_features].values.astype(float)
    if "AMS_g" in cluster_features:
        X[:, cluster_features.index("AMS_g")] *= -1
    X = (X - X.mean(0)) / (X.std(0, ddof=0) + 1e-12)

    sil_results = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=25, random_state=seed).fit(X)
        sil = silhouette_score(X, km.labels_)
        sil_results.append(dict(K=k, Silhouette=sil))
    sil_df = pd.DataFrame(sil_results)
    best_k = int(sil_df.loc[sil_df["Silhouette"].idxmax(), "K"])

    km = KMeans(n_clusters=best_k, n_init=50, random_state=seed).fit(X)
    out = d.copy()
    out["Cluster"] = km.labels_ + 1

    prod_cols = [c for c in ["Animal_ID", "ADG_g", "FCR", "RFI", "TFB_Day"] if c in prod.columns]
    out = out.merge(prod[prod_cols], on="Animal_ID", how="left")
    if "Feed_Frequency_Group" in prod.columns:
        out = out.merge(prod[["Animal_ID", "Feed_Frequency_Group"]], on="Animal_ID", how="left")
    else:
        out["Feed_Frequency_Group"] = None

    summ = out.groupby("Cluster").agg(
        N=("Animal_ID", "size"), Mean_TFB=("TFB", "mean"),
        Mean_FI=("FI_g", "mean"), Mean_AMS=("AMS_g", "mean")).reset_index()
    hff_cluster = int(summ.loc[summ["Mean_TFB"].idxmax(), "Cluster"])
    out["Feeding_Pattern"] = np.where(out["Cluster"] == hff_cluster, "HFF", "LFF")

    compare = out.groupby("Feeding_Pattern").agg(
        N=("Animal_ID", "size"), ADG_mean=("ADG_g", "mean"),
        FCR_mean=("FCR", "mean"), RFI_mean=("RFI", "mean")).reset_index()

    return dict(result=out, summary=summ, silhouette=sil_df,
                best_k=best_k, compare=compare, features=cluster_features)


def ml_single_target(prod, target, features, seed=123, ntree=300, nrounds=120):
    cols = ["Animal_ID"] + features + [target]
    ml_data = prod[cols].dropna()
    if len(ml_data) < 30:
        return dict(error=f"{target} 有效个体不足 30（{len(ml_data)} 只），该目标跳过。")

    cor_rows = []
    for f in features:
        r, p = stats.spearmanr(ml_data[f], ml_data[target])
        cor_rows.append(dict(Feature=f, Correlation=r, P_value=p))
    cor_df = pd.DataFrame(cor_rows)
    cor_df["Abs_Correlation"] = cor_df["Correlation"].abs()
    cor_df = cor_df.sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)

    X = ml_data[features].values.astype(float)
    y = ml_data[target].values.astype(float)
    idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
    Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]

    rf = RandomForestRegressor(n_estimators=ntree, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    imp_df = pd.DataFrame({"Feature": features, "IncMSE": rf.feature_importances_})
    imp_df = imp_df.sort_values("IncMSE", ascending=False).reset_index(drop=True)

    eval_df, pred_df, xgb_ok = None, None, False
    try:
        import xgboost as xgb
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dtest = xgb.DMatrix(Xte, label=yte)
        params = dict(objective="reg:squarederror", eval_metric="rmse",
                      eta=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8)
        model = xgb.train(params, dtrain, num_boost_round=nrounds)
        pred = model.predict(dtest)
        eval_df = pd.DataFrame([dict(Target=target, Model="XGBoost",
                                     R2=float(np.corrcoef(yte, pred)[0, 1] ** 2),
                                     RMSE=float(np.sqrt(np.mean((yte - pred) ** 2))),
                                     MAE=float(np.mean(np.abs(yte - pred))))])
        pred_df = pd.DataFrame({"Animal_ID": ml_data["Animal_ID"].values[idx_te],
                                "Actual": yte, "Predicted": pred,
                                "Error": yte - pred})
        xgb_ok = True
    except Exception:
        pass

    return dict(cor=cor_df, importance=imp_df, evaluation=eval_df,
                prediction=pred_df, n=len(ml_data), target=target, xgb_ok=xgb_ok)


def ml_fusion(target_results, w_target=None):
    names = [k for k, v in target_results.items() if v and "importance" in v]
    if not names:
        return dict(error="所有目标的 RF 重要性均失败，无法融合。")
    if w_target is None or len(w_target) != len(names):
        w_target = [1.0 / len(names)] * len(names)

    merged = None
    for i, nm in enumerate(names):
        df = target_results[nm]["importance"][["Feature", "IncMSE"]].copy()
        df = df.rename(columns={"IncMSE": f"{nm}_importance"})
        merged = df if merged is None else merged.merge(df, on="Feature", how="outer")
    merged = merged.fillna(0)

    for i, nm in enumerate(names):
        col = f"{nm}_importance"
        v = merged[col]
        merged[f"{nm}_score"] = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else 0

    merged["Trait_score"] = sum(w * merged[f"{nm}_score"]
                                for w, nm in zip(w_target, names))
    merged = merged.sort_values("Trait_score", ascending=False).reset_index(drop=True)
    merged["Rank"] = merged.index + 1
    return merged


def ml_stability(prod, targets, features, n_rep=10, ntree=200, seed0=1):
    parts = []
    for trait in targets:
        cols = ["Animal_ID"] + features + [trait]
        ml_data = prod[cols].dropna()
        if len(ml_data) < 30:
            continue
        for r in range(n_rep):
            rf = RandomForestRegressor(n_estimators=ntree,
                                       random_state=seed0 + r, n_jobs=-1)
            rf.fit(ml_data[features].values, ml_data[trait].values)
            imp = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_})
            imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)
            imp["Rank"] = imp.index + 1
            imp["Seed"] = r
            imp["Trait"] = trait
            parts.append(imp)
    if not parts:
        return dict(error="稳定性验证数据不足。")
    all_imp = pd.concat(parts, ignore_index=True)
    out = all_imp.groupby("Feature").agg(
        Mean_Importance=("Importance", "mean"),
        SD_Importance=("Importance", "std"),
        Mean_Rank=("Rank", "mean"),
        Top10_Frequency=("Rank", lambda x: float((x <= 10).mean())),
        Appear_Count=("Importance", "size"),
    ).reset_index().sort_values("Mean_Importance", ascending=False)
    return out


def ml_final_score(trait_score, stability, w_imp=0.7, w_stab=0.3):
    final = trait_score[["Feature", "Trait_score"]].merge(stability, on="Feature", how="left")
    if len(final) == 0:
        return dict(error="综合评分无可用特征。")
    max_rank = final["Mean_Rank"].max()
    final["RankScore"] = 1 - (final["Mean_Rank"] - 1) / (max_rank - 1) if max_rank > 1 else 1
    rng = final["SD_Importance"].max() - final["SD_Importance"].min()
    final["SD_norm"] = ((final["SD_Importance"] - final["SD_Importance"].min()) / rng
                        if rng > 0 else 0.5)
    final["SDScore"] = 1 - final["SD_norm"]
    final["StabilityScore"] = (0.4 * final["Top10_Frequency"].fillna(0)
                               + 0.4 * final["RankScore"]
                               + 0.2 * final["SDScore"])
    final["FinalScore"] = w_imp * final["Trait_score"] + w_stab * final["StabilityScore"]
    final = final.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    final["Rank"] = final.index + 1
    return final


def ml_report(final_rank):
    def category(f):
        if "Ratio" in f: return "采食节律"
        if "CV" in f or "Robust" in f: return "行为稳定性"
        if "FI" in f or "ADFI" in f: return "采食能力"
        if any(k in f for k in ["Bouts", "Meal", "Duration", "TFD", "IMI", "AFBD"]):
            return "采食模式"
        return "综合行为指标"

    def level(s):
        return "核心指标" if s >= 0.65 else ("重点关注" if s >= 0.45 else "辅助指标")

    def evidence(r):
        if r["Top10_Frequency"] >= 0.8 and r["FinalScore"] >= 0.65:
            return "强证据"
        if r["Top10_Frequency"] >= 0.5:
            return "中等证据"
        return "探索性指标"

    out = final_rank.copy()
    out["Trait_Category"] = out["Feature"].map(category)
    out["Recommendation_Level"] = out["FinalScore"].map(level)
    out["Evidence_Level"] = out.apply(evidence, axis=1)
    interp = {
        "采食能力": "反映个体采食水平和营养摄入能力，与生长性能和饲料利用效率密切相关。",
        "行为稳定性": "反映采食行为波动程度，可评价个体行为一致性和生产稳定性。",
        "采食节律": "反映昼夜采食分配模式，可揭示个体采食时间策略差异。",
        "采食模式": "反映采食事件组织方式、频率和持续特征，可用于描述行为模式差异。",
    }
    out["Biological_Interpretation"] = out["Trait_Category"].map(interp).fillna(
        "综合反映个体采食行为特征，可作为智能育种候选指标。")
    appl = {"核心指标": "建议优先纳入智能育种候选性状体系。",
            "重点关注": "建议结合生产性能和遗传参数进一步验证。",
            "辅助指标": "可作为探索性行为表型进行持续积累。"}
    out["Breeding_Application"] = out["Recommendation_Level"].map(appl)
    return out


def ml_run_all(prod, targets, w_target=None, w_imp=0.7, w_stab=0.3, n_rep=10):
    targets = [t for t in targets if t and t != "---"]
    if not targets:
        return dict(error="请至少选择一个预测目标性状。")
    features = ml_features_v5(prod)
    if len(features) < 5:
        return dict(error="行为特征不足（至少 5 个）。")

    clu = ml_cluster(prod)
    tr = {t: ml_single_target(prod, t, features) for t in targets}
    fusion = ml_fusion(tr, w_target)
    if not isinstance(fusion, pd.DataFrame):
        return dict(error=fusion.get("error"), cluster=clu, targets=tr)
    stab = ml_stability(prod, targets, features, n_rep=n_rep)
    if not isinstance(stab, pd.DataFrame):
        return dict(error=stab.get("error"), cluster=clu, targets=tr,
                    fusion=fusion, stability=stab)
    final = ml_final_score(fusion, stab, w_imp, w_stab)
    if not isinstance(final, pd.DataFrame):
        return dict(error=final.get("error"), cluster=clu, targets=tr,
                    fusion=fusion, stability=stab)
    report = ml_report(final)
    return dict(cluster=clu, targets=tr, fusion=fusion, stability=stab,
                final=final, top10=final.head(10), report=report, features=features)


# ============================================================
# 7. 简单模式
# ============================================================
SIMPLE_GOAL_MAP = {
    "save":   ["FCR", "RFI"],
    "grow":   ["ADG_g", "FBW_kg", "Gain_kg"],
    "fat":    ["SkinFat_Rate", "AbFat_Rate"],
    "meat":   ["BMP"],
    "rhythm": ["CV_Daily_FI", "Cosinor_A", "Fano", "Day_FI_Ratio"],
}


def run_simple_all(feed_files, bw_files, ped_path=None, idmap_path=None,
                   goals=("save",), ratio=0.3, sex_balance=True,
                   auto_time=True, experiment_start=None,
                   day_start="06:00", day_end="18:00"):
    import time
    t0 = time.time()
    target = list(dict.fromkeys([t for g in goals for t in SIMPLE_GOAL_MAP.get(g, [])]))
    use_ped = ped_path is not None

    cfg = dict(
        auto_time_range=auto_time,
        experiment_start=None if auto_time else experiment_start,
        training_end=None, training_time=None,
        week_length=7, feed_sd=3, bw_sd=3, imi_threshold=300, min_intake=1,
        hff_method="median", hff_cutoff=None,
        day_start=day_start, day_end=day_end,
        target_trait=target, indicator_min_r=0.3, indicator_max_n=3,
        use_pedigree=use_ped, use_reml_h2=use_ped,
        fixed_effects=("Sex",), retention_ratio=ratio,
        sex_balance=sex_balance, run_retention=True)

    feed_raw, _ = read_matching_excel_files(feed_files, "feed")
    bw_raw, _   = read_matching_excel_files(bw_files, "weight")
    ped   = read_pedigree(ped_path) if use_ped else None
    idmap = read_idmap(idmap_path)

    clean = run_clean_v5(feed_raw, bw_raw, cfg)
    bout  = build_bouts(clean["feed"], cfg["imi_threshold"], cfg["min_intake"])
    feeding = calc_individual_feeding(bout)
    prod = calc_production(clean["feed"], clean["bw"], feeding)
    prod, _ = assign_hff_lff(prod, cfg["hff_method"], cfg["hff_cutoff"])
    prod["HFF"] = (prod["Feed_Frequency_Group"] == "HFF").astype(int)
    prod = (prod.merge(compute_daynight_v5(bout, day_start, day_end), on="Animal_ID", how="left")
                 .merge(compute_behavior_cv_v5(bout), on="Animal_ID", how="left")
                 .merge(compute_cosinor_v5(bout), on="Animal_ID", how="left")
                 .merge(compute_fano_v5(bout), on="Animal_ID", how="left"))

    rhythm = calc_rhythm(bout)
    weekly_fcr = calc_weekly_fcr(clean["feed"], clean["bw"], cfg["week_length"])
    link = link_animals(clean["feed"], clean["bw"], ped, idmap)
    if "eID" in link.columns and "Sex" in link.columns:
        prod = prod.merge(link[["eID", "Sex"]].drop_duplicates("eID"),
                          left_on="Animal_ID", right_on="eID", how="left")

    MAX_TRAITS = 8
    meas = [t for t in target if t in prod.columns][:MAX_TRAITS]
    ind_slots = MAX_TRAITS - len(meas)
    ind_traits = (select_indicator_traits(prod, target, 0.3, 3)
                  if ind_slots > 0 else pd.DataFrame(columns=["Indicator"]))
    ind_traits = ind_traits.head(ind_slots)
    idx_traits = list(dict.fromkeys(meas + ind_traits["Indicator"].tolist()))
    n_meas, n_ind = len(meas), len(idx_traits) - len(meas)
    w = {}
    if n_ind > 0 and n_meas > 0:
        for t in meas: w[t] = 0.7 / n_meas
        for t in idx_traits:
            if t not in meas: w[t] = 0.3 / n_ind
    else:
        for t in idx_traits: w[t] = 1.0 / len(idx_traits)

    cfg["index_traits"] = idx_traits
    cfg["weights"] = w
    rr = run_retention_module(prod, link, cfg)

    growth = clean["bw_week_stats"]
    growth_plot = pd.DataFrame({
        "Week": growth["Week_Number"] if "Week_Number" in growth.columns else growth.iloc[:, 0],
        "Mean_BW": growth["Mean"] if "Mean" in growth.columns else np.nan})

    A = dict(
        summary=pd.DataFrame({
            "指标": ["评估路线", "参与打分性状数", "参与打分性状", "总个体",
                     "留种个体", "留种比例", "用时(秒)"],
            "值": [rr["genetic"]["mode"] if rr["genetic"] else "-",
                   str(len(idx_traits)), "、".join(idx_traits),
                   str(rr["retention"]["total"]) if rr["retention"] else "-",
                   str(rr["retention"]["n_keep"]) if rr["retention"] else "-",
                   f"{int(rr['retention']['ratio']*100)}%" if rr["retention"] else "-",
                   f"{time.time()-t0:.1f}"]}),
        retained=rr["retention"]["retained"] if rr["retention"] else None,
        ratio=rr["retention"]["ratio"] if rr["retention"] else ratio,
        all_index=rr["genetic"]["index"] if rr["genetic"] else None,
        indicator_traits=rr["indicator_traits"],
        warnings=rr["warnings"])

    B = dict(week_feed=clean["feed_week_stats"], week_bw=clean["bw_week_stats"],
             weekly_fcr=weekly_fcr, growth=growth_plot)

    daynight_cols = [c for c in ["Animal_ID", "Day_FI_g", "Night_FI_g", "Day_Bouts",
                                  "Night_Bouts", "Day_FI_Ratio", "Day_Bout_Ratio"]
                     if c in prod.columns]
    cv_cols = [c for c in ["Animal_ID", "CV_Duration", "CV_FR", "CV_Daily_Bouts",
                            "CV_Daily_FI", "CV_Daily_TFD"] if c in prod.columns]
    C = dict(rhythm_daily=rhythm["daily"], rhythm_hourly=rhythm["hourly"],
             daynight=prod[daynight_cols], cv=prod[cv_cols],
             anomalies=flag_anomalies(prod, A["all_index"]))

    return dict(A=A, B=B, C=C, prod=prod, elapsed=time.time() - t0)


def flag_anomalies(prod, index_df=None):
    out = []
    if "Day_Bout_Ratio" in prod.columns and "TFB_Day" in prod.columns:
        thr = prod["TFB_Day"].quantile(0.05)
        d = prod[(prod["Day_Bout_Ratio"] > 0.97) | (prod["TFB_Day"] < thr)][["Animal_ID"]].copy()
        d["异常类型"] = "采食节律异常"
        d["具体表现"] = np.where(prod.loc[d.index, "Day_Bout_Ratio"] > 0.97,
                                "夜间几乎不采食（白天占比>97%）",
                                "日访饲次数过低（群体最低5%）")
        if len(d): out.append(d)
    if "Gain_kg" in prod.columns:
        thr = prod["Gain_kg"].quantile(0.05)
        d = prod[prod["Gain_kg"] < thr][["Animal_ID"]].copy()
        d["异常类型"] = "体重增长异常"
        d["具体表现"] = "总增重过低（群体最低5%）"
        if len(d): out.append(d)
    if index_df is not None and "Index" in index_df.columns:
        thr = index_df["Index"].quantile(0.05)
        d = index_df[index_df["Index"] < thr][["Animal_ID"]].copy()
        d["异常类型"] = "综合得分极低"
        d["具体表现"] = "综合得分处于群体最低5%"
        if len(d): out.append(d)
    if not out:
        return pd.DataFrame(columns=["Animal_ID", "异常类型", "具体表现"])
    return pd.concat(out, ignore_index=True).drop_duplicates().reset_index(drop=True)