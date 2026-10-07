# -*- coding: utf-8 -*-
"""鸭芯智选 · 肉鸭智能化采食行为分析工具 V5.0 —— Streamlit 界面"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import duck_core as dc

# 中文字体（Linux 服务器若无中文字体，可换成 'DejaVu Sans' 或安装思源黑体）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="鸭芯智选 V5.0", layout="wide",
                   initial_sidebar_state="expanded")

PAL = dc.PAL_RPBG


# ============================================================
# 通用工具
# ============================================================
def df_to_excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            if df is not None and len(df) > 0:
                df.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


def show_df(df, height=400, key=None):
    if df is None or len(df) == 0:
        st.info("暂无数据")
        return
    st.dataframe(df, use_container_width=True, height=height, key=key)


def dl_button(df, filename, label="下载 Excel"):
    if df is None or len(df) == 0:
        return
    st.download_button(label, df_to_excel_bytes({"Sheet1": df}), filename,
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def mpl_to_st(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# 侧边栏：模式切换
# ============================================================
st.sidebar.title("🦆 鸭芯智选 V5.0")
mode = st.sidebar.radio("选择模式", ["🦆 简单模式", "🔬 专家模式"], index=0)


# ============================================================
# 简单模式
# ============================================================
if mode == "🦆 简单模式":
    st.title("🦆 简单模式：一键选留种鸭")
    st.caption("上传数据 → 选育种目标 → 一键出留种名单（含生产报表 / 采食行为）")

    with st.sidebar:
        st.header("① 上传数据")
        simp_feed = st.file_uploader("采食表（必传，可多选）", type=["xls", "xlsx"],
                                     accept_multiple_files=True, key="simp_feed")
        simp_bw = st.file_uploader("体重表（必传，可多选）", type=["xls", "xlsx"],
                                   accept_multiple_files=True, key="simp_bw")
        simp_ped = st.file_uploader("系谱（可选，有系谱更准）", type=["xls", "xlsx"], key="simp_ped")
        simp_idmap = st.file_uploader("ID对照表（可选）", type=["xls", "xlsx"], key="simp_idmap")

        st.header("时间范围")
        simp_auto_time = st.checkbox("自动使用数据全部时间（默认）", value=True)
        simp_start = None
        if not simp_auto_time:
            simp_start = st.date_input("试验开始日期", pd.Timestamp.today() - pd.Timedelta(days=30))

        st.header("② 育种目标（可多选）")
        goal_save = st.checkbox("吃得省（料重比低、省饲料）", value=True)
        goal_grow = st.checkbox("长得快（日增重高、出栏重）", value=True)
        goal_fat  = st.checkbox("皮脂厚（烤鸭用，未测自动用文献关联指标）", value=False)
        goal_meat = st.checkbox("胸肌厚（瘦肉多，未测自动用文献关联指标）", value=False)
        goal_rhythm = st.checkbox("采食规律（节律稳定）", value=False)

        st.header("③ 留种设置")
        simp_ratio = st.slider("留种比例 (%)", 5, 80, 30, step=5)
        simp_sex_balance = st.checkbox("尽量公母均衡", value=True)
        run_simple = st.button("开始选留种鸭", type="primary", use_container_width=True)

        if simp_ped is None:
            st.info("未上传系谱：将自动采用权威文献遗传参数（不影响使用）")
        else:
            st.success("已上传系谱：将用本场数据实测遗传力，评估更准")

    if run_simple:
        if not simp_feed or not simp_bw:
            st.error("请至少上传采食表和体重表各一份。")
        else:
            goals = []
            if goal_save: goals.append("save")
            if goal_grow: goals.append("grow")
            if goal_fat: goals.append("fat")
            if goal_meat: goals.append("meat")
            if goal_rhythm: goals.append("rhythm")
            if not goals: goals = ["save"]

            with st.spinner("简单模式计算中（约 1-2 分钟）…"):
                try:
                    res = dc.run_simple_all(
                        simp_feed, simp_bw,
                        ped_path=simp_ped, idmap_path=simp_idmap,
                        goals=goals, ratio=simp_ratio / 100.0,
                        sex_balance=simp_sex_balance,
                        auto_time=simp_auto_time,
                        experiment_start=simp_start)
                    st.session_state["simple_res"] = res
                except Exception as e:
                    st.error(f"运行失败：{e}")
                    st.session_state.pop("simple_res", None)

    res = st.session_state.get("simple_res")
    if res:
        A, B, C = res["A"], res["B"], res["C"]
        for w in A["warnings"]:
            st.warning(w)
        if C["anomalies"] is not None and len(C["anomalies"]) > 0:
            st.warning(f"⚠ 检出 {len(C['anomalies'])} 条异常记录，见『采食行为』页（暂缓复测，不淘汰）")

        tabA, tabB, tabC = st.tabs(["🦆 选留种鸭", "📊 生产报表", "🔍 采食行为"])

        with tabA:
            st.subheader("结果摘要")
            st.table(A["summary"])
            st.subheader(f"留种名单（前 {int(A['ratio']*100)}%）")
            keep = A["retained"]
            if keep is not None:
                show_df(keep.head(200))
            st.subheader("综合得分分布")
            if A["all_index"] is not None:
                idx = A["all_index"].copy()
                keep_ids = set(keep["Animal_ID"]) if keep is not None else set()
                idx["Is_Retained"] = idx["Animal_ID"].isin(keep_ids)
                fig = px.histogram(idx, x="Index", color="Is_Retained", nbins=40,
                                   color_discrete_map={True: PAL[0], False: PAL[1]},
                                   labels={"Is_Retained": "是否留种", "Index": "综合得分"})
                st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "导出全部结果（Excel）",
                df_to_excel_bytes({
                    "A_留种名单": keep,
                    "A_全部个体得分": A["all_index"],
                    "A_指示性状": A["indicator_traits"],
                    "B_周统计_采食": B["week_feed"],
                    "B_周统计_体重": B["week_bw"],
                    "B_每周FCR": B["weekly_fcr"],
                    "C_昼夜分配": C["daynight"],
                    "C_行为变异性": C["cv"],
                    "C_异常个体": C["anomalies"],
                }),
                f"简单模式_结果_{pd.Timestamp.today().date()}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with tabB:
            st.subheader("周统计 · 采食"); show_df(B["week_feed"])
            st.subheader("周统计 · 体重"); show_df(B["week_bw"])
            st.subheader("每周 FCR")
            if B["weekly_fcr"] is not None and len(B["weekly_fcr"]) > 0:
                agg = B["weekly_fcr"].groupby("Week_Number")["Weekly_FCR"].mean().reset_index()
                fig = px.bar(agg, x="Week_Number", y="Weekly_FCR")
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("周平均体重（生长曲线）")
            if B["growth"] is not None and len(B["growth"]) > 0:
                fig = px.line(B["growth"], x="Week", y="Mean_BW", markers=True)
                st.plotly_chart(fig, use_container_width=True)

        with tabC:
            st.subheader("日访饲节律")
            fig = px.line(C["rhythm_daily"], x="Date", y="Mean_Bouts_Per_Duck", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("24 小时访饲节律（按周）")
            fig = px.line(C["rhythm_hourly"], x="Hour_Block", y="Mean_Bouts_Per_Duck",
                          color="Week_Number", markers=True)
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("昼夜分配"); show_df(C["daynight"])
            st.subheader("行为变异性"); show_df(C["cv"])
            st.subheader("异常个体预警（暂缓复测，不淘汰）"); show_df(C["anomalies"])


# ============================================================
# 专家模式
# ============================================================
else:
    st.title("🔬 专家模式")
    mods = st.tabs(["① 数据与清洗", "② 个体指标", "③ 组间时序相关",
                    "④ 创新行为指标", "⑤ 遗传评估与留种", "⑥ AI 智能选种探索"])

    # ---------- ① 数据与清洗 ----------
    with mods[0]:
        with st.sidebar:
            st.header("数据上传")
            feed_files = st.file_uploader("采食原始数据（可多选，自动读全部 Sheet）",
                                          type=["xls", "xlsx"],
                                          accept_multiple_files=True, key="exp_feed")
            bw_files = st.file_uploader("体重原始数据（可多选，自动读全部 Sheet）",
                                        type=["xls", "xlsx"],
                                        accept_multiple_files=True, key="exp_bw")
            ped_file = st.file_uploader("系谱（可选：翅号/父本笼号/母本笼号/性别）",
                                        type=["xls", "xlsx"], key="exp_ped")
            idmap_file = st.file_uploader("eID-ID 对照表（可选）",
                                          type=["xls", "xlsx"], key="exp_idmap")

            st.header("清洗参数")
            auto_time = st.checkbox("自动读取数据最早/最晚时间", value=True, key="exp_auto_time")
            exp_start = None
            if not auto_time:
                exp_start = st.date_input("试验开始日期",
                                          pd.Timestamp.today() - pd.Timedelta(days=30),
                                          key="exp_start")
            week_length = st.number_input("周长度（天）", 1, 30, 7)
            feed_sd = st.number_input("采食异常阈值（均值±k倍SD）", 1.0, 10.0, 3.0)
            bw_sd = st.number_input("体重异常阈值（均值±k倍SD）", 1.0, 10.0, 3.0)
            imi_threshold = st.number_input("餐间隔阈值（秒）", 30, 3600, 300)
            min_intake = st.number_input("最小单次采食量（g）", 0.0, 100.0, 1.0)

            run_clean = st.button("运行数据清洗", type="primary", use_container_width=True)

        if run_clean:
            if not feed_files or not bw_files:
                st.error("请上传采食和体重数据。")
            else:
                cfg = dict(auto_time_range=auto_time,
                           experiment_start=None if auto_time else exp_start,
                           training_end=None, training_time=None,
                           week_length=int(week_length),
                           feed_sd=float(feed_sd), bw_sd=float(bw_sd),
                           imi_threshold=float(imi_threshold),
                           min_intake=float(min_intake),
                           target_trait=["FCR"], indicator_min_r=0.3, indicator_max_n=5,
                           use_pedigree=ped_file is not None, h2=0.3,
                           fixed_effects=("Sex",),
                           index_traits=["FCR", "ADG_g", "FBW_kg"],
                           weights={"FCR": 0.5, "ADG_g": 0.3, "FBW_kg": 0.2},
                           retention_ratio=0.3, sex_balance=True,
                           hff_method="median", hff_cutoff=None,
                           run_retention=False,
                           day_start="06:00", day_end="18:00")
                with st.spinner("数据读取与清洗中…"):
                    try:
                        rv = dc.run_pipeline_v5(feed_files, bw_files, ped_file, idmap_file, cfg)
                        st.session_state["pipe"] = rv
                    except Exception as e:
                        st.error(f"清洗失败：{e}")

        rv = st.session_state.get("pipe")
        if rv:
            st.success(f"清洗完成：采食 {len(rv['clean']['feed'])} 行，"
                       f"体重 {len(rv['clean']['bw'])} 行，耗时 {rv['elapsed']:.1f} 秒")
            st.subheader("工作表读取情况")
            show_df(pd.concat([rv["feed_raw_qc"], rv["bw_raw_qc"]], ignore_index=True))

            t1, t2 = st.tabs(["时间范围 QC", "字段缺失 QC"])
            with t1:
                dq = rv["clean"]["datetime_qc"]
                st.info(f"数据时间范围：{dq['Start'].iloc[0]} ～ {dq['End'].iloc[0]}，"
                        f"共 {dq['N'].iloc[0]} 条采食记录。")
                st.dataframe(dq.astype(str))
            with t2:
                show_df(rv["clean"]["na_qc"])

            st.subheader("周统计（采食）"); show_df(rv["clean"]["feed_week_stats"])
            st.subheader("周统计（体重）"); show_df(rv["clean"]["bw_week_stats"])

    # ---------- ② 个体指标 ----------
    with mods[1]:
        rv = st.session_state.get("pipe")
        if not rv:
            st.info("请先在①运行数据清洗。")
        else:
            st.subheader("采食行为")
            show_df(rv["feeding"])
            st.subheader("生产性能")
            cols = [c for c in ["Animal_ID", "Feed_Frequency_Group", "TFB", "TFB_Day",
                                "FI_g", "FI_Day_g", "AMS_g", "TFD_sec", "AFBD_sec",
                                "IMI_sec", "FR_g_sec", "IBW_kg", "FBW_kg",
                                "Gain_kg", "ADG_g", "ADFI_g", "FCR", "RFI",
                                "Feed_Efficiency"] if c in rv["prod"].columns]
            show_df(rv["prod"][cols])

            st.subheader("单次采食指标（按周）")
            wk = (rv["clean"]["feed"][["Animal_ID", "Time1", "Week_Number"]]
                    .assign(Date=lambda d: d["Time1"].dt.date)
                    .drop_duplicates(["Animal_ID", "Date"]))
            bout_wk = rv["bout"].merge(wk[["Animal_ID", "Date", "Week_Number"]],
                                       on=["Animal_ID", "Date"], how="left")
            for yvar, ylab in [("FR_g_sec", "采食速率 (g/s)"),
                               ("IMI_sec", "采食间隔 (s)"),
                               ("Feed_Duration_sec", "采食时长 (s)")]:
                if yvar in bout_wk.columns:
                    fig = px.box(bout_wk.dropna(subset=[yvar, "Week_Number"]),
                                 x="Week_Number", y=yvar, color="Week_Number",
                                 labels={"Week_Number": "Week", yvar: ylab})
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("总体生长曲线")
            bwc = rv["clean"]["bw"].dropna(subset=["BW_kg"]).copy()
            bwc["Date"] = bwc["Time1"].dt.date
            daily = (bwc.sort_values(["Animal_ID", "Time1"])
                       .groupby(["Animal_ID", "Date"]).last().reset_index())
            daily["Exp_Day"] = (pd.to_datetime(daily["Date"]) -
                                pd.to_datetime(daily["Date"]).min()).dt.days
            agg = daily.groupby("Exp_Day")["BW_kg"].mean().reset_index()
            fig = px.line(agg, x="Exp_Day", y="BW_kg", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- ③ 组间时序相关 ----------
    with mods[2]:
        rv = st.session_state.get("pipe")
        if not rv:
            st.info("请先在①运行数据清洗。")
        else:
            hff_method = st.selectbox("HFF/LFF 分组方法", ["median", "custom"],
                                      format_func=lambda x: "中位数" if x == "median" else "自定义阈值")
            hff_cutoff = None
            if hff_method == "custom":
                hff_cutoff = st.number_input("自定义阈值（次/天）", 1.0, 500.0, 30.0)
            if st.button("运行组间分析", type="primary"):
                prod, cutoff = dc.assign_hff_lff(rv["prod"], hff_method, hff_cutoff)
                rv["prod"] = prod
                st.session_state["pipe"] = rv
                st.success(f"HFF/LFF 分组阈值：{cutoff:.1f} 次/天")

            if "Feed_Frequency_Group" in rv["prod"].columns:
                st.subheader("HFF/LFF 指标对比")
                cmp = rv["prod"].groupby("Feed_Frequency_Group").agg(
                    N=("Animal_ID", "size"),
                    TFB_Day=("TFB_Day", "mean"),
                    FI_Day_g=("FI_Day_g", "mean"),
                    ADG_g=("ADG_g", "mean"),
                    FCR=("FCR", "mean"),
                    RFI=("RFI", "mean")).round(2).reset_index()
                st.dataframe(cmp)
                st.subheader("访饲节律（日）")
                st.plotly_chart(px.line(rv["rhythm"]["daily"], x="Date",
                                        y="Mean_Bouts_Per_Duck", markers=True),
                                use_container_width=True)
                st.subheader("访饲节律（24h）")
                st.plotly_chart(px.line(rv["rhythm"]["hourly"], x="Hour_Block",
                                        y="Mean_Bouts_Per_Duck",
                                        color="Week_Number", markers=True),
                                use_container_width=True)
                st.subheader("每周 FCR")
                show_df(rv["weekly_fcr"])

    # ---------- ④ 创新行为指标 ----------
    with mods[3]:
        rv = st.session_state.get("pipe")
        if not rv:
            st.info("请先在①运行数据清洗。")
        else:
            c1, c2 = st.columns(2)
            day_start = c1.text_input("白天开始(HH:MM)", "06:00")
            day_end = c2.text_input("白天结束(HH:MM)", "18:00")
            if st.button("计算创新指标", type="primary"):
                bout = rv["bout"]
                prod = (rv["prod"]
                        .merge(dc.compute_daynight_v5(bout, day_start, day_end),
                               on="Animal_ID", how="left")
                        .merge(dc.compute_behavior_cv_v5(bout), on="Animal_ID", how="left")
                        .merge(dc.compute_cosinor_v5(bout), on="Animal_ID", how="left")
                        .merge(dc.compute_fano_v5(bout), on="Animal_ID", how="left"))
                rv["prod"] = prod
                st.session_state["pipe"] = rv
                st.success("创新指标计算完成")

            prod = rv["prod"]
            t1, t2, t3, t4, t5 = st.tabs(["昼夜分配", "行为变异性 CV",
                                          "余弦节律拟合", "Fano 指数", "创新指标×FCR"])
            with t1:
                cols = [c for c in ["Animal_ID", "Day_FI_g", "Night_FI_g",
                                    "Day_Bouts", "Night_Bouts",
                                    "Day_FI_Ratio", "Day_Bout_Ratio"] if c in prod.columns]
                show_df(prod[cols])
                if "Day_FI_Ratio" in prod.columns and "FCR" in prod.columns:
                    d = prod.dropna(subset=["Day_FI_Ratio", "FCR"])
                    fig = px.scatter(d, x="Day_FI_Ratio", y="FCR", trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
            with t2:
                cols = [c for c in ["Animal_ID", "CV_Duration", "CV_FR", "Robust_CV_IMI",
                                    "CV_Daily_Bouts", "CV_Daily_FI", "CV_Daily_TFD"]
                        if c in prod.columns]
                show_df(prod[cols])
            with t3:
                cols = [c for c in ["Animal_ID", "Cosinor_M", "Cosinor_A",
                                    "Peak_Hour", "Cosinor_R2"] if c in prod.columns]
                show_df(prod[cols])
                if "Peak_Hour" in prod.columns:
                    st.plotly_chart(px.histogram(prod, x="Peak_Hour", nbins=24),
                                    use_container_width=True)
            with t4:
                if "Fano" in prod.columns:
                    show_df(prod[["Animal_ID", "Fano"]])
            with t5:
                innov = [c for c in ["Day_FI_Ratio", "Day_Bout_Ratio", "CV_Duration",
                                     "CV_FR", "CV_Daily_Bouts", "CV_Daily_FI",
                                     "Cosinor_A", "Fano"] if c in prod.columns]
                rows = []
                for v in innov:
                    tmp = prod[[v, "FCR"]].dropna() if "FCR" in prod.columns else pd.DataFrame()
                    if len(tmp) >= 5:
                        r, p = dc.stats.spearmanr(tmp[v], tmp["FCR"])
                        rows.append(dict(创新指标=v, Spearman_r=round(r, 3),
                                         P_value=dc.format_p(p), N=len(tmp)))
                show_df(pd.DataFrame(rows))

    # ---------- ⑤ 遗传评估与留种 ----------
    with mods[4]:
        rv = st.session_state.get("pipe")
        if not rv:
            st.info("请先在①运行数据清洗。")
        else:
            with st.sidebar:
                st.header("① 育种目标")
                target_traits = st.multiselect(
                    "目标性状（可多选）",
                    ["FCR", "RFI", "ADG_g", "FBW_kg", "FI_Day_g", "FR_g_sec",
                     "TFB_Day", "AMS_g", "ADFI_g", "IBW_kg", "Gain_kg",
                     "Day_FI_Ratio", "SkinFat_Rate", "AbFat_Rate", "BMP"],
                    default=["FCR"])
                indicator_min_r = st.number_input("指示性状阈值(|r|≥)", 0.0, 1.0, 0.3)
                indicator_max_n = st.number_input("指示性状上限", 1, 10, 5)

                st.header("② 遗传评估路线")
                use_ped = st.radio("评估路线", [True, False],
                                   format_func=lambda x: "路线A：PBLUP（有系谱）" if x
                                   else "路线B：文献遗传参数（无系谱）",
                                   index=0 if ped_file else 1)
                use_reml = st.checkbox("路线A运行时自动实测本场 h²", value=True)

                st.header("③ 选择指数性状与权重")
                idx_traits = st.multiselect(
                    "参与指数的性状",
                    dc.INDEX_TRAIT_VALUES,
                    default=[t for t in ["FCR", "ADG_g", "FBW_kg"]
                             if t in rv["prod"].columns])
                weights = {}
                for t in idx_traits:
                    weights[t] = st.number_input(f"{t} 权重", 0.0, 1.0, 0.2,
                                                 step=0.05, key=f"w_{t}")

                st.header("④ 留种设置")
                retention_ratio = st.slider("留种比例 (%)", 1, 90, 30) / 100.0
                sex_balance = st.checkbox("留种时尽量性别均衡", value=True)

                run_gen = st.button("运行遗传评估与留种", type="primary",
                                    use_container_width=True)

            if run_gen:
                cfg = dict(target_trait=target_traits or ["FCR"],
                           indicator_min_r=float(indicator_min_r),
                           indicator_max_n=int(indicator_max_n),
                           use_pedigree=use_ped,
                           use_reml_h2=use_reml,
                           index_traits=idx_traits,
                           weights=weights or None,
                           h2_by_trait=None,
                           fixed_effects=("Sex",),
                           retention_ratio=retention_ratio,
                           sex_balance=sex_balance)
                with st.spinner("遗传评估与留种计算中…"):
                    try:
                        rr = dc.run_retention_module(rv["prod"], rv["link"], cfg)
                        rv["indicator_traits"] = rr["indicator_traits"]
                        rv["genetic"] = rr["genetic"]
                        rv["retention"] = rr["retention"]
                        rv["key_metrics"] = rr["key_metrics"]
                        st.session_state["pipe"] = rv
                        for w in rr["warnings"]:
                            st.warning(w)
                        if rr.get("error"):
                            st.error(rr["error"])
                    except Exception as e:
                        st.error(f"遗传评估失败：{e}")

            if rv.get("key_metrics") is not None:
                st.subheader("关键指标卡")
                st.table(rv["key_metrics"])
                g = rv.get("genetic")
                if g:
                    t1, t2, t3, t4, t5 = st.tabs(
                        ["① 指示性状筛选", "② 遗传力估计", "③ 育种值 EBV",
                         "④ 选择指数分布", "⑤ 留种名单"])
                    with t1:
                        show_df(rv["indicator_traits"])
                    with t2:
                        if g.get("h2_est") is not None:
                            show_df(g["h2_est"])
                        else:
                            st.info("本次未做本场实测遗传力，使用文献/输入 h²。")
                    with t3:
                        show_df(g.get("ebv_tbl"))
                    with t4:
                        idx = g.get("index")
                        if idx is not None:
                            keep_ids = set(rv["retention"]["retained"]["Animal_ID"])
                            idx = idx.copy()
                            idx["Is_Retained"] = idx["Animal_ID"].isin(keep_ids)
                            fig = px.histogram(idx, x="Index", color="Is_Retained",
                                               nbins=40,
                                               color_discrete_map={True: PAL[0],
                                                                   False: PAL[1]})
                            st.plotly_chart(fig, use_container_width=True)
                    with t5:
                        show_df(rv["retention"]["retained"])

    # ---------- ⑥ AI 智能选种探索 ----------
    with mods[5]:
        rv = st.session_state.get("pipe")
        if not rv:
            st.info("请先在①运行数据清洗。")
        else:
            with st.sidebar:
                st.header("AI 智能选种探索")
                ml_targets = st.multiselect("预测目标性状",
                                            ["ADG_g", "FCR", "RFI"],
                                            default=["ADG_g", "FCR", "RFI"])
                st.subheader("目标权重（Trait_score 融合）")
                w_adg = st.number_input("ADG 权重", 0.0, 1.0, 0.33, step=0.05)
                w_fcr = st.number_input("FCR 权重", 0.0, 1.0, 0.33, step=0.05)
                w_rfi = st.number_input("RFI 权重", 0.0, 1.0, 0.34, step=0.05)
                st.subheader("最终评分权重")
                w_imp = st.number_input("重要性得分权重", 0.0, 1.0, 0.7, step=0.05)
                w_stab = st.number_input("稳定性得分权重", 0.0, 1.0, 0.3, step=0.05)
                run_ml = st.button("运行 AI 智能选种分析", type="primary",
                                   use_container_width=True)

            if run_ml:
                w_target = []
                for t in ml_targets:
                    w_target.append({"ADG_g": w_adg, "FCR": w_fcr, "RFI": w_rfi}[t])
                with st.spinner("AI 智能选种分析中…"):
                    ml_res = dc.ml_run_all(rv["prod"], ml_targets,
                                           w_target=w_target,
                                           w_imp=w_imp, w_stab=w_stab)
                    st.session_state["ml"] = ml_res
                if ml_res.get("error"):
                    st.error(ml_res["error"])

            ml = st.session_state.get("ml")
            if ml and not ml.get("error"):
                t1, t2, t3, t4, t5 = st.tabs(
                    ["采食模式聚类", "特征重要性", "综合评分 Top10",
                     "稳定性验证", "育种报告"])
                with t1:
                    show_df(ml["cluster"]["result"])
                    show_df(ml["cluster"]["compare"])
                with t2:
                    show_df(ml["fusion"])
                    evals = [v["evaluation"] for v in ml["targets"].values()
                             if v and v.get("evaluation") is not None]
                    if evals:
                        show_df(pd.concat(evals, ignore_index=True))
                with t3:
                    show_df(ml["final"].head(10))
                with t4:
                    show_df(ml["stability"])
                with t5:
                    show_df(ml["report"])
                    st.download_button(
                        "下载育种报告 Excel",
                        df_to_excel_bytes({
                            "Final_Ranking": ml["report"],
                            "Top10": ml["top10"],
                            "Trait_Importance": ml["fusion"],
                            "Stability": ml["stability"],
                            "Cluster_Result": ml["cluster"]["result"],
                            "Cluster_Summary": ml["cluster"]["summary"],
                        }),
                        f"duck_ai_breeding_report_{pd.Timestamp.today().date()}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")