"""IESO 2025 Ontario Demand 日前负荷预测：归档数据离线回放。"""

import argparse
import csv
import hashlib
import io
import json
from collections import deque
from dataclasses import dataclass
from datetime import time, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://reports-public.ieso.ca/public/Demand/PUB_Demand_2025.csv"
EST = timezone(timedelta(hours=-5), "EST")  # 报表全年固定 EST，不随夏令时切换
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
HOURS = 24
BIAS_BATCHES = 7
METHODS = {
    "d2_same_hour": "D-2 same hour",
    "d7_same_hour": "D-7 same hour",
    "ma24": "MA24",
    "stl": "STL + trend forecast",
    "stl_ma_bias": "STL + MA bias correction",
}


@dataclass(frozen=True)
class Schedule:
    issue_offset: pd.Timedelta
    delay: pd.Timedelta

    def issue(self, target_day):
        return target_day - DAY + self.issue_offset


@dataclass(frozen=True)
class Parameters:
    history_days: int
    trend_rule: str


def read_load(source):
    """读取原始 IESO 报表。URL 和本地文件共用同一个解析入口。"""
    if source.startswith("https://"):
        with urlopen(source, timeout=30) as response:
            content = response.read()
    else:
        content = Path(source).read_bytes()
    decoded = content.decode("utf-8-sig")
    columns = {"Date", "Hour", "Ontario Demand"}
    header = next(
        (i for i, row in enumerate(csv.reader(io.StringIO(decoded))) if columns.issubset(row)),
        None,
    )
    if header is None:
        raise ValueError("没有找到 IESO 表头 Date、Hour、Ontario Demand，请核对文件来源")
    raw = pd.read_csv(io.StringIO(decoded), skiprows=header)
    report_day = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="raise")
    he = pd.to_numeric(raw["Hour"], errors="raise")
    if not (he.isin(range(1, 25)).all() and report_day.notna().all()):
        raise ValueError("原始日期不能为空，Hour 必须是 HE01 至 HE24")
    if not (report_day.dt.year == 2025).all():
        raise ValueError("本案例回测日历固定为 2025 年，请使用对应年度报表")
    frame = raw.copy()  # 原始日期、原始 HE 和两列负荷均保留
    frame["report_day"] = report_day.dt.date.astype(str)
    frame["target_end"] = report_day.dt.tz_localize(EST) + pd.to_timedelta(he, unit="h")
    frame["load_mw"] = pd.to_numeric(raw["Ontario Demand"], errors="coerce")
    frame["load_mw"] = frame["load_mw"].replace([np.inf, -np.inf], np.nan)
    duplicates = int(frame.duplicated("target_end").sum())
    unique = frame.drop_duplicates(subset=list(raw.columns))
    if unique["target_end"].duplicated().any():
        raise ValueError("同一时段有不同的报表记录，需先核对版本，无法自动选值")
    annual_grid = pd.date_range("2025-01-01 01:00", "2026-01-01 00:00", freq="h", tz=EST)
    observations = unique.set_index("target_end")["load_mw"].sort_index()
    load = observations.reindex(annual_grid)
    load.index.name = "target_end"

    # 用开发期的小时变化分布定义筛查阈值。标记跳变，保留观测原值。
    change = load.diff()
    dev = change.loc[:pd.Timestamp("2025-09-01", tz=EST)].abs().dropna()
    center = float(dev.median())
    mad = float((dev - center).abs().median())
    threshold = center + 6 * 1.4826 * mad
    flagged = change.abs() > threshold
    ramp_flags = pd.DataFrame({"load_mw": load[flagged], "change_mw": change[flagged]})
    stats = {
        "source": source,
        "official_url": SOURCE_URL,
        "sha256": hashlib.sha256(content).hexdigest(),
        "raw_columns": list(raw.columns),
        "first_rows": raw.head(5).to_dict(orient="records"),
        "rows": len(raw),
        "expected_hours": len(annual_grid),
        "first_report_date": str(raw["Date"].min()),
        "last_report_date": str(raw["Date"].max()),
        "first_end": str(observations.index.min()),
        "last_end": str(observations.index.max()),
        "duplicate_hour_rows": duplicates,
        "source_null_load_rows": int(frame["load_mw"].isna().sum()),
        "interval_gaps": int((observations.index.to_series().diff().dropna() != HOUR).sum()),
        "missing_grid_hours": [str(t) for t in annual_grid.difference(observations.index)],
        "missing_or_invalid_load_hours": int(load.isna().sum()),
        "nonpositive_hours": int((load <= 0).sum()),
        "min_mw": float(load.min()),
        "max_mw": float(load.max()),
        "ramp_flag_threshold_mw": threshold,
        "flagged_ramps": int(flagged.sum()),
    }
    return frame, load, ramp_flags, stats


def forecast_day(load, target_day, schedule, params):
    """只用发报时已经可用的历史，计算三个基线和 STL 基础预测。"""
    issue_time = schedule.issue(target_day)
    eligible = load.loc[: issue_time - schedule.delay]
    observed = eligible.dropna()
    if len(observed) < HOURS:
        raise ValueError(f"{target_day.date()} 发报前的有效小时数不足 24")
    last_observed_end = observed.index[-1]
    available_history = eligible.loc[:last_observed_end]
    # 仅在已截断的历史内补值。优先参考上周同小时，其次沿用更早已知值。
    filled = available_history.fillna(available_history.shift(7 * HOURS)).ffill()
    stl_history = filled.tail(params.history_days * HOURS)
    if len(stl_history) != params.history_days * HOURS or stl_history.isna().any():
        raise ValueError(f"{target_day.date()} 没有足够连续历史用于 STL")

    target_ends = pd.date_range(target_day + HOUR, periods=HOURS, freq="h")
    lead_hours = np.asarray((target_ends - last_observed_end) / HOUR, dtype=int)
    assert last_observed_end + schedule.delay <= issue_time and lead_hours[0] > 0
    fit = STL(stl_history, period=HOURS, seasonal=13, trend=49, low_pass=25, robust=True).fit()
    # 从最新观测往后推到目标日末，再取目标日 24 个点。
    steps = np.arange(1, lead_hours[-1] + 1)
    if params.trend_rule == "flat24":
        projected_trend = np.full(len(steps), fit.trend.iloc[-24:].mean())
    else:
        slope, level = np.polyfit(np.arange(-71, 1), fit.trend.iloc[-72:], deg=1)
        projected_trend = level + slope * steps
    seasonal_cycle = fit.seasonal.iloc[-HOURS:].to_numpy()
    projected_season = seasonal_cycle[(steps - 1) % HOURS]
    trend_target = projected_trend[lead_hours - 1]
    season_target = projected_season[lead_hours - 1]

    frame = pd.DataFrame({
        "target_date": target_day.date().isoformat(),
        "hour_ending": np.arange(1, 25),
        "issue_time": issue_time,
        "last_observed_end": last_observed_end,
        "last_observed_available_at": last_observed_end + schedule.delay,
        "target_end": target_ends,
        "lead_hours_from_last_observation": lead_hours,
        "d2_same_hour": available_history.reindex(target_ends - 2 * DAY).to_numpy(),
        "d7_same_hour": available_history.reindex(target_ends - 7 * DAY).to_numpy(),
        "ma24": observed.iloc[-HOURS:].mean(),
        "stl": trend_target + season_target,
        "forecast_trend_mw": trend_target,
        "forecast_season_mw": season_target,
    })
    return frame, fit


def choose_parameters(load, schedule, validation_days, test_start):
    """按验证期 STL 基础 MAE 选配置。只有已到齐的验证批次参加选择。"""
    first_issue = schedule.issue(test_start)
    eligible_days = [day for day in validation_days if day + DAY + schedule.delay <= first_issue]
    if not eligible_days:
        raise ValueError("首个测试发报前，没有可用于调参的完整验证批次")
    records = []
    for history_days in (28, 56):
        for rule in ("flat24", "linear72"):
            params = Parameters(history_days, rule)
            errors = []
            for day in eligible_days:
                frame, _ = forecast_day(load, day, schedule, params)
                actual = load.reindex(frame["target_end"]).to_numpy()
                errors.extend(np.abs(actual - frame["stl"].to_numpy()))
            records.append({"history_days": history_days, "trend_rule": rule,
                            "validation_days": len(eligible_days),
                            "valid_points": int(np.isfinite(errors).sum()),
                            "mae_mw": float(np.nanmean(errors))})
    table = pd.DataFrame(records).sort_values(["mae_mw", "history_days", "trend_rule"])
    best = table.iloc[0]
    if not np.isfinite(best["mae_mw"]):
        raise ValueError("验证期没有有效实际负荷，无法选择参数")
    return Parameters(int(best["history_days"]), best["trend_rule"]), table, eligible_days


def replay(load, schedule, params, dates):
    """历史按发报顺序回放。先结算已到齐的旧批次，再发出新批次。"""
    pending = deque()
    past_errors = deque(maxlen=BIAS_BATCHES)
    batches = []
    for target_day in dates:
        issue = schedule.issue(target_day)
        while pending and pending[0][0] <= issue:
            ready_at, previous = pending.popleft()
            actual = load.reindex(previous["target_end"]).to_numpy()
            if np.isfinite(actual).all():
                past_errors.append((previous["target_date"].iat[0], ready_at,
                                    actual - previous["stl"].to_numpy()))

        frame, _ = forecast_day(load, target_day, schedule, params)
        active = len(past_errors) == BIAS_BATCHES
        assert all(ready_at <= issue for _, ready_at, _ in past_errors)
        bias = np.mean([error for _, _, error in past_errors], axis=0) if active else np.zeros(HOURS)
        frame["stl_ma_bias"] = frame["stl"] + bias
        frame["correction_mw"] = bias
        frame["available_error_batches"] = len(past_errors)
        frame["correction_applied"] = active
        frame["correction_source_dates"] = ",".join(day for day, _, _ in past_errors) if active else ""
        frame["latest_correction_source_available_at"] = past_errors[-1][1] if active else pd.NaT
        batches.append(frame)
        pending.append((frame["target_end"].iat[-1] + schedule.delay, frame))
    return pd.concat(batches, ignore_index=True)


def evaluate(predicted, load):
    """用共同有效点比较所有方法。峰值指标要求一整天 24 点都有效。"""
    result = predicted.copy()
    result["actual_mw"] = load.reindex(result["target_end"]).to_numpy()
    result["score_eligible"] = np.isfinite(result[["actual_mw", *METHODS]].to_numpy()).all(axis=1)
    result["day_type"] = np.where(pd.to_datetime(result["target_date"]).dt.dayofweek < 5,
                                  "Mon-Fri", "Weekend")
    valid = result.loc[result["score_eligible"]]
    if valid.empty:
        raise ValueError("没有可用于共同评分的目标小时")

    scores = []
    peaks = []
    for group, data in [("All", result), *result.groupby("day_type", sort=False),
                        *[(f"HE{he:02d}", block) for he, block in result.groupby("hour_ending")]]:
        matched = data.loc[data["score_eligible"]]
        for method in METHODS:
            errors = matched["actual_mw"] - matched[method]
            scores.append({"group": group, "method": method, "valid_hours": len(matched),
                           "expected_hours": len(data), "coverage": len(matched) / len(data),
                           "mae_mw": errors.abs().mean(), "rmse_mw": np.sqrt((errors ** 2).mean()),
                           "mean_actual_minus_pred_mw": errors.mean()})

    for day, block in valid.groupby("target_date", sort=True):
        if len(block) != HOURS:
            continue
        # 行按 HE 排序，idxmax 在并列时自然选最早的小时。
        actual = block.loc[block["actual_mw"].idxmax()]
        for method in METHODS:
            prediction = block.loc[block[method].idxmax()]
            peak_error = actual["actual_mw"] - prediction[method]
            peaks.append({"target_date": day, "method": method, "actual_peak_mw": actual["actual_mw"],
                          "predicted_peak_mw": prediction[method], "actual_peak_he": actual["hour_ending"],
                          "predicted_peak_he": prediction["hour_ending"], "peak_error_mw": peak_error,
                          "abs_peak_error_mw": abs(peak_error), "peak_underestimate_mw": max(0, peak_error),
                          "abs_peak_time_error_hours": abs(actual["hour_ending"] - prediction["hour_ending"])})
    peak_details = pd.DataFrame(peaks)
    if peak_details.empty:
        raise ValueError("没有完整 24 点可用的日期，无法计算日峰值指标")
    peak_summary = peak_details.groupby("method", sort=False).agg(
        peak_days=("target_date", "size"),
        peak_mae_mw=("abs_peak_error_mw", "mean"),
        mean_peak_underestimate_mw=("peak_underestimate_mw", "mean"),
        peak_underestimate_days=("peak_error_mw", lambda x: (x > 0).sum()),
        peak_time_mae_hours=("abs_peak_time_error_hours", "mean"),
    ).reset_index()
    return result, pd.DataFrame(scores), peak_details, peak_summary


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.8f")


def draw_figures(out, first_fit, predictions, ordinary, diagnostic):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for ax, series, name, color in zip(
        axes, [first_fit.observed, first_fit.trend, first_fit.seasonal, first_fit.resid],
        ["Observed", "Trend", "Daily seasonal", "Remainder"],
        ["#284f67", "#137a71", "#b16427", "#6d5792"],
    ):
        tail = series.iloc[-7 * HOURS:]
        ax.plot(tail.index, tail.to_numpy(), color=color, linewidth=1.3)
        ax.set_ylabel(f"{name}\n(MW)")
        ax.grid(alpha=.22)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=EST))
    axes[-1].set_xlabel("Hour ending (EST, UTC-05:00)")
    fig.suptitle("Ontario demand | STL history available for the first test forecast", weight="bold")
    fig.savefig(out / "01_stl_decomposition.png", dpi=170)
    plt.close(fig)

    for filename, day, description in [("02_typical_day.png", ordinary, "Typical weekday"),
                                        ("03_diagnostic_day.png", diagnostic, "Error diagnostic")]:
        data = predictions.loc[predictions["target_date"] == day]
        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        ax.plot(data["hour_ending"], data["actual_mw"], color="#172e40", lw=2.6, label="Actual")
        styles = [("d7_same_hour", "#7b6c9e", ":"), ("stl", "#137a71", "--"),
                  ("stl_ma_bias", "#bd5c25", "-")]
        for method, color, line in styles:
            ax.plot(data["hour_ending"], data[method], color=color, ls=line, lw=1.9, label=METHODS[method])
        obs_peak = data.loc[data["actual_mw"].idxmax()]
        pred_peak = data.loc[data["stl_ma_bias"].idxmax()]
        ax.scatter(obs_peak["hour_ending"], obs_peak["actual_mw"], color="#172e40", zorder=6, marker="o")
        ax.scatter(pred_peak["hour_ending"], pred_peak["stl_ma_bias"], color="#bd5c25", zorder=6, marker="D")
        ax.set_title(f"{description} | Ontario demand | {day}\n"
                     f"Actual peak {obs_peak['actual_mw']:,.0f} MW at HE{obs_peak['hour_ending']:02d}  |  "
                     f"Corrected peak {pred_peak['stl_ma_bias']:,.0f} MW at HE{pred_peak['hour_ending']:02d}",
                     loc="left", weight="bold")
        ax.set(xlim=(.65, 24.35), xlabel="Target hour ending (EST)", ylabel="Ontario demand (MW)")
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24], labels=[f"HE{i:02d}" for i in (1, 4, 8, 12, 16, 20, 24)])
        ax.grid(alpha=.22)
        ax.legend(loc="best", ncol=2, frameon=False)
        fig.savefig(out / filename, dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data" / "PUB_Demand_2025.csv"), help="本地 IESO CSV 或 HTTPS URL")
    parser.add_argument("--out", type=Path, default=ROOT / "results", help="结果目录")
    parser.add_argument("--issue-time", default="16:00", help="D-1 日的发报时刻 HH:MM，固定 EST")
    parser.add_argument("--delay-minutes", type=int, default=60, help="负荷在小时结束后多少分钟可用")
    args = parser.parse_args()
    clock = time.fromisoformat(args.issue_time)
    if clock.tzinfo is not None or clock.second or clock.microsecond or args.delay_minutes < 0:
        parser.error("发报时刻需使用 HH:MM，延迟需是非负整数分钟")
    schedule = Schedule(pd.Timedelta(hours=clock.hour, minutes=clock.minute),
                        pd.Timedelta(minutes=args.delay_minutes))
    try:
        source_rows, load, ramp_flags, quality = read_load(args.input)
    except OSError as exc:
        parser.exit(2, f"读取数据失败：{exc}\n请下载 {SOURCE_URL} 后用 --input 指定本地文件。\n")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print("数据检查：", json.dumps(quality, ensure_ascii=False), flush=True)
    save_csv(source_rows, out / "normalized_load.csv")
    save_csv(ramp_flags.reset_index(), out / "ramp_flags.csv")
    (out / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")

    validation_dates = pd.date_range("2025-09-01", "2025-09-30", freq="D", tz=EST)
    test_dates = pd.date_range("2025-10-01", "2025-12-31", freq="D", tz=EST)
    params, validation, used_dates = choose_parameters(load, schedule, validation_dates, test_dates[0])
    save_csv(validation, out / "validation_candidates.csv")
    print("验证选定参数：", params, flush=True)
    dates = validation_dates.append(test_dates)
    predicted = replay(load, schedule, params, dates)
    testing = predicted.loc[predicted["target_date"] >= "2025-10-01"].copy()

    # 关键检验：改动发报时还不可用的原始值，保留这次发报的所有预测。
    first_issue = schedule.issue(test_dates[0])
    changed = load.copy()
    unavailable = changed.index + schedule.delay > first_issue
    changed.loc[unavailable] += 5000
    retest = replay(changed, schedule, params, validation_dates.append(test_dates[:1]))
    retest = retest.loc[retest["target_date"] == "2025-10-01"]
    original = testing.loc[testing["target_date"] == "2025-10-01"]
    compare = [*METHODS, "correction_mw"]
    np.testing.assert_array_equal(original[compare].to_numpy(), retest[compare].to_numpy())
    assert (testing["last_observed_available_at"] <= testing["issue_time"]).all()
    corrected = testing.loc[testing["correction_applied"]]
    assert (corrected["latest_correction_source_available_at"] <= corrected["issue_time"]).all()
    assert len(testing) == len(test_dates) * HOURS
    for day, batch in testing.groupby("target_date", sort=False):
        assert batch["hour_ending"].tolist() == list(range(1, 25)), day
        assert batch["target_end"].iat[-1] == pd.Timestamp(day, tz=EST) + DAY, day
    if (args.issue_time == "16:00" and args.delay_minutes == 60
            and original["last_observed_end"].iat[0] == first_issue - HOUR):
        assert original["lead_hours_from_last_observation"].tolist() == list(range(10, 34))

    scored, scores, daily_peaks, peak_summary = evaluate(testing, load)
    # 导出 9 月基础预测，便于逐条重算第一批测试的 7 个校准误差。
    calibration = predicted.loc[predicted["target_date"] < "2025-10-01", [
        "target_date", "hour_ending", "issue_time", "last_observed_end", "target_end", "stl",
    ]].copy()
    calibration["actual_mw"] = load.reindex(calibration["target_end"]).to_numpy()
    calibration["batch_available_at"] = calibration.groupby("target_date")["target_end"].transform("max") + schedule.delay
    calibration["actual_minus_base_mw"] = calibration["actual_mw"] - calibration["stl"]
    save_csv(calibration, out / "calibration_predictions.csv")
    # 长表按方法存预测与校正量，宽表保留 STL 预测分量供复查。
    meta = ["target_date", "hour_ending", "target_end", "issue_time", "last_observed_end",
            "last_observed_available_at", "lead_hours_from_last_observation", "day_type",
            "actual_mw", "score_eligible", "correction_mw", "available_error_batches",
            "correction_applied", "correction_source_dates", "latest_correction_source_available_at"]
    long = scored.melt(id_vars=meta, value_vars=list(METHODS), var_name="method", value_name="prediction_mw")
    uncorrected = long["method"] != "stl_ma_bias"
    long.loc[uncorrected, "correction_mw"] = 0.0
    long.loc[uncorrected, "correction_applied"] = False
    long.loc[uncorrected, "correction_source_dates"] = ""
    long.loc[uncorrected, "latest_correction_source_available_at"] = pd.NaT
    long["actual_minus_prediction_mw"] = long["actual_mw"] - long["prediction_mw"]
    save_csv(long.sort_values(["target_date", "hour_ending", "method"]), out / "hourly_predictions.csv")
    save_csv(scored, out / "forecast_components.csv")
    save_csv(scores, out / "error_metrics.csv")
    save_csv(daily_peaks, out / "daily_peaks.csv")
    save_csv(peak_summary, out / "peak_metrics.csv")

    complete_days = daily_peaks.loc[daily_peaks["method"] == "stl", "target_date"]
    paired = scored.loc[scored["target_date"].isin(complete_days)].copy()
    paired["base_ae"] = (paired["actual_mw"] - paired["stl"]).abs()
    paired["corrected_ae"] = (paired["actual_mw"] - paired["stl_ma_bias"]).abs()
    daily_errors = paired.groupby(["target_date", "day_type"])[["base_ae", "corrected_ae"]].mean().reset_index()
    weekdays = daily_errors.loc[daily_errors["day_type"] == "Mon-Fri"]
    typical_pool = weekdays if not weekdays.empty else daily_errors
    ordinary = typical_pool.loc[(typical_pool["base_ae"] - typical_pool["base_ae"].median()).abs().idxmin(), "target_date"]
    daily_errors["correction_change_mae_mw"] = daily_errors["corrected_ae"] - daily_errors["base_ae"]
    # 诊断图选校正退步最大的一天。若没有退步，则选校正后误差最大的一天。
    diagnostic_col = "correction_change_mae_mw" if daily_errors["correction_change_mae_mw"].max() > 0 else "corrected_ae"
    diagnostic = daily_errors.loc[daily_errors[diagnostic_col].idxmax(), "target_date"]
    save_csv(daily_errors, out / "daily_errors.csv")
    _, first_fit = forecast_day(load, test_dates[0], schedule, params)
    draw_figures(out, first_fit, scored, ordinary, diagnostic)
    run_info = {
        "issue_time_est": args.issue_time, "assumed_delay_minutes": args.delay_minutes,
        "first_test_issue": str(first_issue), "validation_target_start": str(used_dates[0].date()),
        "validation_target_end": str(used_dates[-1].date()), "history_days": params.history_days,
        "trend_rule": params.trend_rule, "period": HOURS, "seasonal": 13, "trend": 49,
        "low_pass": 25, "robust": True, "correction_batches": BIAS_BATCHES,
        "tested_target_days": len(test_dates), "tested_hours": len(scored),
        "valid_common_hours": int(scored["score_eligible"].sum()),
        "corrected_target_days": int(corrected["target_date"].nunique()),
        "ordinary_day": ordinary, "diagnostic_day": diagnostic,
        "diagnostic_selection": diagnostic_col,
        "changed_unavailable_actuals": int(unavailable.sum()), "first_issue_invariance_check": "PASS",
        "all_batch_availability_and_alignment_checks": "PASS",
    }
    (out / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(scores.loc[scores["group"] == "All"].round(3).to_string(index=False))
    print(peak_summary.round(3).to_string(index=False))
    print("预测不变性、可用时点及小时对齐检查通过。结果目录：", out)


if __name__ == "__main__":
    main()
