# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:40:17 2025

@author: aless
"""

# - Single file containing the registry and a few useful steps.
# - Add new steps by writing a function with the decorator @step("name").

from typing import Callable, Dict, List, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random
import logging
logger = logging.getLogger("pipeline")


# --- utility ---
def _slugify(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w\-\.]", "", text)
    return text


# ---------- Registry ----------

STEP_REGISTRY: Dict[str, Callable] = {}

def step(name: str):
    """Decorator to register pipeline steps by name."""
    def deco(fn: Callable):
        if name in STEP_REGISTRY:
            raise ValueError(f"Step '{name}' already registered.")
        STEP_REGISTRY[name] = fn
        return fn
    return deco


def _ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    

# ----------- Assign colors -----------------

@step("assign_colors_by_technology")
def assign_colors_by_technology(
    df: pd.DataFrame,
    ctx: dict,
    technology_col: str = "Technology name",
    cmap_name: str = "tab20",           # any Matplotlib colormap
    seed: int = 42,                     # fixed seed for reproducibility
    shuffle: bool = False,              # shuffle color assignment if True
    **_,
):
    """Assign a distinct color to each technology and store it in ctx['color_map']."""
    if technology_col not in df.columns:
        raise KeyError(f"Column '{technology_col}' not found.")

    techs = sorted(df[technology_col].dropna().unique().tolist())
    if not techs:
        return df, ctx

    cmap = cm.get_cmap(cmap_name, len(techs))
    colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    random.seed(seed)
    if shuffle:
        random.shuffle(colors)

    color_map = {tech: colors[i % len(colors)] for i, tech in enumerate(techs)}
    ctx["color_map"] = color_map
    return df, ctx


# ---------- Basic steps ----------

@step("drop_columns")
def drop_columns(df: pd.DataFrame, ctx: dict, names: Optional[List[str]] = None, **_):
    """Drop given columns if present."""
    names = names or []
    cols = [c for c in names if c in df.columns]
    if cols:
        df = df.drop(columns=cols, errors="ignore")
    return df, ctx


@step("coerce_numeric")
def coerce_numeric(df: pd.DataFrame, ctx: dict, columns=None, **_):
    """Convert given columns to numeric (coerce errors to NaN)."""
    columns = columns or []
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, ctx


@step("filter_rows")
def filter_rows(df: pd.DataFrame, ctx: dict, query: Optional[str] = None, **_):
    """Filter rows using pandas 'query' expression."""
    if query:
        df = df.query(query)
    return df, ctx


@step("rename_columns")
def rename_columns(df: pd.DataFrame, ctx: dict, mapping: dict = None, **_):
    """Rename columns using a provided mapping."""
    mapping = mapping or {}
    df = df.rename(columns=mapping)
    return df, ctx


@step("select_columns")
def select_columns(df: pd.DataFrame, ctx: dict, names: Optional[List[str]] = None, **_):
    """Keep only the given columns (if they exist)."""
    names = names or []
    keep = [c for c in names if c in df.columns]
    if keep:
        df = df[keep]
    return df, ctx


# ---------- Stats steps ----------

@step("group_stats")
def group_stats(
    df: pd.DataFrame,
    ctx: dict,
    by: Optional[List[str]] = None,
    metrics: Optional[dict] = None,
    save_as: Optional[str] = None,
    **_,
):
    """
    Generic grouped statistics.
    Example metrics: {"CAPEX": ["mean", "median"], "efficiency": ["mean"]}
    """
    by = by or []
    metrics = metrics or {}
    if not by or not metrics:
        return df, ctx

    out = df.groupby(by).agg(metrics)
    out.columns = ['_'.join(col).strip() if isinstance(col, tuple) else str(col) for col in out.columns.values]
    out = out.reset_index()

    if save_as:
        _ensure_parent_dir(save_as)
        out.to_csv(save_as, index=False)
        ctx.setdefault("artifacts", []).append(save_as)

    return out, ctx


@step("stats_by_technology")
def stats_by_technology(
    df: pd.DataFrame,
    ctx: dict,
    technology_col: str = "Technology name",
    columns: Optional[List[str]] = None,          # e.g. ["CAPEX","OPEX"]
    metrics: Optional[List[str]] = None,          # e.g. ["mean","median","min","max","count","std"]
    output_dir: Optional[str] = None,             # default: ctx["base_out_dir"]
    filename_prefix: str = "statistics",
    replace_df: bool = False,                     # if True, merge results back and replace df
    **_,
):
    """
    Compute grouped statistics by technology for multiple columns.
    Saves one CSV per column and stores each table in ctx["stats"][col].
    """
    if technology_col not in df.columns:
        raise KeyError(f"Missing technology column: '{technology_col}'")

    columns = columns or []
    if not columns:
        return df, ctx

    metrics = metrics or ["mean", "median", "min", "max", "count", "std"]

    # Decide output directory
    base_out_dir = ctx.get("base_out_dir", "./out")
    out_dir = output_dir or base_out_dir
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))

    # Compute stats per column
    merged = None
    for col in columns:
        if col not in df.columns:
            # skip missing columns gracefully
            continue

        # groupby & agg
        out = df.groupby(technology_col)[col].agg(metrics).reset_index()

        # Save CSV: statistics_<col>.csv
        slug = _slugify(col)
        fname = os.path.join(out_dir, f"{filename_prefix}_{slug}.csv")
        _ensure_parent_dir(fname)
        out.to_csv(fname, index=False)
        ctx.setdefault("artifacts", []).append(fname)

        # Store in context
        ctx.setdefault("stats", {})[col] = out

        # Prepare merged df (optional)
        if replace_df:
            # rename metrics columns with suffix to avoid collisions when merging multiple cols
            rename_map = {m: f"{col}_{m}" for m in metrics}
            out_ren = out.rename(columns=rename_map)
            merged = out_ren if merged is None else merged.merge(out_ren, on=technology_col, how="outer")

    # Replace df only if requested and at least one col processed
    if replace_df and merged is not None:
        return merged, ctx

    # Otherwise passthrough original df
    return df, ctx




# ---------- Plot steps ----------

@step("plot_bar_by_technology")
def plot_bar_by_technology(
    df: pd.DataFrame,
    ctx: dict,
    stats_source_col: str,                    # e.g. "CAPEX" or "OPEX"
    technology_col: str = "Technology name",
    value_col: str = "mean",                  # one of the computed metrics (must exist in stats table)
    error_col: str = "std",                   # if missing in stats table, no error bars
    count_col: str = "count",                 # if missing in stats table, no annotation
    top_n: Optional[int] = None,              # default=None -> show ALL; set to int to limit
    out_dir: Optional[str] = None,            # directory; filename is auto
    title: Optional[str] = None,
    cmap_name: str = "tab20",                 # used only if no ctx["color_map"]
    tick_labelsize: int = 9,                  # smaller x tick labels
    **_,
):
    """
    General bar plot by technology for any stats_source_col computed by stats_by_technology.
    One bar per technology: height = <value_col>, error = <error_col> (if present), annotation = <count_col> (if present).
    Saves to <out_dir>/barplot_<stats_source_col>_<value_col>.png (auto if out_dir is None).
    """
    logger.info(f"[plot_bar_by_technology] source='{stats_source_col}', value='{value_col}', top_n={top_n}")

    # Retrieve stats table from ctx and validate
    stats_all = ctx.get("stats", {})
    if stats_source_col not in stats_all:
        raise KeyError(
            f"No stats found in ctx for column '{stats_source_col}'. "
            f"Run 'stats_by_technology' with columns including '{stats_source_col}' first."
        )
    sdf = stats_all[stats_source_col]

    if technology_col not in sdf.columns:
        raise KeyError(f"Column '{technology_col}' not found in stats for '{stats_source_col}'.")
    if value_col not in sdf.columns:
        raise KeyError(f"Column '{value_col}' not found in stats for '{stats_source_col}'.")

    # Sort and optionally trim
    sdf_ord = sdf.sort_values(by=value_col, ascending=False).copy()
    if isinstance(top_n, int) and top_n > 0:
        sdf_ord = sdf_ord.head(top_n)

    # Prepare color map: use ctx if available, otherwise create and store
    color_map = ctx.get("color_map")
    techs = sdf_ord[technology_col].tolist()

    if not color_map:
        logger.info("[plot_bar_by_technology] No color_map in ctx; generating a temporary palette.")
        cmap = cm.get_cmap(cmap_name, len(techs))
        color_map = {tech: mcolors.to_hex(cmap(i)) for i, tech in enumerate(techs)}
        ctx["color_map"] = ctx.get("color_map", {})
        ctx["color_map"].update(color_map)

    colors = [color_map.get(t, "#999999") for t in techs]

    # Values and optional errors
    y = sdf_ord[value_col].values
    yerr = sdf_ord[error_col].values if error_col in sdf_ord.columns else None

    # Decide output directory & filename
    base_out_dir = ctx.get("base_out_dir", "./out")
    out_dir = out_dir or base_out_dir
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))
    out_fp = os.path.join(out_dir, f"barplot_{_slugify(stats_source_col)}_{_slugify(value_col)}.png")

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    x = np.arange(len(techs))
    bars = plt.bar(
        x, y,
        yerr=yerr if yerr is not None else None,
        capsize=3 if yerr is not None else 0,
        edgecolor="black",
        linewidth=0.6
    )
    for bar, c in zip(bars, colors):
        bar.set_color(c)

    # Axis/labels styling
    plt.xticks(x, techs, rotation=45, ha="right", fontsize=tick_labelsize)
    plt.ylabel(value_col, fontweight="bold")
    plt.xlabel(technology_col, fontweight="bold")
    ttl = title or f"{stats_source_col} — {value_col} by {technology_col}"
    plt.title(ttl, fontweight="bold")
    plt.tight_layout()

    # Annotate counts ABOVE bars (taking error bar into account if present)
    if count_col in sdf_ord.columns:
        counts = sdf_ord[count_col].values
        # compute a small offset relative to data range
        y_top_for_offset = y + (yerr if yerr is not None else 0)
        data_span = (np.nanmax(y_top_for_offset) - np.nanmin(y_top_for_offset)) if len(y_top_for_offset) else 1.0
        offset = 0.02 * data_span if data_span > 0 else 0.02  # 2% of span

        for xi, yi, cnt, err in zip(x, y, counts, (yerr if yerr is not None else np.zeros_like(y))):
            if pd.notna(yi):
                y_annot = yi + (err if pd.notna(err) else 0.0) + offset
                try:
                    txt = f"{int(cnt)}"
                except Exception:
                    txt = f"{cnt}"
                plt.text(xi, y_annot, txt, ha="center", va="bottom", fontsize=9)

    plt.savefig(out_fp, dpi=200)
    plt.close()

    logger.info(f"[plot_bar_by_technology] Saved plot to {out_fp}")
    ctx.setdefault("artifacts", []).append(out_fp)
    return df, ctx


@step("plot_gaussian_with_samples_by_technology")
def plot_gaussian_with_samples_by_technology(
    df: pd.DataFrame,
    ctx: dict,
    stats_source_col: str,                     # e.g., "CAPEX" or "OPEX" (used to pick mean/std from ctx['stats'])
    data_source_col: str = None,               # column in df holding raw samples; default: = stats_source_col
    technology_col: str = "Technology name",
    mean_col: str = "mean",
    std_col: str = "std",
    tech_select: Optional[List[str]] = None,   # ["all"] or list of technologies
    out_dir: Optional[str] = None,             # default: <base_out_dir>/gaussians_<source>_with_samples
    x_sigma_span: float = 4.0,                 # x-range around mean
    show_hist: bool = False,                   # overlay density histogram
    bins: int = 30,                            # bins for histogram
    show_rug: bool = True,                     # show raw sample points along the x-axis baseline
    rug_size: float = 30.0,                    # marker size for rug points
    rug_alpha: float = 0.7,                    # alpha for rug points
    line_width: float = 2.0,                   # gaussian line width
    tick_labelsize: int = 9,                   # ticks font size
    cmap_name: str = "tab20",                  # only used if no ctx["color_map"]
    use_original_df: bool = False,             # take samples from ctx["_original_df"] instead of current df
    **_,
):
    """
    Plot Normal(mu, sigma) curves (from aggregated stats) and overlay empirical samples.
    - Reads mean/std from ctx['stats'][stats_source_col].
    - Reads raw samples from `df[data_source_col]` (or ctx['_original_df'] if use_original_df=True).
    - One PNG per technology.
    """
    logger.info(f"[plot_gaussian_with_samples_by_technology] source='{stats_source_col}', tech_select={tech_select}, hist={show_hist}, rug={show_rug}")

    # pick stats table
    stats_all = ctx.get("stats", {})
    if stats_source_col not in stats_all:
        raise KeyError(
            f"No stats found in ctx for '{stats_source_col}'. "
            f"Run 'stats_by_technology' including '{stats_source_col}' first."
        )
    sdf = stats_all[stats_source_col]

    # validate required columns in stats
    for col in [technology_col, mean_col, std_col]:
        if col not in sdf.columns:
            raise KeyError(f"Column '{col}' required in stats for '{stats_source_col}'.")

    # where to get raw samples from
    base_df = ctx.get("_original_df") if use_original_df else df
    data_source_col = data_source_col or stats_source_col
    if technology_col not in base_df.columns or data_source_col not in base_df.columns:
        raise KeyError(
            f"Columns '{technology_col}' and/or '{data_source_col}' not found in the selected data frame "
            f"({'_original_df' if use_original_df else 'df'})."
        )

    # select technologies
    if tech_select is None or (len(tech_select) == 1 and str(tech_select[0]).lower() == "all"):
        tech_list = sorted(sdf[technology_col].dropna().unique().tolist())
    else:
        tech_list = tech_select

    # output dir
    base_out_dir = ctx.get("base_out_dir", "./out")
    out_dir = out_dir or os.path.join(base_out_dir, f"gaussians_{_slugify(stats_source_col)}_with_samples")
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))

    # colors
    color_map = ctx.get("color_map")
    if not color_map:
        logger.info("[plot_gaussian_with_samples_by_technology] No color_map in ctx; creating a temporary palette.")
        cmap = cm.get_cmap(cmap_name, len(tech_list))
        color_map = {t: mcolors.to_hex(cmap(i)) for i, t in enumerate(tech_list)}
        ctx["color_map"] = ctx.get("color_map", {})
        ctx["color_map"].update(color_map)

    for tech in tech_list:
        row = sdf[sdf[technology_col] == tech]
        if row.empty:
            logger.warning(f"[plot_gaussian_with_samples_by_technology] Technology '{tech}' not found in stats. Skipping.")
            continue

        mu = float(row[mean_col].iloc[0])
        sigma = float(row[std_col].iloc[0])
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            logger.warning(f"[plot_gaussian_with_samples_by_technology] Invalid sigma for '{tech}' (mu={mu}, sigma={sigma}). Skipping.")
            continue

        # raw samples for this technology
        sub = base_df[base_df[technology_col] == tech]
        xsamp = pd.to_numeric(sub[data_source_col], errors="coerce").dropna().values
        # define x-range covering both gaussian span and sample min/max (if any)
        if xsamp.size:
            xmin_data, xmax_data = np.nanmin(xsamp), np.nanmax(xsamp)
        else:
            xmin_data, xmax_data = np.nan, np.nan

        xg_min = mu - x_sigma_span * sigma
        xg_max = mu + x_sigma_span * sigma
        x_min = np.nanmin([xg_min, xmin_data]) if np.isfinite(xmin_data) else xg_min
        x_max = np.nanmax([xg_max, xmax_data]) if np.isfinite(xmax_data) else xg_max

        xs = np.linspace(x_min, x_max, 400)
        pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)

        color = color_map.get(tech, "#333333")
        fname = os.path.join(out_dir, f"gaussian_samples_{_slugify(stats_source_col)}_{_slugify(tech)}.png")

        plt.figure(figsize=(10, 5))

        # optional histogram (density)
        if show_hist and xsamp.size:
            plt.hist(xsamp, bins=bins, density=True, alpha=0.25)

        # gaussian curve
        plt.plot(xs, pdf, linewidth=line_width, color=color)

        # rug plot for samples (points along baseline)
        if show_rug and xsamp.size:
            # place points near y=0; scale marker size relative to figure
            y_rug = np.zeros_like(xsamp)
            plt.scatter(xsamp, y_rug, s=rug_size, alpha=rug_alpha, edgecolors="none")

        # axes/labels
        plt.xlabel(stats_source_col, fontweight="bold")
        plt.ylabel("Density", fontweight="bold")
        plt.title(f"{tech} — Normal(μ={mu:.2f}, σ={sigma:.2f})", fontweight="bold")
        plt.xticks(fontsize=tick_labelsize)
        plt.yticks(fontsize=tick_labelsize)
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

        logger.info(f"[plot_gaussian_with_samples_by_technology] Saved {fname}")
        ctx.setdefault("artifacts", []).append(fname)

    return df, ctx

