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
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network
import textwrap
logger = logging.getLogger("pipeline")


# --- utility ---
def _slugify(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w\-\.]", "", text)
    return text

def _split_tokens(s, seps=(";", ",", "|")):
    """Split string s by any of the separators; return cleaned list (drop empty)."""
    if pd.isna(s):
        return []
    if not isinstance(s, str):
        s = str(s)
    pattern = "|".join(map(re.escape, seps))
    parts = re.split(pattern, s)
    return [p.strip() for p in parts if p and p.strip()]

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


@step("assign_sector_colors")
def assign_sector_colors(
    df, ctx,
    sector_col: str = "Sector",
    palette: str = "tab20",
    seed: int = 42,
    **_,
):
    """Assign a distinct color to each Sector and store in ctx['sector_colors']."""
    sectors = sorted(s for s in df[sector_col].dropna().unique().tolist())
    cmap = cm.get_cmap(palette, max(1, len(sectors)))
    sector_colors = {s: mcolors.to_hex(cmap(i)) for i, s in enumerate(sectors)}
    ctx["sector_colors"] = sector_colors
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


# ----------------- Plots maps ---------------------

@step("build_flow_graph")
def build_flow_graph(
    df: pd.DataFrame,
    ctx: dict,
    input_col: str = "Input",
    technology_col: str = "Technology name",
    output_col: str = "Output",
    token_separators: list = (";", ",", "|"),
    # --- how to get raw weights ---
    weight_mode: str = "stats",                # "stats" | "column" | "uniform"
    weight_stats_source_col: str = "CAPEX",    # if stats
    weight_stats_metric: str = "mean",         # e.g. "mean" | "count" | "median"
    weight_column: str = None,                 # if column
    # --- post-processing/normalization of weights ---
    weight_scale: float = 1.0,                 # e.g. 0.01 to divide by 100
    weight_transform: str = "none",            # "none" | "log10" | "sqrt"
    weight_clip_min: float = 0.0,              # clip after transform
    weight_clip_max: float = float("inf"),
    min_weight: float = 0.0,                   # prune tiny edges after transform
    **_,
):
    """
    Build a flow graph Input -> Technology -> Output with weights.
    Saves nodes and edges in ctx["flow_graph"].
    """
    logger.info("[build_flow_graph] building edges and nodes")

    # Validate columns
    for col in [input_col, technology_col, output_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df.")

    # Decide weight per TECHNOLOGY (for both edge families)
    tech_weight = {}

    if weight_mode == "stats":
        stats_all = ctx.get("stats", {})
        if weight_stats_source_col not in stats_all:
            raise KeyError(
                f"No stats found in ctx for '{weight_stats_source_col}'. "
                f"Run 'stats_by_technology' first."
            )
        sdf = stats_all[weight_stats_source_col]
        if technology_col not in sdf.columns or weight_stats_metric not in sdf.columns:
            raise KeyError(f"Stats table for '{weight_stats_source_col}' lacks '{technology_col}' or '{weight_stats_metric}'.")
        tech_weight = dict(zip(sdf[technology_col], sdf[weight_stats_metric]))
        weight_name = f"{weight_stats_source_col}_{weight_stats_metric}"

    elif weight_mode == "column":
        if weight_column is None or weight_column not in df.columns:
            raise KeyError("weight_mode='column' requires an existing 'weight_column' in df.")
        # aggregate per technology (sum by default)
        grp = df.groupby(technology_col, dropna=True)[weight_column].sum(min_count=1)
        tech_weight = grp.to_dict()
        weight_name = f"{weight_column}_sum"

    elif weight_mode == "uniform":
        techs = df[technology_col].dropna().unique().tolist()
        tech_weight = {t: 1.0 for t in techs}
        weight_name = "uniform"

    else:
        raise ValueError("weight_mode must be one of {'stats','column','uniform'}")
        
    def _postproc(w):
        w2 = float(w) * float(weight_scale)
        if weight_transform == "log10":
            # avoid log(0)
            w2 = np.log10(max(w2, 1e-9))
        elif weight_transform == "sqrt":
            w2 = np.sqrt(max(w2, 0.0))
        # clip
        w2 = min(max(w2, weight_clip_min), weight_clip_max)
        return w2

    if weight_mode in ("stats", "column", "uniform"):
        # apply to all tech weights
        tech_weight = {t: _postproc(v) for t, v in tech_weight.items() if pd.notna(v)}
        weight_name = f"{weight_name}_scaled{'' if weight_scale==1 else f'x{weight_scale}'}_{weight_transform}"


    # Build edges (Input->Tech and Tech->Output), expanding multi tokens
    edges = []
    for _, row in df[[input_col, technology_col, output_col]].iterrows():
        tech = row[technology_col]
        if pd.isna(tech):
            continue
        w = tech_weight.get(tech, np.nan)
        if not pd.notna(w):
            continue  # skip tech with no weight

        inputs = _split_tokens(row[input_col], token_separators)
        outputs = _split_tokens(row[output_col], token_separators)

        for inp in inputs:
            edges.append(("input", inp, "tech", tech, w))
        for outp in outputs:
            edges.append(("tech", tech, "output", outp, w))

    if not edges:
        raise ValueError("No edges built. Check your columns and separators.")

    edges_df = pd.DataFrame(edges, columns=["src_type", "src", "dst_type", "dst", "weight"])

    # Aggregate same edges (sum weights)
    edges_df = (
        edges_df.groupby(["src_type", "src", "dst_type", "dst"], as_index=False)["weight"]
        .sum()
    )

    # Prune tiny weights
    if min_weight > 0:
        edges_df = edges_df[edges_df["weight"] >= min_weight].reset_index(drop=True)

    # Nodes df
    inputs = edges_df.loc[edges_df["src_type"] == "input", "src"].tolist()
    outputs = edges_df.loc[edges_df["dst_type"] == "output", "dst"].tolist()
    techs = edges_df.loc[edges_df["src_type"] == "tech", "src"].tolist() + edges_df.loc[edges_df["dst_type"] == "tech", "dst"].tolist()
    techs = sorted(set(techs))

    nodes = []
    for name in sorted(set(inputs)):
        nodes.append(("input", name))
    for name in techs:
        nodes.append(("tech", name))
    for name in sorted(set(outputs)):
        nodes.append(("output", name))

    nodes_df = pd.DataFrame(nodes, columns=["type", "name"])

    # Sankey index mapping
    node_index = {name: i for i, name in enumerate(nodes_df["name"].tolist())}
    

    ctx["flow_graph"] = {
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "node_index": node_index,
        "weight_name": weight_name,
    }

    logger.info(f"[build_flow_graph] nodes={len(nodes_df)}, edges={len(edges_df)}, weight={weight_name}")
    return df, ctx


@step("plot_carrier_networkx")
def plot_carrier_networkx(
    df: pd.DataFrame,
    ctx: dict,
    out_dir: str = None,
    edge_width_min: float = 0.6,
    edge_width_max: float = 6.0,
    alpha: float = 0.7,
    connectionstyle: str = "arc3,rad=0.06",
    tick_labelsize: int = 9,
    node_size: int = 900,
    layout_mode: str = "sector_columns",
    num_sector_cols: int = 20,
    x_pad: float = 0.10,
    in_out_offset: float = 0.035,
    sort_within_sector: str = "degree",
    jitter_y: float = 0.006,
    hide_self_loops: bool = False,
    sector_col: str = "Sector",
    input_col: str = "Input",
    output_col: str = "Output",
    technology_col: str = "Technology name",
    token_separators: list = (";", ",", "|"),
    # NEW ↓↓↓
    sector_band_mode: str = "proportional",   # "proportional" | "equal"
    sector_band_gap: float = 0.02,            # vertical gap between sector bands (0..0.2)
    min_row_spacing: float = 0.01,           # minimum vertical spacing between nodes within a band
    adaptive_fig: bool = True,                 # adapt figure height to densest band
    base_figsize: tuple = (18, 10),            # (w,h) inches; h auto-scaled if adaptive_fig
    height_per_node: float = 0.16,             # inches added per node over threshold in densest band
    dense_threshold: int = 10,                 # nodes over this in a band start to grow figure height
    **_,
):
    """
    Carrier graph:
    - 'sector_columns': crea colonne per settore (5/6 ecc.). Dentro ogni colonna:
        input (x - offset), mixed (x), output (x + offset)
      Colore nodo = colore del settore (da ctx['sector_colors']).
    - altre modalità come prima ('three_column', 'two_column').
    """
    cg = ctx.get("carrier_graph")
    if not cg:
        raise RuntimeError("carrier_graph not found. Run 'build_carrier_graph' first.")
    ed = cg["edges_df"].copy()
    weight_name = cg.get("weight_name", "weight")

    if hide_self_loops:
        ed = ed[ed["in_car"] != ed["out_car"]].reset_index(drop=True)

    # 1) pesi per normalizzare larghezze
    raw_w = ed["weight"].to_numpy(dtype=float)
    if raw_w.size:
        wmin, wmax = float(np.nanmin(raw_w)), float(np.nanmax(raw_w))
    else:
        wmin, wmax = 0.0, 1.0

    # 2) set di carrier e categorie
    in_set  = set(ed["in_car"].unique())
    out_set = set(ed["out_car"].unique())
    carriers = sorted(in_set | out_set)
    pure_inputs  = set(in_set - out_set)
    pure_outputs = set(out_set - in_set)
    mixed        = set(in_set & out_set)

    # 3) stima settore per ogni carrier: prendi il settore più frequente
    #    tra le tecnologie che lo usano (come input o output)
    def _split_tokens(s):
        if pd.isna(s): return []
        if not isinstance(s, str): s = str(s)
        patt = "|".join(map(re.escape, token_separators))
        return [t.strip() for t in re.split(patt, s) if t and t.strip()]

    usage = {}  # carrier -> counter(sector)
    from collections import Counter
    for _, r in df[[sector_col, input_col, output_col]].iterrows():
        sec = r.get(sector_col)
        if pd.isna(sec): continue
        sec = str(sec)
        for c in _split_tokens(r.get(input_col)):
            usage.setdefault(c, Counter()).update([sec])
        for c in _split_tokens(r.get(output_col)):
            usage.setdefault(c, Counter()).update([sec])

    carrier_sector = {}
    for c in carriers:
        if c in usage and len(usage[c]) > 0:
            carrier_sector[c] = usage[c].most_common(1)[0][0]
        else:
            carrier_sector[c] = "Unknown"

    # 4) colori per settore (da ctx, o generane uno)
    sector_colors = ctx.get("sector_colors")
    if not sector_colors:
        # fallback semplice
        secs = sorted(set(carrier_sector.values()))
        cmap = cm.get_cmap("tab20", len(secs))
        sector_colors = {s: mcolors.to_hex(cmap(i)) for i, s in enumerate(secs)}
        ctx["sector_colors"] = sector_colors

    # 5) degree pesato per ordinamento dentro le colonne
    w_in  = ed.groupby("out_car")["weight"].sum(min_count=1).rename("in_w")
    w_out = ed.groupby("in_car")["weight"].sum(min_count=1).rename("out_w")
    deg_df = pd.DataFrame({"carrier": carriers}).merge(w_in, left_on="carrier", right_index=True, how="left") \
                                               .merge(w_out, left_on="carrier", right_index=True, how="left")
    deg_df.fillna(0.0, inplace=True)
    deg_df["deg_w"] = deg_df["in_w"] + deg_df["out_w"]
    def _deg(c): return float(deg_df.loc[deg_df["carrier"]==c, "deg_w"].values[0])

    # 6) positions
    pos = {}
    rng = np.random.default_rng(42)

    def _stack(names, x_center, y0, y1):
        """Place 'names' evenly between [y0, y1] with optional jitter."""
        names = sorted(names, key=lambda c: -_deg(c)) if sort_within_sector=="degree" else sorted(names)
        n = max(1, len(names))
        # compute vertical spacing respecting min_row_spacing
        span = max(0.001, y1 - y0)
        step = max(span / max(n,1), min_row_spacing)
        # recompute y-range if step got clamped
        total = step * (n - 1)
        if total > span:
            y0_adj = max(0.01, (y0 + y1)/2 - total/2)
            y1_adj = min(0.99, y0_adj + total)
        else:
            y0_adj, y1_adj = y0, y1

        ys = np.linspace(y0_adj, y1_adj, n) if n > 1 else np.array([(y0+y1)/2])
        if jitter_y and n > 1:
            ys = ys + rng.normal(0.0, jitter_y, size=n)
            ys = np.clip(ys, 0.01, 0.99)
        for i, name in enumerate(names):
            pos[name] = (float(x_center), float(ys[i]))

    if layout_mode == "sector_columns":
        # sectors ordered by total degree (dense first)
        sectors_sorted = sorted(
            set(carrier_sector.values()),
            key=lambda s: -sum(_deg(c) for c in carriers if carrier_sector[c]==s)
        )
        ncols = max(1, min(num_sector_cols, len(sectors_sorted)))
        xs_grid = np.linspace(x_pad, 1.0 - x_pad, ncols)

        # count per sector (for band sizing)
        sec_counts = {
            s: sum(1 for c in carriers if carrier_sector[c]==s)
            for s in sectors_sorted
        }
        total_nodes = sum(sec_counts.values())
        # assign vertical bands [y_start, y_end] per sector
        bands = {}
        if sector_band_mode == "equal" or total_nodes == 0:
            band_height = (1.0 - sector_band_gap*(len(sectors_sorted)-1)) / max(1, len(sectors_sorted))
            y = 0.02
            for s in sectors_sorted:
                y0 = y
                y1 = min(0.98, y0 + band_height)
                bands[s] = (y0, y1)
                y = y1 + sector_band_gap
        else:
            # proportional to count, enforcing a minimum height
            min_h = 0.05  # minimum band height
            weights = np.array([max(1, sec_counts[s]) for s in sectors_sorted], dtype=float)
            weights = weights / weights.sum()
            free_h = 1.0 - sector_band_gap*(len(sectors_sorted)-1) - min_h*len(sectors_sorted)
            free_h = max(0.0, free_h)
            heights = min_h + free_h*weights
            y = 0.02
            for s, h in zip(sectors_sorted, heights):
                y0 = y
                y1 = min(0.98, y0 + float(h))
                bands[s] = (y0, y1)
                y = y1 + sector_band_gap

        # place nodes inside bands; each sector-column has 3 sub-columns
        sec2x = {}
        for idx, s in enumerate(sectors_sorted):
            sec2x[s] = xs_grid[idx % ncols]

        # collect band max densities for adaptive sizing
        band_max_nodes = 0

        for s in sectors_sorted:
            x0 = sec2x[s]
            y0, y1 = bands[s]
            in_names   = [c for c in carriers if carrier_sector[c]==s and c in pure_inputs]
            mix_names  = [c for c in carriers if carrier_sector[c]==s and c in mixed]
            out_names  = [c for c in carriers if carrier_sector[c]==s and c in pure_outputs]
            band_max_nodes = max(band_max_nodes, len(in_names), len(mix_names), len(out_names))

            _stack(in_names,  x0 - in_out_offset, y0, y1)
            _stack(mix_names, x0,                 y0, y1)
            _stack(out_names, x0 + in_out_offset, y0, y1)

    else:
        # previous modes (two/three columns), keep your existing code here
        cols = []
        if layout_mode == "three_column":
            cols = [ (sorted(pure_inputs, key=lambda c:-_deg(c)), 0.0),
                     (sorted(mixed,      key=lambda c:-_deg(c)), 0.5),
                     (sorted(pure_outputs,key=lambda c:-_deg(c)), 1.0) ]
        else:
            cols = [ (sorted(in_set,  key=lambda c:-_deg(c)), 0.0),
                     (sorted(out_set, key=lambda c:-_deg(c)), 1.0) ]
        for names, x in cols:
            _stack(names, x, 0.02, 0.98)

    # 7) grafo + larghezze
    G = nx.DiGraph()
    G.add_nodes_from(carriers)
    for _, r in ed.iterrows():
        G.add_edge(r["in_car"], r["out_car"], weight=float(r["weight"]))

    if wmax > wmin:
        widths = edge_width_min + (raw_w - wmin) * (edge_width_max - edge_width_min) / (wmax - wmin)
    else:
        widths = np.full_like(raw_w, (edge_width_min + edge_width_max)/2.0)

    # 8) disegno
    base_out = ctx.get("base_out_dir", "./out")
    out_dir = out_dir or base_out
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))
    out_fp = os.path.join(out_dir, f"carrier_network_{_slugify(weight_name)}_{layout_mode}.png")

    fig_w, fig_h = base_figsize
    if adaptive_fig and layout_mode == "sector_columns":
        # grow with densest band beyond threshold
        grow = max(0, band_max_nodes - dense_threshold)
        fig_h = fig_h + grow * height_per_node
    plt.figure(figsize=(fig_w, fig_h))

    # colori per settore
    node_colors = [sector_colors.get(carrier_sector[c], "#CCCCCC") for c in carriers]
    nx.draw_networkx_nodes(G, pos, nodelist=carriers, node_color=node_colors,
                           node_size=node_size, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, font_size=tick_labelsize)

    # archi (curvi, trasparenti)
    for (u, v), w in zip(G.edges(), widths):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=float(w), alpha=alpha, arrows=True, arrowsize=10,
                               connectionstyle=connectionstyle)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=200)
    plt.close()
    logger.info(f"[plot_carrier_networkx] Saved {out_fp}")
    ctx.setdefault("artifacts", []).append(out_fp)
    return df, ctx


logger = logging.getLogger("pipeline")

def _wrap_label(s: str, max_chars: int = 18) -> str:
    """Wrap a label at spaces, max width max_chars, preserving short words."""
    if not isinstance(s, str):
        s = str(s)
    # avoid breaking very short labels
    if len(s) <= max_chars:
        return s
    return "\n".join(textwrap.wrap(s, width=max_chars, break_long_words=False))

@step("plot_flow_sankey")
def plot_flow_sankey(
    df: pd.DataFrame,
    ctx: dict,
    out_dir: str = None,

    # --- filtering controls (double threshold, per-layer top-K) ---
    min_link_value_in: float = 0.0,       # threshold for Input->Tech links
    min_link_value_out: float = 0.0,      # threshold for Tech->Output links
    top_links_per_source_in: int = None,  # keep only top-K per input source (after threshold)
    top_links_per_source_out: int = None, # keep only top-K per tech source (after threshold)

    # --- grouping of small links into "Other" nodes (optional) ---
    group_others_in: bool = False,        # aggregate small Input->Tech residuals per input into "Other inputs"
    group_others_out: bool = True,        # aggregate small Tech->Output residuals per tech into "Other outputs"
    others_label_in: str = "Other inputs",
    others_label_out: str = "Other outputs",

    # --- node ordering & labels ---
    sort_nodes_by: str = "throughput",    # "throughput" | "alpha"
    wrap_labels: bool = True,
    wrap_max_chars: int = 20,

    # --- node appearance ---
    node_pad: int = 12,
    node_thickness: int = 14,
    tech_color_alpha: float = 0.90,
    font_size: int = 12,
    title: str = None,

    # --- link colors ---
    link_color_mode: str = "source",      # "source" | "target" | "static"
    static_link_rgba: str = "rgba(160,160,160,0.35)",

    # --- optional role-based positioning (inputs left, outputs right, techs spread by role) ---
    use_role_positioning: bool = True,    # if True, set x/y arrays
    arrangement: str = "snap",            # "snap" honors x/y; otherwise plotly arranges
    x_input: float = 0.02,
    x_output: float = 0.98,
    tech_x_min: float = 0.07,
    tech_x_max: float = 0.93,
    y_pad: float = 0.04,                  # top/bottom padding (0..0.5)

    # --- export ---
    save_png: bool = False,
    png_scale: int = 2,
    **_,
):
    """
    Clean Sankey from ctx['flow_graph'] with:
      - dual thresholds (in/out) and per-layer Top-K
      - optional 'Other' aggregation on each side
      - node label wrapping
      - link coloring by {source|target|static}
      - optional role-based positioning: Input at x_input, Output at x_output, Tech spread by out/(in+out)
      - tooltips showing src, dst, and weight (with weight_name)
      - optional PNG export (requires kaleido)
    """
    # --- load graph from context ---
    fg = ctx.get("flow_graph")
    if not fg:
        raise RuntimeError("flow_graph not found in ctx. Run 'build_flow_graph' first.")

    nodes_df = fg["nodes_df"].copy()
    edges_df = fg["edges_df"].copy()
    weight_name = fg.get("weight_name", "weight")

    # --- split by type ---
    inputs = nodes_df[nodes_df["type"] == "input"]["name"].tolist()
    techs  = nodes_df[nodes_df["type"] == "tech"]["name"].tolist()
    outputs= nodes_df[nodes_df["type"] == "output"]["name"].tolist()

    # optional ordering by throughput
    if sort_nodes_by == "throughput":
        def _throughput(names):
            s = pd.concat([
                edges_df.loc[edges_df["src"].isin(names), ["src", "weight"]].rename(columns={"src": "name"}),
                edges_df.loc[edges_df["dst"].isin(names), ["dst", "weight"]].rename(columns={"dst": "name"})
            ], ignore_index=True)
            return s.groupby("name")["weight"].sum().to_dict() if not s.empty else {}
        th_in  = _throughput(inputs)
        th_t   = _throughput(techs)
        th_out = _throughput(outputs)
        inputs.sort(key=lambda n: -float(th_in.get(n, 0.0)))
        techs.sort(key=lambda n: -float(th_t.get(n, 0.0)))
        outputs.sort(key=lambda n: -float(th_out.get(n, 0.0)))
    else:
        inputs.sort(); techs.sort(); outputs.sort()

    # label wrapping
    if wrap_labels:
        label_map = {n: _wrap_label(n, wrap_max_chars) for n in (inputs + techs + outputs)}
    else:
        label_map = {n: n for n in (inputs + techs + outputs)}

    ordered = inputs + techs + outputs
    node_index = {name: i for i, name in enumerate(ordered)}

    # ---- COLORS FOR NODES ----
    color_map = ctx.get("color_map", {})  # technology -> hex
    node_colors = []
    for name in ordered:
        if name in techs:
            rgba = mcolors.to_rgba(color_map.get(name, "#4C78A8"), alpha=tech_color_alpha)
        elif name in inputs:
            rgba = mcolors.to_rgba("#BBBBBB", alpha=0.85)
        else:
            rgba = mcolors.to_rgba("#DDDDDD", alpha=0.85)
        node_colors.append(f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})")

    # ---- SPLIT LINKS IN TWO LAYERS ----
    ed = edges_df.copy()
    ed_in  = ed[ed["dst"].isin(techs)   & ed["src"].isin(inputs)].copy()   # input -> tech
    ed_out = ed[ed["src"].isin(techs)   & ed["dst"].isin(outputs)].copy()  # tech  -> output

    # thresholds
    ed_in  = ed_in[ed_in["weight"]  >= float(min_link_value_in)]
    ed_out = ed_out[ed_out["weight"] >= float(min_link_value_out)]

    # top-K per source (optional)
    if isinstance(top_links_per_source_in, int) and top_links_per_source_in > 0:
        ed_in = (ed_in.sort_values(["src", "weight"], ascending=[True, False])
                      .groupby("src", as_index=False).head(top_links_per_source_in))
    if isinstance(top_links_per_source_out, int) and top_links_per_source_out > 0:
        ed_out = (ed_out.sort_values(["src", "weight"], ascending=[True, False])
                       .groupby("src", as_index=False).head(top_links_per_source_out))

    # ---- GROUP 'OTHER' (heuristic on residuals) ----
    def _inject_other_node(name_other: str, side: str):
        """Ensure 'Other' node exists in the proper column list and update indexes."""
        nonlocal ordered, node_index
        if side == "out":
            if name_other not in outputs:
                outputs.append(name_other)
        else:  # "in"
            if name_other not in inputs:
                inputs.append(name_other)
        ordered = inputs + techs + outputs
        node_index = {name: i for i, name in enumerate(ordered)}

    def _group_others(df_links: pd.DataFrame, side: str):
        """Aggregate small links (below median per source) into an 'Other' sink node."""
        if df_links.empty:
            return df_links
        label_other = others_label_in if side == "in" else others_label_out
        _inject_other_node(label_other, side=side)

        grouped_parts = []
        for src, chunk in df_links.groupby("src"):
            if len(chunk) <= 2:
                grouped_parts.append(chunk)
                continue
            thr = float(chunk["weight"].median())
            keep = chunk[chunk["weight"] >= thr]
            other_val = float(chunk[chunk["weight"] < thr]["weight"].sum())
            if other_val > 0:
                grouped_parts.append(keep)
                grouped_parts.append(pd.DataFrame([{"src": src, "dst": label_other, "weight": other_val}]))
            else:
                grouped_parts.append(chunk)
        return pd.concat(grouped_parts, ignore_index=True)

    if group_others_in:
        ed_in = _group_others(ed_in, side="in")
    if group_others_out:
        ed_out = _group_others(ed_out, side="out")

    # ---- rebuild labels/colors after possible 'Other' injection ----
    ordered = inputs + techs + outputs
    labels = [label_map.get(n, n) for n in ordered]
    node_colors = []
    for name in ordered:
        if name in techs:
            rgba = mcolors.to_rgba(ctx.get("color_map", {}).get(name, "#4C78A8"), alpha=tech_color_alpha)
        elif name in inputs:
            rgba = mcolors.to_rgba("#BBBBBB", alpha=0.85)
        else:
            rgba = mcolors.to_rgba("#DDDDDD", alpha=0.85)
        node_colors.append(f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})")
    node_index = {name: i for i, name in enumerate(ordered)}  # rebuild

    # ---- build sankey arrays (concat the two layers) ----
    def _build_arrays(df_links: pd.DataFrame):
        src_idx = [node_index.get(s, None) for s in df_links["src"]]
        dst_idx = [node_index.get(t, None) for t in df_links["dst"]]
        mask = [i is not None and j is not None for i, j in zip(src_idx, dst_idx)]
        sources = [i for i, m in zip(src_idx, mask) if m]
        targets = [j for j, m in zip(dst_idx, mask) if m]
        values  = [float(w) for w, m in zip(df_links["weight"], mask) if m]
        return sources, targets, values

    s1, t1, v1 = _build_arrays(ed_in)
    s2, t2, v2 = _build_arrays(ed_out)
    sources = s1 + s2
    targets = t1 + t2
    values  = v1 + v2

    # ---- link colors ----
    link_colors = []
    if link_color_mode in ("source", "target"):
        cmap = ctx.get("color_map", {})
        for s, t in zip(sources, targets):
            node_name = ordered[s] if link_color_mode == "source" else ordered[t]
            if node_name in techs:  # vivid only for tech-origin/tech-target
                rgba = mcolors.to_rgba(cmap.get(node_name, "#4C78A8"), alpha=0.45)
            else:
                rgba = mcolors.to_rgba("#999999", alpha=0.25)
            link_colors.append(f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})")
    else:
        link_colors = [static_link_rgba] * len(sources)

    # ---- tooltips ----
    src_labels = [labels[s] for s in sources]
    dst_labels = [labels[t] for t in targets]
    customdata = np.vstack([src_labels, dst_labels, np.array(values, dtype=float)]).T
    hovertemplate = (
        "<b>%{customdata[0]}</b> → <b>%{customdata[1]}</b><br>"
        f"weight ({weight_name}): <b>%{{customdata[2]:.3g}}</b><extra></extra>"
    )

    # ---- optional role-based positioning ----
    sankey_node_kwargs = dict(
        pad=node_pad, thickness=node_thickness,
        line=dict(color="black", width=0.4),
        label=labels, color=node_colors
    )

    if use_role_positioning:
        # totals in/out per tech
        w_in  = (ed_in .groupby("dst")["weight"].sum(min_count=1) if not ed_in.empty else pd.Series(dtype=float)).to_dict()
        w_out = (ed_out.groupby("src")["weight"].sum(min_count=1) if not ed_out.empty else pd.Series(dtype=float)).to_dict()
        tech_role, tech_throughput = {}, {}
        for t in techs:
            tin  = float(w_in.get(t, 0.0))
            tout = float(w_out.get(t, 0.0))
            tot  = tin + tout
            s = (tout / tot) if tot > 0 else 0.5
            tech_role[t] = s
            tech_throughput[t] = tot

        def _even_y(names):
            n = max(1, len(names))
            y0, y1 = y_pad, 1.0 - y_pad
            return np.linspace(y0, y1, n) if n > 1 else np.array([(y0 + y1) / 2])

        # choose order within columns (throughput-desc if requested)
        if sort_nodes_by == "throughput":
            inputs_sorted  = sorted(inputs,  key=lambda n: -sum(v for s, v in zip(sources, values) if ordered[s] == n))
            techs_sorted   = sorted(techs,   key=lambda n: -tech_throughput.get(n, 0.0))
            outputs_sorted = sorted(outputs, key=lambda n: -sum(v for t, v in zip(targets, values) if ordered[t] == n))
        else:
            inputs_sorted, techs_sorted, outputs_sorted = inputs, techs, outputs

        x_arr = np.zeros(len(ordered), dtype=float)
        y_arr = np.zeros(len(ordered), dtype=float)

        # inputs
        in_ys = _even_y(inputs_sorted)
        for i, name in enumerate(inputs_sorted):
            idx = node_index[name]
            x_arr[idx] = x_input
            y_arr[idx] = in_ys[i]

        # techs
        tech_ys = _even_y(techs_sorted)
        for i, name in enumerate(techs_sorted):
            idx = node_index[name]
            role = tech_role.get(name, 0.5)
            # add small jitter proportional to index within similar-role group
            rng = np.random.default_rng(42)
            bins = np.array([0.0, 0.08, 0.25, 0.5, 0.75, 0.92, 1.0])
            bin_idx = np.digitize(role, bins) - 1
            span = tech_x_max - tech_x_min
            x_base = tech_x_min + bin_idx / (len(bins)-1) * span
            x_arr[idx] = x_base + 0.02 * (rng.random() - 0.5)  # leggero jitter locale

            y_arr[idx] = tech_ys[i]

        # outputs
        out_ys = _even_y(outputs_sorted)
        for i, name in enumerate(outputs_sorted):
            idx = node_index[name]
            x_arr[idx] = x_output
            y_arr[idx] = out_ys[i]

        sankey_node_kwargs.update(dict(x=x_arr, y=y_arr))
        arrangement_used = arrangement
    else:
        arrangement_used = "perpendicular"

    # ---- figure ----
    fig = go.Figure(data=[go.Sankey(
        node=sankey_node_kwargs,
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors, customdata=customdata, hovertemplate=hovertemplate
        ),
        arrangement=arrangement_used
    )])

    the_title = title or f"Flows (weight = {weight_name})"
    fig.update_layout(title_text=the_title, font=dict(size=font_size))

    # ---- save ----
    base_out = ctx.get("base_out_dir", "./out")
    out_dir = out_dir or base_out
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))
    out_html = os.path.join(out_dir, f"sankey_{_slugify(weight_name)}.html")
    fig.write_html(out_html)
    logger.info(f"[plot_flow_sankey] Saved HTML to {out_html}")
    ctx.setdefault("artifacts", []).append(out_html)

    if save_png:
        try:
            out_png = os.path.join(out_dir, f"sankey_{_slugify(weight_name)}.png")
            fig.write_image(out_png, scale=int(png_scale))
            logger.info(f"[plot_flow_sankey] Saved PNG to {out_png}")
            ctx["artifacts"].append(out_png)
        except Exception as e:
            logger.warning(f"[plot_flow_sankey] PNG export failed (install kaleido): {e}")

    return df, ctx





@step("build_carrier_graph")
def build_carrier_graph(
    df: pd.DataFrame,
    ctx: dict,
    input_col: str = "Input",
    output_col: str = "Output",
    token_separators: list = (";", ",", "|"),
    # weights: "count" (default), or take from stats: ("stats", col="CAPEX", metric="mean")
    weight_mode: str = "count",              # "count" | "stats"
    weight_stats_source_col: str = "CAPEX",
    weight_stats_metric: str = "count",      # often "count" or "mean"
    technology_col: str = "Technology name", # used only for stats-mode
    min_weight: float = 0.0,
    directed: bool = True,
    **_,
):
    """
    Build a carrier-to-carrier graph: InputCarrier -> OutputCarrier.
    Edge weight is count by default, or a stats value per technology accumulated across pairs.
    """
    if input_col not in df.columns or output_col not in df.columns:
        raise KeyError("Input/Output columns not found.")

    # explode multi-carrier cells
    rows = []
    for _, r in df[[input_col, output_col, technology_col]].iterrows():
        ins = _split_tokens(r[input_col], token_separators)
        outs = _split_tokens(r[output_col], token_separators)
        tech = r.get(technology_col, None)
        for i in ins:
            for o in outs:
                rows.append((i, o, tech))
    if not rows:
        raise ValueError("No carrier pairs built. Check separators/columns.")
    pairs = pd.DataFrame(rows, columns=["in_car", "out_car", "tech"])

    if weight_mode == "count":
        ed = pairs.groupby(["in_car", "out_car"], as_index=False).size().rename(columns={"size": "weight"})
        weight_name = "count"
    elif weight_mode == "stats":
        stats_all = ctx.get("stats", {})
        if weight_stats_source_col not in stats_all:
            raise KeyError(f"stats for '{weight_stats_source_col}' not in ctx. Run stats_by_technology first.")
        sdf = stats_all[weight_stats_source_col][[technology_col, weight_stats_metric]].rename(columns={weight_stats_metric: "w"})
        pairs = pairs.merge(sdf, left_on="tech", right_on=technology_col, how="left")
        ed = pairs.groupby(["in_car", "out_car"], as_index=False)["w"].sum(min_count=1).rename(columns={"w": "weight"})
        weight_name = f"{weight_stats_source_col}_{weight_stats_metric}"
    else:
        raise ValueError("weight_mode must be 'count' or 'stats'.")

    if min_weight > 0:
        ed = ed[ed["weight"] >= min_weight].reset_index(drop=True)

    nodes = sorted(set(ed["in_car"]).union(set(ed["out_car"])))
    nd = pd.DataFrame({"carrier": nodes})

    ctx["carrier_graph"] = {
        "edges_df": ed,
        "nodes_df": nd,
        "directed": directed,
        "weight_name": weight_name,
    }
    logger.info(f"[build_carrier_graph] carriers={len(nodes)}, edges={len(ed)}, weight={weight_name}")
    return df, ctx


@step("build_tech_similarity_graph")
def build_tech_similarity_graph(
    df: pd.DataFrame,
    ctx: dict,
    technology_col: str = "Technology name",
    sector_col: str = "Sector",
    input_col: str = "Input",
    output_col: str = "Output",
    token_separators: list = (";", ",", "|"),
    weight_mode: str = "count",   # "count" | "jaccard"
    min_weight: float = 1.0,
    **_,
):
    """
    Build a technology co-occurrence graph:
    - Two technologies are connected if they share at least one carrier (input or output).
    - Edge weight = number of shared carriers (or Jaccard).
    - Nodes colored by Sector (first non-null per technology).
    """
    if technology_col not in df.columns:
        raise KeyError("Technology column not found.")

    # tech -> set of carriers
    tech_carriers = {}
    tech_sector   = {}
    for _, r in df[[technology_col, sector_col, input_col, output_col]].iterrows():
        tech = r[technology_col]
        if pd.isna(tech): continue
        ins = _split_tokens(r.get(input_col, None), token_separators)
        outs= _split_tokens(r.get(output_col, None), token_separators)
        s = set(ins) | set(outs)
        tech_carriers.setdefault(tech, set()).update(s)
        # keep first seen sector if any
        if tech not in tech_sector and pd.notna(r.get(sector_col, None)):
            tech_sector[tech] = str(r[sector_col])

    techs = sorted(tech_carriers.keys())
    edges = []
    for i in range(len(techs)):
        for j in range(i+1, len(techs)):
            a, b = techs[i], techs[j]
            A, B = tech_carriers[a], tech_carriers[b]
            inter = len(A & B)
            if inter == 0: continue
            if weight_mode == "count":
                w = inter
            elif weight_mode == "jaccard":
                w = inter / max(1, len(A | B))
            else:
                raise ValueError("weight_mode must be 'count' or 'jaccard'")
            if w >= min_weight:
                edges.append((a, b, w))

    nodes_df = pd.DataFrame({"tech": techs, "sector": [tech_sector.get(t, "Unknown") for t in techs]})
    edges_df = pd.DataFrame(edges, columns=["src", "dst", "weight"])
    ctx["tech_graph"] = {"nodes_df": nodes_df, "edges_df": edges_df}
    logger.info(f"[build_tech_similarity_graph] techs={len(nodes_df)}, edges={len(edges_df)}")
    return df, ctx


@step("plot_interactive_force_pyvis")
def plot_interactive_force_pyvis(
    df: pd.DataFrame,
    ctx: dict,
    out_dir: str = None,

    # --- static spring layout (networkx) ---
    spring_k: float = None,
    spring_iterations: int = 300,
    spring_seed: int = 42,
    pos_scale_px: int = 450,
    center_px: tuple = (0, 0),

    # NEW: global scaling & spacing controls
    layout_scale: float = 1.4,          # multiply all distances after spring_layout
    min_node_distance_px: float = 48.0, # enforce minimum pairwise distance
    repel_iters: int = 8,               # iterations of pairwise repulsion
    repel_strength: float = 0.35,       # how hard to push per violation (0..1)

    # node sizing & styling
    node_size_metric: str = "strength",  # "strength" | "degree"
    node_size_min: int = 8,              # ↓ smaller default
    node_size_max: int = 28,             # ↓ smaller default
    node_font_size: int = 14,            # slightly smaller labels

    # edge styling
    edge_width_scale: float = 2.2,
    edge_color_rgba: str = "rgba(200,200,200,0.6)",

    # colors & canvas
    sector_palette: str = "tab20",
    bgcolor: str = "#111111",
    font_color: str = "#EEEEEE",
    canvas_width: str = "100%",
    canvas_height: str = "900px",
    **_,
):
    """
    Fixed, spherical (spring) layout → then scaled & repelled to avoid overlaps.
    """
    tg = ctx.get("tech_graph")
    if not tg:
        raise RuntimeError("tech_graph not found. Run 'build_tech_similarity_graph' first.")
    nd, ed = tg["nodes_df"].copy(), tg["edges_df"].copy()

    import numpy as np
    import json, math
    import networkx as nx
    import matplotlib.cm as cm, matplotlib.colors as mcolors
    from pyvis.network import Network
    import pandas as pd

    # -- colors
    sector_colors = ctx.get("sector_colors")
    if not sector_colors:
        sectors = sorted(nd["sector"].fillna("Unknown").unique().tolist())
        cmap = cm.get_cmap(sector_palette, len(sectors))
        sector_colors = {s: mcolors.to_hex(cmap(i)) for i, s in enumerate(sectors)}
        ctx["sector_colors"] = sector_colors

    # -- centralities
    if ed.empty:
        deg_src = pd.Series(dtype=float); deg_dst = pd.Series(dtype=float)
        str_src = pd.Series(dtype=float); str_dst = pd.Series(dtype=float)
    else:
        deg_src = ed.groupby("src")["weight"].size()
        deg_dst = ed.groupby("dst")["weight"].size()
        str_src = ed.groupby("src")["weight"].sum()
        str_dst = ed.groupby("dst")["weight"].sum()
    deg_map = (deg_src.add(deg_dst, fill_value=0)).to_dict()
    str_map = (str_src.add(str_dst, fill_value=0.0)).to_dict()

    vals = np.array([float((deg_map if node_size_metric.lower()=="degree" else str_map).get(t, 0.0))
                     for t in nd["tech"]], dtype=float)
    if vals.size and np.nanmax(vals) > np.nanmin(vals):
        ns = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
    else:
        ns = np.zeros_like(vals)
    node_sizes = node_size_min + ns * (node_size_max - node_size_min)
    size_map = dict(zip(nd["tech"], node_sizes))

    # -- build graph for layout
    G = nx.Graph()
    G.add_nodes_from(nd["tech"].tolist())
    for src, dst, w in ed.itertuples(index=False, name=None):
        G.add_edge(src, dst, weight=float(w))

    # -- spring
    pos = nx.spring_layout(G, k=spring_k, iterations=int(spring_iterations),
                           seed=int(spring_seed), weight="weight", dim=2) if len(G) else {}
    cx, cy = center_px

    # -- map to px + global scale
    coords = {n: (cx + float(x) * pos_scale_px * layout_scale,
                  cy + float(y) * pos_scale_px * layout_scale)
              for n, (x, y) in pos.items()}

    # -- post-layout repulsion to enforce min distance (physics remains OFF)
    if len(coords) > 1 and min_node_distance_px > 0 and repel_iters > 0:
        nodes = list(coords.keys())
        arr = np.array([coords[n] for n in nodes], dtype=float)

        def _repel_once(a: np.ndarray, dmin: float, k: float):
            # pairwise pushes for any pair closer than dmin
            n = a.shape[0]
            for i in range(n):
                for j in range(i+1, n):
                    dx = a[j,0] - a[i,0]
                    dy = a[j,1] - a[i,1]
                    dist = math.hypot(dx, dy)
                    if dist < 1e-9:
                        # split slightly in a random small direction
                        shift = dmin * 0.5
                        a[i,0] -= shift; a[j,0] += shift
                        continue
                    if dist < dmin:
                        # push apart proportionally to how much they violate dmin
                        push = (dmin - dist) * k
                        ux, uy = dx/dist, dy/dist
                        a[i,0] -= ux * push * 0.5
                        a[i,1] -= uy * push * 0.5
                        a[j,0] += ux * push * 0.5
                        a[j,1] += uy * push * 0.5
            return a

        for _ in range(int(repel_iters)):
            arr = _repel_once(arr, float(min_node_distance_px), float(repel_strength))
        coords = {n: tuple(arr[i]) for i, n in enumerate(nodes)}

    # -- edge widths
    if not ed.empty:
        w = ed["weight"].astype(float).values
        wmin, wmax = float(np.nanmin(w)), float(np.nanmax(w))
        widths = 1.0 + ((w - wmin) / (wmax - wmin) * edge_width_scale) if wmax > wmin else np.full_like(w, 1.0)
    else:
        widths = np.array([])

    # -- PyVis (physics OFF, nodes fixed)
    net = Network(height=canvas_height, width=canvas_width, directed=False,
                  notebook=False, bgcolor=bgcolor, font_color=font_color)
    options = {
        "nodes": {"shape":"dot","scaling":{"min":node_size_min,"max":node_size_max},
                  "font":{"size":node_font_size,"strokeWidth":2}},
        "edges": {"color":{"color":edge_color_rgba},"smooth":{"type":"dynamic"}},
        "physics": {"enabled": False},
        "interaction": {"hover": True, "zoomView": True, "dragNodes": False}
    }
    net.set_options(json.dumps(options))

    # nodes
    for tech, sector in nd[["tech","sector"]].itertuples(index=False, name=None):
        color = sector_colors.get(sector, "#888888")
        size_val = float(size_map.get(tech, node_size_min))
        x, y = coords.get(tech, (0.0, 0.0))
        title = (f"<b>{tech}</b><br>Sector: {sector}<br>"
                 f"Degree: {float(deg_map.get(tech, 0.0)):.0f}<br>"
                 f"Strength: {float(str_map.get(tech, 0.0)):.3g}")
        net.add_node(tech, label=tech, title=title, color=color, size=size_val,
                     x=float(x), y=float(y), fixed={"x": True, "y": True})

    # edges
    if not ed.empty:
        for i, (src, dst, wei) in enumerate(ed.itertuples(index=False, name=None)):
            net.add_edge(src, dst, value=float(wei), width=float(widths[i]),
                         title=f"{src} ↔ {dst}<br>weight: {float(wei):.3g}")

    # save
    base_out = ctx.get("base_out_dir", "./out")
    out_dir = out_dir or base_out
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))
    out_html = os.path.join(out_dir, "tech_similarity_pyvis.html")
    try:
        net.write_html(out_html, open_browser=False)
    except Exception as e:
        logger.error(f"[plot_interactive_force_pyvis] write_html failed: {e}")
        raise
    logger.info(f"[plot_interactive_force_pyvis] Saved {out_html}")
    ctx.setdefault("artifacts", []).append(out_html)
    return df, ctx




