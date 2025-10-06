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
    # I/O
    out_dir: str = None,
    # estetica archi
    edge_width_min: float = 0.6,
    edge_width_max: float = 6.0,
    alpha: float = 0.7,
    connectionstyle: str = "arc3,rad=0.06",
    # estetica nodi/testi
    tick_labelsize: int = 9,
    node_size: int = 420,
    # layout
    layout_mode: str = "sector_columns",   # "three_column" | "two_column" | "sector_columns"
    num_sector_cols: int = 6,              # quante colonne orizzontali dedicate ai settori
    x_pad: float = 0.10,                   # padding laterale (0..0.5)
    in_out_offset: float = 0.035,          # offset orizzontale delle sotto-colonne (in/mixed/out)
    sort_within_sector: str = "degree",    # "degree" | "alpha"
    jitter_y: float = 0.006,
    hide_self_loops: bool = False,
    sector_col: str = "Sector",
    input_col: str = "Input",
    output_col: str = "Output",
    technology_col: str = "Technology name",
    token_separators: list = (";", ",", "|"),
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

    # 6) posizioni
    pos = {}
    rng = np.random.default_rng(42)

    if layout_mode == "sector_columns":
        # mappa settore -> centro colonna su linspace
        sectors_sorted = sorted(set(carrier_sector.values()),
                                key=lambda s: -sum(_deg(c) for c in carriers if carrier_sector[c]==s))
        ncols = max(1, min(num_sector_cols, len(sectors_sorted)))
        xs_grid = np.linspace(x_pad, 1.0 - x_pad, ncols)

        sec2x = {}
        for idx, s in enumerate(sectors_sorted):
            sec2x[s] = xs_grid[idx % ncols]

        def _stack(names, x_center, offset):
            names = sorted(names, key=lambda c: -_deg(c)) if sort_within_sector=="degree" else sorted(names)
            n = max(1, len(names))
            ys = np.linspace(0.02, 0.98, n)
            if jitter_y and n > 1:
                ys = ys + rng.normal(0.0, jitter_y, size=n)
                ys = np.clip(ys, 0.01, 0.99)
            for i, name in enumerate(names):
                pos[name] = (float(x_center + offset), float(ys[i]))

        # per ciascun settore, distribuisci input/mixed/output con piccoli offset
        for s in sectors_sorted:
            x0 = sec2x[s]
            in_names   = [c for c in carriers if carrier_sector[c]==s and c in pure_inputs]
            mix_names  = [c for c in carriers if carrier_sector[c]==s and c in mixed]
            out_names  = [c for c in carriers if carrier_sector[c]==s and c in pure_outputs]
            _stack(in_names,  x0, -in_out_offset)
            _stack(mix_names, x0, 0.0)
            _stack(out_names, x0, +in_out_offset)

    else:
        # fallback: modalità precedenti
        def _order(names): 
            return sorted(names, key=lambda c: -_deg(c)) if sort_within_sector=="degree" else sorted(names)
        if layout_mode == "three_column":
            cols = [ (sorted(pure_inputs, key=lambda c:-_deg(c)), 0.0),
                     (sorted(mixed, key=lambda c:-_deg(c)), 0.5),
                     (sorted(pure_outputs, key=lambda c:-_deg(c)), 1.0) ]
        else:
            cols = [ (sorted(in_set, key=lambda c:-_deg(c)), 0.0),
                     (sorted(out_set, key=lambda c:-_deg(c)), 1.0) ]
        for names, x in cols:
            n = max(1, len(names))
            ys = np.linspace(0.02, 0.98, n)
            if jitter_y and n > 1:
                ys = ys + rng.normal(0.0, jitter_y, size=n)
                ys = np.clip(ys, 0.01, 0.99)
            for i, name in enumerate(names):
                pos[name] = (x, float(ys[i]))

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

    plt.figure(figsize=(18, 10))
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



@step("plot_flow_sankey")
def plot_flow_sankey(
    df: pd.DataFrame,
    ctx: dict,
    out_dir: str = None,
    min_link_value: float = 0.0,        # drop small links
    group_others: bool = True,          # group small links into "Other"
    others_label: str = "Other",
    top_links_per_source: int = None,   # keep only top-K per source node (after threshold)
    tech_color_alpha: float = 0.9,
    sort_nodes: bool = True,            # sort nodes alphabetically within each column
    **_,
):
    """
    Cleaner Sankey diagram (HTML) using flow_graph from build_flow_graph.
    Applies thresholding and optional 'Other' grouping.
    """
    fg = ctx.get("flow_graph")
    if not fg:
        raise RuntimeError("flow_graph not found in ctx. Run 'build_flow_graph' first.")

    nodes_df = fg["nodes_df"].copy()
    edges_df = fg["edges_df"].copy()
    weight_name = fg.get("weight_name", "weight")

    # split by type and sort
    if sort_nodes:
        inputs = sorted(nodes_df[nodes_df["type"]=="input"]["name"].tolist())
        techs  = sorted(nodes_df[nodes_df["type"]=="tech"]["name"].tolist())
        outputs= sorted(nodes_df[nodes_df["type"]=="output"]["name"].tolist())
    else:
        inputs = nodes_df[nodes_df["type"]=="input"]["name"].tolist()
        techs  = nodes_df[nodes_df["type"]=="tech"]["name"].tolist()
        outputs= nodes_df[nodes_df["type"]=="output"]["name"].tolist()

    ordered = inputs + techs + outputs
    node_index = {name: i for i, name in enumerate(ordered)}

    # threshold links
    ed = edges_df.copy()
    ed = ed[ed["weight"] >= float(min_link_value)]

    # top-K per source (optional)
    if top_links_per_source and top_links_per_source > 0:
        ed = (
            ed.sort_values(["src", "weight"], ascending=[True, False])
              .groupby("src", as_index=False)
              .head(top_links_per_source)
        )

    # group small links into "Other"
    if group_others:
        # detect which targets are missing after filtering and redirect them to "Other"
        # simpler: aggregate remaining small mass by src_type to one sink per column end
        # We'll only group on the "tech->output" side for clarity
        # (but you could extend to inputs similarly)
        pass  # keep simple; thresholding + top-K already cleans a lot

    # Labels + colors: techs keep their color_map
    labels = ordered
    color_map = ctx.get("color_map", {})
    node_colors = []
    for name in ordered:
        if name in techs:
            rgba = mcolors.to_rgba(color_map.get(name, "#4C78A8"), alpha=tech_color_alpha)
        elif name in inputs:
            rgba = mcolors.to_rgba("#BBBBBB", alpha=0.8)
        else:
            rgba = mcolors.to_rgba("#DDDDDD", alpha=0.8)
        node_colors.append(f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})")

    # build source/target/value arrays
    sources, targets, values = [], [], []
    for _, r in ed.iterrows():
        if r["src"] not in node_index or r["dst"] not in node_index:
            continue
        sources.append(node_index[r["src"]])
        targets.append(node_index[r["dst"]])
        values.append(float(r["weight"]))

    # figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=12, thickness=14, line=dict(color="black", width=0.5),
            label=labels, color=node_colors
        ),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(title_text=f"Flows (weight = {weight_name})", font=dict(size=12))

    base_out = ctx.get("base_out_dir", "./out")
    out_dir = out_dir or base_out
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))
    out_html = os.path.join(out_dir, f"sankey_clean_{_slugify(weight_name)}.html")
    fig.write_html(out_html)
    logger.info(f"[plot_flow_sankey_clean] Saved {out_html}")
    ctx.setdefault("artifacts", []).append(out_html)
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
    physics_solver: str = "forceAtlas2Based",   # "forceAtlas2Based" | "barnesHut" | "repulsion"
    sector_palette: str = "tab20",              # Matplotlib colormap for sectors
    node_size_base: int = 10,
    edge_width_scale: float = 2.0,              # multiply weight for line width
    **_,
):
    """
    Interactive force-directed graph (PyVis) of technology similarity.
    Nodes colored by Sector, edge width ∝ weight.
    """
    tg = ctx.get("tech_graph")
    if not tg:
        raise RuntimeError("tech_graph not found. Run 'build_tech_similarity_graph' first.")
    nd, ed = tg["nodes_df"], tg["edges_df"]

    # sector -> color
    import matplotlib.cm as cm, matplotlib.colors as mcolors
    sectors = sorted(nd["sector"].unique().tolist())
    cmap = cm.get_cmap(sector_palette, len(sectors))
    sector_colors = {s: mcolors.to_hex(cmap(i)) for i, s in enumerate(sectors)}

    net = Network(height="800px", width="100%", directed=False, notebook=False, bgcolor="#111111", font_color="#EEEEEE")
    net.toggle_physics(True)
    options = {
        "nodes": {
            "shape": "dot",
            "scaling": {"min": 5, "max": 40}
        },
        "edges": {
            "color": {"color": "#aaaaaa"},
            "smooth": {"type": "dynamic"}
        },
        "physics": {
            "solver": physics_solver,
            "stabilization": {"iterations": 150}
        }
    }
    import json
    net.set_options(json.dumps(options))

    # add nodes
    for _, r in nd.iterrows():
        tech, sector = r["tech"], r["sector"]
        color = sector_colors.get(sector, "#888888")
        net.add_node(tech, label=tech, title=f"{tech} | Sector: {sector}", color=color, size=node_size_base)

    # normalize edge widths for readability
    if not ed.empty:
        w = ed["weight"].astype(float).values
        wmin, wmax = float(np.min(w)), float(np.max(w))
        if wmax > wmin:
            widths = 1.0 + (w - wmin) / (wmax - wmin) * edge_width_scale
        else:
            widths = np.full_like(w, 1.0)
        for (src, dst, wei), lw in zip(ed.itertuples(index=False, name=None), widths):
            net.add_edge(src, dst, value=float(wei), width=float(lw))

    base_out = ctx.get("base_out_dir", "./out")
    out_dir = out_dir or base_out
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))
    out_html = os.path.join(out_dir, "tech_similarity_pyvis.html")

    # Scrivi l'HTML senza aprire il browser (più robusto di show())
    try:
        net.write_html(out_html, open_browser=False)
    except Exception as e:
        logger.error(f"[plot_interactive_force_pyvis] write_html failed: {e}")
        raise

    logger.info(f"[plot_interactive_force_pyvis] Saved {out_html}")
    ctx.setdefault("artifacts", []).append(out_html)
    return df, ctx

