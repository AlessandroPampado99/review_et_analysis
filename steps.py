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


@step("plot_flow_networkx")
def plot_flow_networkx(
    df: pd.DataFrame,
    ctx: dict,
    out_dir: str = None,                        # default: base_out_dir
    node_size: int = 300,
    edge_width_min: float = 0.6,      # minimum visible width
    edge_width_max: float = 6.0,      # maximum width
    edge_width_scale: float = 0.002,            # multiply weight by this for width
    input_x: float = 0.0, tech_x: float = 0.5, output_x: float = 1.0,
    tick_labelsize: int = 9,
    **_,
):
    """
    Plot a static 3-part network (input | tech | output) with edge thickness ~ weight.
    Technology nodes colored via ctx['color_map']; inputs/outputs neutral.
    """
    fg = ctx.get("flow_graph")
    if not fg:
        raise RuntimeError("flow_graph not found in ctx. Run 'build_flow_graph' first.")
    nodes_df = fg["nodes_df"]
    edges_df = fg["edges_df"]
    weight_name = fg.get("weight_name", "weight")

    base_out_dir = ctx.get("base_out_dir", "./out")
    out_dir = out_dir or base_out_dir
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))
    out_fp = os.path.join(out_dir, f"network_{_slugify(weight_name)}.png")

    # Build graph
    G = nx.DiGraph()
    for _, r in nodes_df.iterrows():
        G.add_node(r["name"], kind=r["type"])

    for _, r in edges_df.iterrows():
        G.add_edge(r["src"], r["dst"], weight=float(r["weight"]))

    # Positions: stack inputs left, techs center, outputs right
    pos = {}
    color_map = ctx.get("color_map", {})
    # y positions: spread evenly for each column
    def _stack_positions(names, x_val):
        n = max(len(names), 1)
        ys = np.linspace(0, 1, n)
        for i, name in enumerate(names):
            pos[name] = (x_val, ys[i])

    inputs = nodes_df[nodes_df["type"] == "input"]["name"].tolist()
    techs  = nodes_df[nodes_df["type"] == "tech"]["name"].tolist()
    outputs= nodes_df[nodes_df["type"] == "output"]["name"].tolist()

    _stack_positions(inputs, input_x)
    _stack_positions(techs, tech_x)
    _stack_positions(outputs, output_x)

    # node colors
    node_colors = []
    for n in G.nodes():
        kind = G.nodes[n]["kind"]
        if kind == "tech":
            node_colors.append(color_map.get(n, "#4C78A8"))
        elif kind == "input":
            node_colors.append("#BBBBBB")
        else:  # output
            node_colors.append("#DDDDDD")

    plt.figure(figsize=(14, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, font_size=tick_labelsize)

    # edges with width scaled by weight
    raw_w = np.array([float(G[u][v]["weight"]) for u, v in G.edges()])
    if raw_w.size:
        w_min, w_max = float(np.nanmin(raw_w)), float(np.nanmax(raw_w))
        if w_max > w_min:
            widths = edge_width_min + (raw_w - w_min) * (edge_width_max - edge_width_min) / (w_max - w_min)
        else:
            widths = np.full_like(raw_w, (edge_width_min + edge_width_max) / 2.0)
    else:
        widths = []
    
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowsize=10, alpha=0.7)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=200)
    plt.close()

    logger.info(f"[plot_flow_networkx] Saved {out_fp}")
    ctx.setdefault("artifacts", []).append(out_fp)
    return df, ctx


@step("plot_flow_sankey")
def plot_flow_sankey(
    df: pd.DataFrame,
    ctx: dict,
    out_dir: str = None,                    # default: base_out_dir
    tech_color_alpha: float = 0.9,          # alpha for tech node colors
    save_png: bool = False,                 # requires orca/kaleido if True
    **_,
):
    """
    Plot an interactive Sankey diagram based on the built flow graph.
    - Saves HTML (always), and optionally PNG.
    - Technology nodes colored via ctx['color_map']; inputs/outputs neutral.
    """
    fg = ctx.get("flow_graph")
    if not fg:
        raise RuntimeError("flow_graph not found in ctx. Run 'build_flow_graph' first.")

    nodes_df = fg["nodes_df"]
    edges_df = fg["edges_df"]
    node_index = fg["node_index"]
    weight_name = fg.get("weight_name", "weight")

    base_out_dir = ctx.get("base_out_dir", "./out")
    out_dir = out_dir or base_out_dir
    _ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))
    out_html = os.path.join(out_dir, f"sankey_{_slugify(weight_name)}.html")
    out_png  = os.path.join(out_dir, f"sankey_{_slugify(weight_name)}.png")

    labels = nodes_df["name"].tolist()
    # node colors
    color_map = ctx.get("color_map", {})
    node_colors = []
    for _, r in nodes_df.iterrows():
        if r["type"] == "tech":
            c = mcolors.to_rgba(color_map.get(r["name"], "#4C78A8"), alpha=tech_color_alpha)
        elif r["type"] == "input":
            c = mcolors.to_rgba("#BBBBBB", alpha=0.8)
        else:
            c = mcolors.to_rgba("#DDDDDD", alpha=0.8)
        # plotly wants rgb/rgba strings
        node_colors.append(f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]:.2f})")

    # edges
    sources = []
    targets = []
    values  = []
    for _, r in edges_df.iterrows():
        s = node_index[r["src"]]
        t = node_index[r["dst"]]
        v = float(r["weight"])
        sources.append(s)
        targets.append(t)
        values.append(v)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=12,
            thickness=14,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(
        title_text=f"Flows (weight = {weight_name})",
        font=dict(size=12)
    )

    fig.write_html(out_html)
    logger.info(f"[plot_flow_sankey] Saved HTML to {out_html}")
    ctx.setdefault("artifacts", []).append(out_html)

    if save_png:
        try:
            fig.write_image(out_png, scale=2)
            logger.info(f"[plot_flow_sankey] Saved PNG to {out_png}")
            ctx["artifacts"].append(out_png)
        except Exception as e:
            logger.warning(f"[plot_flow_sankey] PNG export failed (install kaleido): {e}")

    return df, ctx


