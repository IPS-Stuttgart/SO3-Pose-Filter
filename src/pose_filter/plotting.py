"""Small SVG plotting helpers to avoid a matplotlib dependency."""

from __future__ import annotations

from pathlib import Path


def _scale(
    value: float, src_min: float, src_max: float, dst_min: float, dst_max: float
) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) * 0.5
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def line_plot_svg(
    path: str | Path,
    series: dict[str, list[tuple[float, float]]],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 760,
    height: int = 420,
) -> None:
    """Write a simple multi-series SVG line plot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_points = [pt for points in series.values() for pt in points if pt[1] == pt[1]]
    if not all_points:
        path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8"
        )
        return
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(ys)
    pad_l, pad_r, pad_t, pad_b = 70, 24, 45, 58
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2:.1f}' y='24' text-anchor='middle' font-family='Arial' font-size='18'>{title}</text>",
        f"<line x1='{pad_l}' y1='{height-pad_b}' x2='{width-pad_r}' y2='{height-pad_b}' stroke='#333'/>",
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{height-pad_b}' stroke='#333'/>",
        f"<text x='{width/2:.1f}' y='{height-15}' text-anchor='middle' font-family='Arial' font-size='13'>{x_label}</text>",
        f"<text x='18' y='{height/2:.1f}' transform='rotate(-90 18 {height/2:.1f})' text-anchor='middle' font-family='Arial' font-size='13'>{y_label}</text>",
    ]
    for tick in range(5):
        frac = tick / 4.0
        y = _scale(frac, 0, 1, height - pad_b, pad_t)
        val = y_min + frac * (y_max - y_min)
        parts.append(
            f"<line x1='{pad_l-4}' y1='{y:.1f}' x2='{width-pad_r}' y2='{y:.1f}' stroke='#ddd'/>"
        )
        parts.append(
            f"<text x='{pad_l-8}' y='{y+4:.1f}' text-anchor='end' font-family='Arial' font-size='11'>{val:.1f}</text>"
        )
    for idx, (name, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        coords = []
        for x, y in points:
            if y != y:
                continue
            sx = _scale(x, x_min, x_max, pad_l, width - pad_r)
            sy = _scale(y, y_min, y_max, height - pad_b, pad_t)
            coords.append(f"{sx:.1f},{sy:.1f}")
        if coords:
            parts.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='2.2' points='{' '.join(coords)}'/>"
            )
            lx = width - 180
            ly = pad_t + 20 + idx * 20
            parts.append(
                f"<line x1='{lx}' y1='{ly}' x2='{lx+24}' y2='{ly}' stroke='{color}' stroke-width='2.2'/>"
            )
            parts.append(
                f"<text x='{lx+30}' y='{ly+4}' font-family='Arial' font-size='12'>{name}</text>"
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def robustness_plot(
    path: str | Path,
    rows: list[dict],
    metric: str = "filter_error_deg",
    title: str = "Filter Robustness",
    y_label: str = "filter error (deg)",
) -> None:
    """Plot a robustness metric by noise, grouped across occlusion levels."""
    series: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        key = f"occ={row['occlusion_prob']:.2f}"
        series.setdefault(key, []).append((float(row["noise_deg"]), float(row[metric])))
    for points in series.values():
        points.sort(key=lambda p: p[0])
    line_plot_svg(
        path,
        series,
        title=title,
        x_label="measurement noise (deg)",
        y_label=y_label,
    )


def _lerp_channel(start: int, end: int, fraction: float) -> int:
    return int(round(start + (end - start) * fraction))


def _color_lerp(start: str, end: str, fraction: float) -> str:
    fraction = min(1.0, max(0.0, fraction))
    start_rgb = tuple(int(start[idx : idx + 2], 16) for idx in (1, 3, 5))
    end_rgb = tuple(int(end[idx : idx + 2], 16) for idx in (1, 3, 5))
    rgb = tuple(
        _lerp_channel(start_channel, end_channel, fraction)
        for start_channel, end_channel in zip(start_rgb, end_rgb, strict=True)
    )
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def heatmap_svg(
    path: str | Path,
    rows: list[dict],
    x_key: str,
    y_key: str,
    value_key: str,
    title: str,
    x_label: str,
    y_label: str,
    value_label: str,
    width: int = 760,
    height: int = 470,
) -> None:
    """Write a simple SVG heatmap from dense metric rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [
        float(row[value_key])
        for row in rows
        if float(row[value_key]) == float(row[value_key])
    ]
    if not rows or not values:
        path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8"
        )
        return

    xs = sorted({float(row[x_key]) for row in rows})
    ys = sorted({float(row[y_key]) for row in rows}, reverse=True)
    lookup = {(float(row[x_key]), float(row[y_key])): float(row[value_key]) for row in rows}
    v_min, v_max = min(values), max(values)
    pad_l, pad_r, pad_t, pad_b = 95, 35, 58, 74
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    cell_w = plot_w / max(1, len(xs))
    cell_h = plot_h / max(1, len(ys))

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-family='Arial' font-size='18'>{title}</text>",
        f"<text x='{width/2:.1f}' y='{height-20}' text-anchor='middle' font-family='Arial' font-size='13'>{x_label}</text>",
        f"<text x='20' y='{height/2:.1f}' transform='rotate(-90 20 {height/2:.1f})' text-anchor='middle' font-family='Arial' font-size='13'>{y_label}</text>",
    ]
    for y_idx, y_value in enumerate(ys):
        y = pad_t + y_idx * cell_h
        parts.append(
            f"<text x='{pad_l-10}' y='{y+cell_h/2+4:.1f}' text-anchor='end' font-family='Arial' font-size='12'>{y_value:g}</text>"
        )
        for x_idx, x_value in enumerate(xs):
            x = pad_l + x_idx * cell_w
            value = lookup.get((x_value, y_value), float("nan"))
            if value == value:
                fraction = 0.0 if v_max <= v_min else (value - v_min) / (v_max - v_min)
                fill = _color_lerp("#edf8fb", "#2c7fb8", fraction)
                text_fill = "white" if fraction > 0.62 else "#111"
                label = f"{value:.1f}"
            else:
                fill = "#f2f2f2"
                text_fill = "#777"
                label = "nan"
            parts.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{cell_w:.1f}' height='{cell_h:.1f}' fill='{fill}' stroke='white' stroke-width='2'/>"
            )
            parts.append(
                f"<text x='{x+cell_w/2:.1f}' y='{y+cell_h/2+4:.1f}' text-anchor='middle' font-family='Arial' font-size='12' fill='{text_fill}'>{label}</text>"
            )
    for x_idx, x_value in enumerate(xs):
        x = pad_l + x_idx * cell_w
        parts.append(
            f"<text x='{x+cell_w/2:.1f}' y='{pad_t+plot_h+20:.1f}' text-anchor='middle' font-family='Arial' font-size='12'>{x_value:g}</text>"
        )
    parts.append(
        f"<text x='{width-pad_r-110}' y='{pad_t-17}' font-family='Arial' font-size='12'>{value_label}: {v_min:.1f}-{v_max:.1f}</text>"
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def trajectory_plot(path: str | Path, rows: list[dict]) -> None:
    series = {
        "observed": [(float(r["frame"]), float(r["observed_error_deg"])) for r in rows],
        "filter": [(float(r["frame"]), float(r["filter_error_deg"])) for r in rows],
        "filter observed": [
            (float(r["frame"]), float(r["filter_observed_joint_error_deg"])) for r in rows
        ],
        "filter occluded": [
            (float(r["frame"]), float(r["filter_occluded_joint_error_deg"])) for r in rows
        ],
    }
    if rows and "smoother_ema_error_deg" in rows[0]:
        series["ema"] = [
            (float(r["frame"]), float(r["smoother_ema_error_deg"])) for r in rows
        ]
    if rows and "smoother_chordal_error_deg" in rows[0]:
        series["chordal"] = [
            (float(r["frame"]), float(r["smoother_chordal_error_deg"])) for r in rows
        ]
    line_plot_svg(
        path,
        series,
        title="Trajectory Error Preview",
        x_label="frame",
        y_label="mean joint error (deg)",
    )
