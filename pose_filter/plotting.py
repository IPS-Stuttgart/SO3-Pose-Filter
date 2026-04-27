"""Small SVG plotting helpers to avoid a matplotlib dependency."""

from __future__ import annotations

from pathlib import Path


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
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
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
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
        parts.append(f"<line x1='{pad_l-4}' y1='{y:.1f}' x2='{width-pad_r}' y2='{y:.1f}' stroke='#ddd'/>")
        parts.append(f"<text x='{pad_l-8}' y='{y+4:.1f}' text-anchor='end' font-family='Arial' font-size='11'>{val:.1f}</text>")
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
            parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.2' points='{' '.join(coords)}'/>")
            lx = width - 180
            ly = pad_t + 20 + idx * 20
            parts.append(f"<line x1='{lx}' y1='{ly}' x2='{lx+24}' y2='{ly}' stroke='{color}' stroke-width='2.2'/>")
            parts.append(f"<text x='{lx+30}' y='{ly+4}' font-family='Arial' font-size='12'>{name}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def robustness_plot(path: str | Path, rows: list[dict]) -> None:
    """Plot filter error by noise, grouped across occlusion levels."""
    series: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        key = f"occ={row['occlusion_prob']:.2f}"
        series.setdefault(key, []).append((float(row["noise_deg"]), float(row["filter_error_deg"])))
    for points in series.values():
        points.sort(key=lambda p: p[0])
    line_plot_svg(
        path,
        series,
        title="Filter Robustness",
        x_label="measurement noise (deg)",
        y_label="filter error (deg)",
    )


def trajectory_plot(path: str | Path, rows: list[dict]) -> None:
    series = {
        "observed": [(float(r["frame"]), float(r["observed_error_deg"])) for r in rows],
        "filter": [(float(r["frame"]), float(r["filter_error_deg"])) for r in rows],
    }
    if rows and "smoother_ema_error_deg" in rows[0]:
        series["ema"] = [(float(r["frame"]), float(r["smoother_ema_error_deg"])) for r in rows]
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
