#!/usr/bin/env python3
"""Dependency-free SVG helpers shared by the analysis notebooks."""

import html
import math
import statistics

NAVY = "#150079"
VIOLET = "#6748FD"
PEACH = "#FFBD9E"
INK = "#18152B"
MUTED = "#6E6A82"
GRID = "#E4E0F2"
PAPER = "#FBFAFF"


def _text(value):
    return html.escape(str(value))


def _header(width, height, title, subtitle=""):
    subtitle_svg = (
        f'<text x="54" y="58" font-size="15" fill="{MUTED}">{_text(subtitle)}</text>'
        if subtitle else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{_text(title)}">'
        f'<rect width="{width}" height="{height}" fill="{PAPER}"/>'
        f'<text x="54" y="34" font-family="Inter,Arial,sans-serif" font-size="23" '
        f'font-weight="700" fill="{NAVY}">{_text(title)}</text>{subtitle_svg}'
    )


def bar_chart(labels, values, title, subtitle="", value_format=None, width=900, height=470):
    value_format = value_format or (lambda value: f"{value:,.0f}")
    left, right, top, bottom = 210, 54, 88, 42
    chart_w, chart_h = width - left - right, height - top - bottom
    maximum = max(values) if values else 1
    row_h = chart_h / max(len(values), 1)
    parts = [_header(width, height, title, subtitle)]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * row_h + row_h * 0.18
        h = row_h * 0.60
        w = chart_w * value / maximum if maximum else 0
        fill = VIOLET if index % 2 == 0 else NAVY
        parts.append(f'<text x="{left-14}" y="{y+h*0.70:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="{INK}">{_text(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{fill}"/>')
        parts.append(f'<text x="{min(left+w+10,width-right):.1f}" y="{y+h*0.70:.1f}" font-family="Inter,Arial,sans-serif" font-size="14" fill="{INK}">{_text(value_format(value))}</text>')
    parts.append('</svg>')
    return "".join(parts)


def histogram(series, title, subtitle="", bins=16, x_min=None, x_max=None, width=900, height=470):
    all_values = [value for values in series.values() for value in values]
    x_min = min(all_values) if x_min is None else x_min
    x_max = max(all_values) if x_max is None else x_max
    span = x_max - x_min or 1
    counts = {}
    for label, values in series.items():
        bucket = [0] * bins
        for value in values:
            index = min(bins - 1, max(0, int((value - x_min) / span * bins)))
            bucket[index] += 1
        total = sum(bucket) or 1
        counts[label] = [count / total for count in bucket]
    left, right, top, bottom = 68, 36, 88, 58
    chart_w, chart_h = width - left - right, height - top - bottom
    maximum = max(value for values in counts.values() for value in values) or 1
    colors = [VIOLET, PEACH, NAVY]
    parts = [_header(width, height, title, subtitle)]
    for grid_index in range(5):
        y = top + chart_h * grid_index / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{GRID}"/>')
    group_w = chart_w / bins
    bar_w = group_w / max(len(series), 1) * 0.82
    for series_index, (label, values) in enumerate(counts.items()):
        color = colors[series_index % len(colors)]
        for index, value in enumerate(values):
            h = chart_h * value / maximum
            x = left + index * group_w + series_index * group_w / len(series)
            parts.append(f'<rect x="{x:.1f}" y="{top+chart_h-h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.82"/>')
        lx = left + series_index * 180
        parts.append(f'<rect x="{lx}" y="{height-25}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx+21}" y="{height-13}" font-family="Inter,Arial,sans-serif" font-size="13" fill="{INK}">{_text(label)}</text>')
    for index in range(5):
        value = x_min + span * index / 4
        x = left + chart_w * index / 4
        parts.append(f'<text x="{x:.1f}" y="{top+chart_h+23}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="{MUTED}">{value:.0f}</text>')
    parts.append('</svg>')
    return "".join(parts)


def scatter_chart(x_values, y_values, title, subtitle="", width=900, height=510):
    left, right, top, bottom = 78, 42, 88, 66
    chart_w, chart_h = width - left - right, height - top - bottom
    parts = [_header(width, height, title, subtitle)]
    for index in range(6):
        value = index * 20
        x = left + chart_w * index / 5
        y = top + chart_h * (1 - index / 5)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+chart_h}" stroke="{GRID}"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+chart_h+24}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="{MUTED}">{value}</text>')
        parts.append(f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="13" fill="{MUTED}">{value}</text>')
    parts.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top}" stroke="{PEACH}" stroke-width="3"/>')
    for x_value, y_value in zip(x_values, y_values):
        x = left + chart_w * max(0, min(100, x_value)) / 100
        y = top + chart_h * (1 - max(0, min(100, y_value)) / 100)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{VIOLET}" opacity="0.38"/>')
    parts.append(f'<text x="{left+chart_w/2:.1f}" y="{height-15}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="{INK}">Observed workload complexity</text>')
    parts.append(f'<text transform="translate(20 {top+chart_h/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="{INK}">Predicted workload complexity</text>')
    parts.append('</svg>')
    return "".join(parts)


def frontier_chart(rows, title="Held-out cost–quality frontier", width=940, height=540):
    costs = [float(row["policy_cost_usd_est"]) for row in rows]
    lows = [float(row["observed_quality"]) + float(row["quality_delta_ci_low"]) for row in rows]
    highs = [float(row["observed_quality"]) + float(row["quality_delta_ci_high"]) for row in rows]
    qualities = [float(row["policy_quality_est"]) for row in rows]
    x_min, x_max = min(costs) * 0.92, max(costs) * 1.04
    y_min, y_max = min(lows) - 0.01, max(highs) + 0.01
    left, right, top, bottom = 92, 48, 88, 72
    chart_w, chart_h = width - left - right, height - top - bottom
    sx = lambda value: left + (value - x_min) / (x_max - x_min) * chart_w
    sy = lambda value: top + (y_max - value) / (y_max - y_min) * chart_h
    parts = [_header(width, height, title, "198 template-held-out trajectories · lower cost and higher quality is better")]
    for index in range(5):
        yv = y_min + (y_max-y_min) * index / 4
        y = sy(yv)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="13" fill="{MUTED}">{yv:.2f}</text>')
    learned = [row for row in rows if row["policy_type"] == "learned"]
    learned = sorted(learned, key=lambda row: float(row["policy_cost_usd_est"]))
    if learned:
        points = " ".join(f'{sx(float(row["policy_cost_usd_est"])):.1f},{sy(float(row["policy_quality_est"])):.1f}' for row in learned)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{VIOLET}" stroke-width="3" opacity="0.7"/>')
    for row in rows:
        cost = float(row["policy_cost_usd_est"])
        quality = float(row["policy_quality_est"])
        low = float(row["observed_quality"]) + float(row["quality_delta_ci_low"])
        high = float(row["observed_quality"]) + float(row["quality_delta_ci_high"])
        x, y = sx(cost), sy(quality)
        selected = row["validation_selected"] == "1"
        if row["policy_type"] == "incumbent": color, radius = NAVY, 7
        elif row["policy_type"] == "heuristic": color, radius = PEACH, 7
        elif selected: color, radius = VIOLET, 10
        else: color, radius = VIOLET, 5
        if row["policy_type"] != "incumbent":
            parts.append(f'<line x1="{x:.1f}" y1="{sy(high):.1f}" x2="{x:.1f}" y2="{sy(low):.1f}" stroke="{color}" opacity="0.45"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="{PAPER}" stroke-width="2"/>')
        if row["policy_type"] in {"incumbent", "heuristic"} or selected:
            label = "Selected learned router" if selected else ("Logged route" if row["policy_type"] == "incumbent" else "15k heuristic")
            parts.append(f'<text x="{x+12:.1f}" y="{y-10:.1f}" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="700" fill="{color}">{_text(label)}</text>')
    for index in range(5):
        xv = x_min + (x_max-x_min) * index / 4
        x = sx(xv)
        parts.append(f'<text x="{x:.1f}" y="{top+chart_h+25}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="{MUTED}">${xv:.1f}</text>')
    parts.append(f'<text x="{left+chart_w/2:.1f}" y="{height-17}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="{INK}">Assumed opening-input cost</text>')
    parts.append(f'<text transform="translate(24 {top+chart_h/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="{INK}">Estimated outcome quality</text>')
    parts.append('</svg>')
    return "".join(parts)


def line_chart(labels, values, title, subtitle="", width=900, height=450):
    left, right, top, bottom = 78, 42, 88, 62
    chart_w, chart_h = width-left-right, height-top-bottom
    y_min, y_max = min(values)-0.003, max(values)+0.003
    sx = lambda index: left + chart_w * index / max(len(values)-1, 1)
    sy = lambda value: top + chart_h * (y_max-value) / (y_max-y_min or 1)
    parts = [_header(width, height, title, subtitle)]
    for index in range(5):
        yv = y_min + (y_max-y_min)*index/4
        y = sy(yv)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="13" fill="{MUTED}">{yv:+.3f}</text>')
    points = " ".join(f'{sx(i):.1f},{sy(v):.1f}' for i,v in enumerate(values))
    parts.append(f'<polyline points="{points}" fill="none" stroke="{VIOLET}" stroke-width="4"/>')
    for i,(label,value) in enumerate(zip(labels,values)):
        x,y=sx(i),sy(value)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{VIOLET}"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+chart_h+25}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="{MUTED}">{_text(label)}</text>')
    parts.append('</svg>')
    return "".join(parts)


def median_iqr_chart(groups, title, subtitle="", width=900, height=470):
    labels = list(groups)
    left, right, top, bottom = 170, 54, 88, 42
    chart_w, chart_h = width-left-right, height-top-bottom
    row_h = chart_h/max(len(labels),1)
    parts=[_header(width,height,title,subtitle)]
    for index,label in enumerate(labels):
        values=sorted(groups[label])
        q1=values[len(values)//4]; median=statistics.median(values); q3=values[(3*len(values))//4]
        y=top+index*row_h+row_h/2
        x1=left+chart_w*q1/100; xm=left+chart_w*median/100; x3=left+chart_w*q3/100
        parts.append(f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="{INK}">{_text(label)}</text>')
        parts.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x3:.1f}" y2="{y:.1f}" stroke="{PEACH}" stroke-width="14" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{xm:.1f}" cy="{y:.1f}" r="7" fill="{VIOLET}"/>')
        parts.append(f'<text x="{x3+12:.1f}" y="{y+5:.1f}" font-family="Inter,Arial,sans-serif" font-size="13" fill="{MUTED}">median {median:.0f}</text>')
    parts.append('</svg>')
    return "".join(parts)
