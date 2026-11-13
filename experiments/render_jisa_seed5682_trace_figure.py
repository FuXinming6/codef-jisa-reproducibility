"""Render the paper's seed-5682 mechanism figure from its archived JSON trace."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def render_figure(trace: dict[str, Any], output_path: Path) -> None:
    width, height = 1440, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular_path = r"C:\Windows\Fonts\arial.ttf"
    bold_path = r"C:\Windows\Fonts\arialbd.ttf"

    def font(size: int, bold: bool = False):
        path = bold_path if bold else regular_path
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default()

    navy = "#17364D"
    muted = "#5C6E79"
    blue = "#3179A5"
    green = "#2D7C69"
    orange = "#C47732"
    pale_blue = "#F2F7FA"
    pale_orange = "#FCF6EF"
    line = "#D7E1E7"
    left, right = trace["evaluations"]
    draw.text((55, 38), "HIERARCHICAL INSTANCE  |  SEED 5682", font=font(29, True), fill=blue)
    draw.text((55, 78), "Lower static risk does not ensure lower two-stage J", font=font(40, True), fill=navy)
    draw.text((55, 132), "Deterministic replay using the paper's graph generator and shared evaluator", font=font(27), fill=muted)
    draw.line((55, 177, width - 55, 177), fill=line, width=3)

    card_y0, card_y1 = 205, 575
    card_w = 640
    cards = [(55, left, pale_blue, blue), (745, right, pale_orange, orange)]
    for x, item, fill, accent in cards:
        draw.rounded_rectangle((x, card_y0, x + card_w, card_y1), radius=24, fill=fill, outline=line, width=3)
        title = "CoDef selected start" if item["label"].startswith("CoDef") else "Feasible +D01 extension"
        draw.text((x + 25, card_y0 + 24), title, font=font(32, True), fill=navy)
        draw.text((x + 25, card_y0 + 70), " + ".join(item["initial_actions"]), font=font(30, True), fill=accent)
        draw.text((x + card_w - 225, card_y0 + 76), f"initial cost = {item['initial_cost']:.0f}", font=font(23), fill=muted)

        metrics = [
            ("Static network risk", item["static_network_risk"], 1.0, blue, f"{item['static_network_risk']:.5f}"),
            ("Shared objective J", item["objective_J"], 0.25, green if x == 70 else orange, f"{item['objective_J']:.5f}"),
        ]
        for index, (label, value, maximum, color, value_text) in enumerate(metrics):
            y = card_y0 + 148 + index * 100
            draw.text((x + 25, y), label, font=font(25, True), fill=navy)
            bar_x0, bar_x1 = x + 260, x + card_w - 112
            bar_y0, bar_y1 = y + 9, y + 34
            draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=12, fill="#E2E9ED")
            fill_to = bar_x0 + round((bar_x1 - bar_x0) * min(1.0, value / maximum))
            draw.rounded_rectangle((bar_x0, bar_y0, fill_to, bar_y1), radius=12, fill=color)
            draw.text((bar_x1 + 10, y - 3), value_text, font=font(24, True), fill=navy)
            draw.text((bar_x0, y + 39), f"0 to {maximum:.2f}", font=font(20), fill=muted)

    static_drop = left["static_network_risk"] - right["static_network_risk"]
    objective_rise = right["objective_J"] - left["objective_J"]
    draw.text((55, 592), f"Static risk decreases by {static_drop:.5f}", font=font(23, True), fill=blue)
    draw.text((745, 592), f"J increases by {objective_rise:.5f}", font=font(23, True), fill=orange)
    draw.text((55, 640), "Profile-level foothold and selected response", font=font(29, True), fill=navy)

    x0, x1, x2 = 55, 300, 800
    top = 688
    draw.rounded_rectangle((55, top, 1385, top + 45), radius=8, fill="#EAF0F3")
    draw.text((x0 + 12, top + 8), "Profile", font=font(25, True), fill=navy)
    draw.text((x1 + 12, top + 8), "D02 + D08  |  foothold / response", font=font(25, True), fill=navy)
    draw.text((x2 + 12, top + 8), "D01 + D02 + D08  |  foothold / response", font=font(25, True), fill=navy)
    old_profiles = {item["profile"]: item for item in left["profiles"]}
    new_profiles = {item["profile"]: item for item in right["profiles"]}
    labels = {"risk_max": "Risk-max", "shortest_path": "Shortest-path", "asset_targeted": "Asset-targeted"}
    for index, profile in enumerate(("risk_max", "shortest_path", "asset_targeted")):
        y0 = top + 45 + index * 67
        if index % 2 == 0:
            draw.rectangle((55, y0, 1385, y0 + 67), fill="#FAFBFC")
        old, new = old_profiles[profile], new_profiles[profile]
        old_response = ", ".join(old["response_actions"]) if old["response_actions"] else "none"
        new_response = ", ".join(new["response_actions"]) if new["response_actions"] else "none"
        draw.text((x0 + 12, y0 + 15), labels[profile], font=font(26, True), fill=navy)
        draw.text((x1 + 12, y0 + 15), f"{old['foothold']}  /  {{{old_response}}}", font=font(26), fill=muted)
        draw.text((x2 + 12, y0 + 15), f"{new['foothold']}  /  {{{new_response}}}", font=font(26), fill=muted)
        draw.line((55, y0 + 66, 1385, y0 + 66), fill=line, width=1)

    footer = "J = 0.5 mean + 0.5 worst-profile risk + 0.2 sample SD across the three modeled profiles."
    draw.text((55, 952), footer, font=font(24), fill=muted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", dpi=(300, 300), optimize=True)
