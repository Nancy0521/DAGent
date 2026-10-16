"""Find a missing native column header above a crop of middle table rows."""
import re
from statistics import median


def recover(rules, lines, old):
    candidates = []
    for top, divider in zip(rules, rules[1:]):
        if not top[2] < old.y0 or divider[2] >= old.y0:
            continue
        if abs(top[0] - divider[0]) > 3 or abs(top[1] - divider[1]) > 3:
            continue
        cells = [(box, text) for box, text in lines if text and top[2] < box.y0
                 and box.y1 < divider[2] and box.x0 >= top[0] - 2 and box.x1 <= top[1] + 2]
        if len(cells) < 3 or any(len(text) > 45 for _, text in cells):
            continue
        if max(box.y1 for box, _ in cells) - min(box.y0 for box, _ in cells) > 20:
            continue
        if sum(not bool(re.search(r'\d', text)) for _, text in cells) < len(cells) * .7:
            continue
        baseline = median(box.y1 for box, _ in cells)
        if any(abs(box.y1 - baseline) > 2.5 for box, _ in cells):
            continue
        if divider[2] - top[2] > 35:
            continue
        candidates.append(top)
    return candidates[0] if len(candidates) == 1 else None
