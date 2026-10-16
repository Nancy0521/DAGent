"""Fixed, reference-free layout utility for generation."""
import math

VERSION = 'verified_candidate_quality'


def quality_score(layout, search, story, assets):
    body = [e.font_pt for e in layout.elements if e.kind == 'text']
    readability = min(1., min(body, default=0.) / 40.)
    figure_quality = []
    for module in story['modules']:
        for visual in module['visuals']:
            aid = visual['asset_id']
            element = layout.get(aid + '_image')
            asset = assets[aid]
            profile = asset.get('label_profile', {})
            target_pt = 24 if asset['original'].get('kind') == 'table' or module['role'] in ('results', 'analysis') else 20
            points = profile.get('points_per_inch')
            target_width = max(8, min(42, target_pt / points)) if points else profile.get('target_width_inches', 15)
            figure_quality.append(min(1., element.width / max(1., target_width)))
    # Use the actual occupied module area; planner visual_priority does NOT
    # affect this score (its search score is not comparable across preferences).
    regions = search['regions']
    area = sum(r['width'] * r['height'] for r in regions)
    unused = sum(r['unused_area'] for r in regions)
    occupancy = max(0., min(1., 1 - unused / max(area, 1.)))
    components = dict(body_readability=readability,
                      source_figure_scale=sum(figure_quality)/len(figure_quality) if figure_quality else 1.,
                      module_occupancy=occupancy)
    weights = dict(body_readability=.4, source_figure_scale=.4, module_occupancy=.2)
    score = sum(weights[k] * v for k, v in components.items())
    if not math.isfinite(score):
        raise ValueError('Nonfinite candidate quality score')
    return dict(policy=VERSION, score=score, components=components, weights=weights,
                eligible_only_after_all_hard_gates=True, tie_break='earlier_candidate',
                ground_truth_used=False, model_calls=0,
                source_label_sizes_are_proxies=True, semantic_quality_verified=False)
