"""Aggregate matched ratings and quantify intra-reviewer repeat consistency."""

from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean, median, stdev

import numpy as np
from sklearn.metrics import cohen_kappa_score

from experiment2_scoring import CONDITIONS
from human_evaluation_rubric import COMPLETENESS, FAITHFULNESS, SCALES, UNCERTAINTY
from human_evaluation_sampling import RISKS, SOURCES, VARIANTS

VERSION = 'experiment5-analysis-v1'
BOOTSTRAP_SEED = 50386
REPLICATES = 10_000
FAITH = 'mathematical_faithfulness'
COMP = 'completeness'
COMPLETENESS_ORDER = COMPLETENESS[:3]
DETERMINATE = FAITHFULNESS[:2]
OUTCOMES = (FAITH, COMP, *SCALES)


def _distribution(values, categories):
    counts = Counter(values)
    return {str(label): counts[label] for label in categories}


def _describe(values):
    return {'n': len(values), 'mean': mean(values) if values else None,
            'median': median(values) if values else None, 'sd': stdev(values) if len(values) > 1 else None,
            'distribution': _distribution(values, range(1, 6))}


def _percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def _binary(label):
    return int(label == 'faithful') if label in DETERMINATE else None


def _bootstrap(clusters, replicates):
    """Resample numerator and denominator sums together at admission/profile level."""
    if replicates < 1:
        raise ValueError('Use at least one bootstrap replicate.')
    if sum(denominator > 0 for _, denominator in clusters) < 2:
        return None
    values = np.asarray(clusters, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(len(values), size=(replicates, len(values)))
    totals = values[samples].sum(axis=1)
    valid = totals[:, 1] > 0
    return np.percentile(totals[valid, 0] / totals[valid, 1], [2.5, 97.5]).tolist()


def _summarize(rows, replicates):
    faith = _distribution((row[FAITH] for row in rows), FAITHFULNESS)
    intervals = defaultdict(lambda: [0, 0])
    for row in rows:
        value = _binary(row[FAITH])
        totals = intervals[row['cluster']]
        if value is not None:
            totals[0] += value
            totals[1] += 1
    valid = sum(faith[key] for key in DETERMINATE)
    return {'n': len(rows), 'faithfulness': faith, 'determinate_n': valid,
            'determinate_faithful_rate': faith['faithful'] / valid if valid else None,
            'all_selected_faithful_rate': faith['faithful'] / len(rows) if rows else None,
            'determinate_faithful_ci95': _bootstrap(list(intervals.values()), replicates),
            'completeness': _distribution((row[COMP] for row in rows), COMPLETENESS),
            'uncertainty': _distribution((row['uncertainty'] for row in rows), UNCERTAINTY),
            'error_categories': dict(Counter(category for row in rows for category in row['error_categories'])),
            **{field: _describe([row[field] for row in rows if row[field] is not None]) for field in SCALES}}


def _ordered(label, labels):
    return labels.index(label) if label in labels else None


def _transition(left, right):
    return f'{left} → {right}'


def _pair_summary(pairs, field, transform, replicates):
    transitions = Counter(_transition(left[field], right[field]) for left, right in pairs)
    clusters = defaultdict(lambda: [0, 0])
    changes = []
    for left, right in pairs:
        totals = clusters[left['cluster']]
        a, b = transform(left[field]), transform(right[field])
        if a is not None and b is not None:
            difference = b - a
            changes.append(difference)
            totals[0] += difference
            totals[1] += 1
    return {'matched_n': len(pairs), 'eligible_n': len(changes), 'transitions': dict(transitions),
            'improved': sum(change > 0 for change in changes), 'worsened': sum(change < 0 for change in changes),
            'unchanged': changes.count(0), 'change_distribution': dict(Counter(map(str, changes))),
            'mean_change': mean(changes) if changes else None, 'median_change': median(changes) if changes else None,
            'mean_change_ci95': _bootstrap(list(clusters.values()), replicates)}


def _comparisons(pairs, replicates):
    transforms = (_binary, lambda value: _ordered(value, COMPLETENESS_ORDER), lambda value: value, lambda value: value)
    return {field: _pair_summary(pairs, field, transform, replicates)
            for field, transform in zip(OUTCOMES, transforms, strict=True)}


def _condition_pairs(rows, replicates):
    by_case = defaultdict(dict)
    for row in rows:
        cell = by_case[row['case_id']]
        if row['condition'] in cell:
            raise ValueError('Duplicate condition within a matched case.')
        cell[row['condition']] = row
    if any(set(cell) != set(CONDITIONS) for cell in by_case.values()):
        raise ValueError('Every analyzed case must retain its A/B/C triplet.')
    return {f'{right} − {left}': _comparisons([(cell[left], cell[right]) for cell in by_case.values()], replicates)
            for left, right in combinations(CONDITIONS, 2)}


def _variant_pairs(rows, replicates):
    profiles = defaultdict(dict)
    for row in rows:
        cells = profiles[row['cluster'], row['condition']]
        if row['variant'] in cells:
            raise ValueError('Duplicate model variant within a synthetic profile.')
        cells[row['variant']] = row
    if any(set(cell) != set(VARIANTS) for cell in profiles.values()):
        raise ValueError('Every analyzed synthetic profile must retain both model variants.')
    return {condition: _comparisons([(cell[VARIANTS[0]], cell[VARIANTS[1]])
                 for (_, arm), cell in profiles.items() if arm == condition], replicates) for condition in CONDITIONS}


def _human_verifier(rows):
    fields = ('structured_correct', 'narrative', 'release')
    transitions = {field: dict(Counter(_transition(row[FAITH], row['automated'][field]) for row in rows))
                   for field in fields}
    # A missing or ambiguous automated judgment has no binary counterpart.
    flags = {'structured_correct': {True: True, False: False},
             'narrative': {'no_flag': True, 'contradiction': False},
             'release': {'verified_complete': True}}
    discordance = {}
    for field, labels in flags.items():
        comparable = [(row[FAITH] == 'faithful', labels[row['automated'][field]]) for row in rows
                      if row[FAITH] in DETERMINATE and row['automated'][field] in labels]
        discordance[field] = {'comparable_n': len(comparable),
                              'human_faithful_automated_flagged': comparable.count((True, False)),
                              'human_unfaithful_automated_clear': comparable.count((False, True))}
    return {**transitions, 'discordance': discordance}


def _weighted_kappa(pairs, levels):
    eligible = [(a, b) for a, b in pairs if a in levels and b in levels]
    n = len(eligible)
    result = {'eligible_n': n, 'quadratic_weighted_cohens_kappa': None, 'reason': None}
    if n < 8:
        return {**result, 'reason': 'Fewer than eight eligible paired ratings.'}
    first, second = zip(*eligible, strict=True)
    if any(Counter(values).most_common(1)[0][1] / n >= 0.85 for values in (first, second)):
        return {**result, 'reason': 'A category accounts for at least 85% of ratings in one pass.'}
    value = cohen_kappa_score(first, second, labels=list(levels), weights='quadratic')
    return {**result, 'quadratic_weighted_cohens_kappa': float(value) if np.isfinite(value) else None,
            'reason': None if np.isfinite(value) else 'The statistic is undefined.'}


def repeat_consistency(primary, repeats, target=12, selected=12):
    """Compare new blind IDs to their original explanation after annotation is locked."""
    key = lambda row: (row['case_id'], row['condition'])
    originals = {key(row): row for row in primary}
    repeat_keys = [key(row) for row in repeats]
    if len(set(repeat_keys)) != len(repeats) or any(item not in originals for item in repeat_keys):
        raise ValueError('Repeat ratings must link uniquely to primary explanations.')
    matched = [(originals[key(row)], row) for row in repeats]
    result = {'label': 'Intra-reviewer repeat consistency', 'target': target, 'selected': selected,
              'completed': len(matched), 'missing_from_selected': selected - len(matched), 'shortfall_from_target': target - len(matched)}
    for field in (FAITH, COMP):
        pairs = [(first[field], second[field]) for first, second in matched]
        exact = sum(a == b for a, b in pairs)
        result[field] = {'n': len(pairs), 'exact_matches': exact, 'exact_agreement_percent': _percent(exact, len(pairs))}
        if field == COMP:
            result[field]['weighted_kappa'] = _weighted_kappa(pairs, COMPLETENESS_ORDER)
    differences_by_row = defaultdict(list)
    for field in SCALES:
        pairs = [(first[field], second[field]) for first, second in matched]
        differences = []
        for number, (a, b) in enumerate(pairs):
            if a is not None and b is not None:
                differences.append(abs(b - a))
                differences_by_row[number].append(abs(b - a))
        result[field] = {'n': len(differences), 'mean_absolute_difference': mean(differences) if differences else None,
                         'median_absolute_difference': median(differences) if differences else None,
                         'absolute_difference_distribution': _distribution(differences, range(5)),
                         'more_than_one_point_percent': _percent(sum(value > 1 for value in differences), len(differences)),
                         'weighted_kappa': _weighted_kappa(pairs, range(1, 6))}
    both = [values for values in differences_by_row.values() if len(values) == 2]
    result['either_scale_more_than_one_point'] = {
        'n_with_both_scales': len(both), 'percent': _percent(sum(max(values) > 1 for values in both), len(both))}
    return result


def analyze(primary, repeats=(), *, study='primary', repeat_target=12, repeat_selected=12, replicates=REPLICATES):
    conditions = {condition: [row for row in primary if row['condition'] == condition] for condition in CONDITIONS}
    result = {
        'analysis_version': VERSION, 'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_replicates': replicates,
        'study': study, 'explanations': len(primary), 'clusters': len({row['cluster'] for row in primary}),
        'by_condition': {condition: _summarize(rows, replicates) for condition, rows in conditions.items()},
        'paired': _condition_pairs(primary, replicates),
        'ratings_by_faithfulness': {condition: {label: {field: _describe([row[field] for row in rows
                  if row[FAITH] == label and row[field] is not None]) for field in SCALES} for label in DETERMINATE}
                  for condition, rows in conditions.items()},
        'human_vs_frozen_verifier': {condition: _human_verifier(rows) for condition, rows in conditions.items()},
        'condition_c_by_final_source': {source: _summarize([row for row in conditions['C'] if row['source'] == source], replicates)
                                       for source in SOURCES},
    }
    if study == 'primary':
        result['by_risk_group'] = {group: {condition: _summarize([row for row in rows if row['group'] == group], replicates)
                                          for condition, rows in conditions.items()} for group in RISKS}
        result['repeat_consistency'] = repeat_consistency(primary, repeats, repeat_target, repeat_selected)
    else:
        families = sorted({row['group'] for row in primary})
        result['synthetic_by_family_and_variant'] = {family: {variant: {condition: _summarize(
            [row for row in rows if row['group'] == family and row['variant'] == variant], replicates)
            for condition, rows in conditions.items()} for variant in VARIANTS} for family in families}
        result['variant_pairs'] = _variant_pairs(primary, replicates)
    return result


def example_ids(primary, repeats):
    """Pick at most one observed example per criterion, with private traceable blind IDs."""
    originals = {(row['case_id'], row['condition']): row for row in primary}
    candidates = {
        'readable_unfaithful_baseline': [row for row in primary if row['condition'] == 'A'
              and row[FAITH] == 'unfaithful' and (row['readability_1_to_5'] or 0) >= 4],
        'readable_faithful_grounded': [row for row in primary if row['condition'] == 'B'
              and row[FAITH] == 'faithful' and (row['readability_1_to_5'] or 0) >= 4],
        'correct_structured_drivers_and_flagged_prose': [row for row in primary if row['condition'] == 'B'
              and row[FAITH] == 'unfaithful' and row['automated']['structured_correct'] is True
              and row['automated']['narrative'] == 'contradiction'],
        'faithful_low_readability_fallback': [row for row in primary if row['source'] == 'deterministic_fallback'
              and row[FAITH] == 'faithful' and 0 < (row['readability_1_to_5'] or 0) <= 2],
        'counterintuitive_example': [row for row in primary if row['variant'] == 'counterintuitive'],
    }
    examples = {name: min((row['review_id'] for row in rows), default=None) for name, rows in candidates.items()}
    examples['repeat_judgment_changed'] = next(({'initial': first['review_id'], 'repeat': row['review_id']}
        for row in sorted(repeats, key=lambda item: item['review_id'])
        if (first := originals[row['case_id'], row['condition']]) and
        any(first[field] != row[field] for field in (FAITH, SCALES[0]))), None)
    return examples
