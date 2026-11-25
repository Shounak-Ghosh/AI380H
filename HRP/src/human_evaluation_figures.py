"""Report figures built exclusively from aggregate Experiment 5 results."""

from experiment2_scoring import CONDITIONS
from human_evaluation_analysis import COMP, FAITH
from human_evaluation_rubric import COMPLETENESS, SCALES
from human_evaluation_sampling import SOURCES

COLORS = ('#456A8A', '#168578', '#A75039')
SOURCE_LABELS = ('Grounded draft', 'Qwen3 revision', 'Deterministic fallback')


def _rate_plot(ax, groups):
    for index, (label, row) in enumerate(groups.items()):
        value, interval = row['determinate_faithful_rate'], row['determinate_faithful_ci95']
        if value is not None:
            ax.plot(index, 100 * value, 'o', color=COLORS[index % 3])
            if interval is not None:
                ax.vlines(index, 100 * interval[0], 100 * interval[1], color=COLORS[index % 3])
        ax.text(index, -8, f"n={row['determinate_n']}/{row['n']}", ha='center', fontsize=9)
    ax.set(xticks=list(range(len(groups))), xticklabels=list(groups), ylim=(-13, 105),
           yticks=range(0, 101, 20), ylabel='Faithful among determinate (%)')


def build_figures(result):
    if not result['explanations']:
        return {}
    import matplotlib.pyplot as plt
    import numpy as np

    figures = {}
    conditions = result['by_condition']
    figures['01_faithfulness'], ax = plt.subplots(layout='constrained', figsize=(6, 3.7))
    _rate_plot(ax, conditions)
    ax.set_title('Human-rated mathematical faithfulness')

    figures['02_text_ratings'], axes = plt.subplots(1, 2, layout='constrained', figsize=(9, 3.5), sharey=True)
    scores = np.arange(1, 6)
    for ax, field, label in zip(axes, SCALES, ('Clarity', 'Readability'), strict=True):
        for index, (condition, row) in enumerate(conditions.items()):
            ax.bar(scores + (index - 1) * .24, list(row[field]['distribution'].values()), width=.23,
                   label=condition, color=COLORS[index])
        ax.set(xticks=scores, xlabel=f'{label} rating')
    axes[0].set_ylabel('Explanations')
    axes[1].legend(frameon=False)

    figures['03_faithfulness_readability'], ax = plt.subplots(layout='constrained', figsize=(6, 4))
    for condition, color in zip(CONDITIONS, COLORS, strict=True):
        row = conditions[condition]
        x, y = row['determinate_faithful_rate'], row['readability_1_to_5']['mean']
        if x is not None and y is not None:
            ax.scatter(100 * x, y, label=condition, color=color)
    ax.set(xlim=(-3, 103), ylim=(1, 5.1), xlabel='Faithful among determinate (%)', ylabel='Mean readability')
    if ax.has_data():
        ax.legend(frameon=False)

    figures['04_matched_changes'], axes = plt.subplots(1, 2, layout='constrained', figsize=(9, 3.6))
    for ax, field, title in zip(axes, (FAITH, SCALES[1]), ('Faithfulness', 'Readability'), strict=True):
        for index, row in enumerate(result['paired'].values()):
            metric = row[field]
            if metric['mean_change'] is not None:
                ax.plot(metric['mean_change'], index, 'o', color=COLORS[index])
            if metric['mean_change_ci95'] is not None:
                ax.hlines(index, *metric['mean_change_ci95'], color=COLORS[index])
        ax.axvline(0, color='#8A9097', linewidth=1)
        ax.set(yticks=range(3), yticklabels=list(result['paired']), xlabel='Mean within-case change', title=title)

    figures['05_condition_c_sources'], axes = plt.subplots(1, 2, layout='constrained', figsize=(10, 4))
    sources = result['condition_c_by_final_source']
    for ax, field, levels, title in ((axes[0], SCALES[1], tuple(map(str, scores)), 'Readability'),
                                      (axes[1], COMP, COMPLETENESS, 'Completeness')):
        for index, source in enumerate(SOURCES):
            counts = sources[source][field]
            if field == SCALES[1]:
                counts = counts['distribution']
            ax.bar(np.arange(len(levels)) + (index - 1) * .24, [counts[level] for level in levels],
                   width=.23, label=f'{SOURCE_LABELS[index]} (n={sources[source]["n"]})', color=COLORS[index])
        ax.set(xticks=range(len(levels)), xticklabels=[level.replace('_', ' ').capitalize() for level in levels],
               title=title, ylabel='Explanations')
        ax.tick_params(axis='x', rotation=15)
    axes[0].legend(frameon=False, fontsize=8)

    figures['06_human_vs_verifier'], axes = plt.subplots(1, 3, layout='constrained', figsize=(10, 3.6), sharey=True)
    human = ('faithful', 'unfaithful', 'indeterminate', 'invalid')
    automated = ('contradiction', 'ambiguous', 'no_flag')
    for ax, condition in zip(axes, CONDITIONS, strict=True):
        counts = result['human_vs_frozen_verifier'][condition]['narrative']
        matrix = [[counts.get(f'{label} → {flag}', 0) for flag in automated] for label in human]
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(1, conditions[condition]['n']))
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center',
                    color='white' if value > conditions[condition]['n'] * .55 else '#17212B')
        ax.set(title=f'Condition {condition}', xticks=range(3), yticks=range(4), yticklabels=human,
               xticklabels=['Contradiction', 'Ambiguous', 'No flag'])
        ax.tick_params(axis='x', rotation=25)
    axes[0].set_ylabel('Human faithfulness label')

    consistency = result.get('repeat_consistency')
    if consistency and consistency['completed']:
        figures['07_repeat_consistency'], ax = plt.subplots(layout='constrained', figsize=(6.2, 3.6))
        differences = np.arange(5)
        for index, (field, label) in enumerate(zip(SCALES, ('Clarity', 'Readability'), strict=True)):
            metric = consistency[field]
            ax.bar(differences + (index - .5) * .35, list(metric['absolute_difference_distribution'].values()),
                   width=.34, color=COLORS[index], label=f'{label} (n={metric["n"]})')
        ax.set(xticks=differences, xlabel='Absolute rating difference', ylabel='Repeat pairs',
               title='Intra-reviewer repeat consistency')
        ax.legend(frameon=False)
    if result['study'] == 'synthetic':
        figures['08_synthetic_comparison'], axes = plt.subplots(1, 3, figsize=(11, 3.4), layout='constrained')
        for ax, condition in zip(axes, CONDITIONS, strict=True):
            groups = {f'{family}\n{variant}': cells[condition]
                      for family, variants in result['synthetic_by_family_and_variant'].items()
                      for variant, cells in variants.items()}
            _rate_plot(ax, groups)
            ax.set_title(f'Condition {condition}')
            ax.tick_params(axis='x', labelsize=7)
    return figures
