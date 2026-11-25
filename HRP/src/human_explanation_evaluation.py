"""Offline CLI and notebook entry points for the Experiment 5 study."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from experiment2_analysis import read_checkpoint
from human_evaluation_analysis import VERSION, analyze, example_ids
from human_evaluation_figures import build_figures
from human_evaluation_rubric import calibration_session, load_annotations, primary_lock, repeat_lock, review_session
from human_evaluation_sampling import (
    CONDITIONS, SOURCES, load_source, prepare, read_packet, software_versions, source_hash, write_packet,
)
from llm_runner import _locked_directory, _save
from llm_verification import fingerprint
from modeling import DEFAULT_COHORT_ROOT

DEFAULT_DIRECTORY = DEFAULT_COHORT_ROOT / 'experiment5-private-v1'


def _link(annotations, crosswalk):
    return [{**crosswalk[key], **row} for key, row in annotations.items()]


def _annotation_state(directory, manifest, files):
    crosswalk = files['crosswalk.json']
    primary, primary_hash = load_annotations(directory, manifest, 'primary', crosswalk['primary'].keys())
    repeats, repeat_hash = ({}, None)
    if manifest['study'] == 'primary':
        repeats, repeat_hash = load_annotations(directory, manifest, 'repeat', crosswalk['repeat'].keys())
        repeat_lock(directory, manifest, repeat_hash, len(repeats))
    locked = primary_lock(directory, manifest, primary_hash, len(primary))
    if repeats and locked is None:
        raise ValueError('Repeat annotations require a locked primary pass.')
    return primary, repeats, {'primary': primary_hash, 'repeat': repeat_hash}, locked


def _progress(manifest, primary, repeats, locked):
    selected = manifest['primary_selected']
    return {
        'status': ('Awaiting human review' if len(primary) < selected or not locked else
                   'Primary target shortfall' if selected < manifest['primary_target'] else
                   'Primary review complete'),
        **{field: manifest[field] for field in ('study', 'reviewers', 'accounting', 'source_sha256',
            'primary_target', 'primary_selected', 'repeat_target', 'repeat_selected', 'excluded_source_outputs',
            'selected_by_condition', 'rubric_version', 'protocol_version')},
        'primary_completed': len(primary), 'primary_pending': selected - len(primary),
        'repeat_completed': len(repeats), 'repeat_pending': manifest['repeat_selected'] - len(repeats),
        'first_pass_locked': bool(locked),
    }


def notebook_run(mode='setup', *, directory=None, study='primary', checkpoint=None, write_worksheets=False):
    if mode == 'fictional':
        result, secondary = fictional_results()
        return {'status': 'Scripted fictional ratings for software validation', 'result': result, 'synthetic_result': secondary}
    if mode not in ('setup', 'review', 'analyze'):
        raise ValueError('Choose setup, review, analyze, or fictional.')
    if write_worksheets and mode != 'setup':
        raise ValueError('Worksheet creation requires setup mode.')
    directory = Path(directory or (DEFAULT_DIRECTORY if study == 'primary' else DEFAULT_DIRECTORY.with_name(
        'experiment5-synthetic-private-v1'))).expanduser()
    if mode in ('setup', 'analyze'):
        cases, hashes = load_source(study, checkpoint)
        manifest, files = prepare(cases, study, hashes)
    if write_worksheets:
        write_packet(directory, manifest, files)
    if not (directory / 'manifest.json').exists():
        if mode == 'setup':
            return _progress(manifest, {}, {}, None)
        return {'status': 'Awaiting human review', 'detail': 'No frozen worksheet packet is available.'}
    saved, packet = read_packet(directory)
    if saved['study'] != study or (mode in ('setup', 'analyze') and
            (saved['files'] != manifest['files'] or saved['source_sha256'] != manifest['source_sha256'])):
        raise ValueError('The saved packet does not match the frozen sources and selection.')
    primary, repeats, hashes, locked = _annotation_state(directory, saved, packet)
    progress = _progress(saved, primary, repeats, locked)
    if mode == 'review':
        aggregate_file = directory / 'results-v1' / 'aggregate.json'
        if aggregate_file.exists():
            aggregate, digest = read_checkpoint(aggregate_file)
            index, _ = read_checkpoint(aggregate_file.with_name('index.json'))
            if index['aggregate.json'] != digest or aggregate['provenance']['study_manifest_sha256'] != fingerprint(saved):
                raise ValueError('Saved aggregate checksum or provenance mismatch.')
            if aggregate['provenance']['annotation_sha256'] == hashes:
                progress['result'] = aggregate
            else:
                progress['aggregate_status'] = 'Saved aggregate refers to an earlier annotation snapshot.'
    if mode != 'analyze' or not locked:
        return progress
    crosswalk = packet['crosswalk.json']
    primary_rows = _link(primary, crosswalk['primary'])
    repeat_rows = _link(repeats, crosswalk.get('repeat', {}))
    result = analyze(primary_rows, repeat_rows, study=study, repeat_target=saved['repeat_target'],
                     repeat_selected=saved['repeat_selected'])
    result['provenance'] = {
        'study_manifest_sha256': fingerprint(saved), 'source_sha256': saved['source_sha256'],
        'crosswalk_sha256': saved['files']['crosswalk.json'], 'annotation_sha256': hashes,
        'rubric_sha256': saved['rubric_sha256'], 'protocol_sha256': saved['protocol_sha256'],
        'analysis_version': VERSION, 'software': software_versions(),
        'code_sha256': {name: source_hash(Path(__file__).with_name(name)) for name in (
            'human_evaluation_analysis.py', 'human_evaluation_figures.py', 'human_evaluation_rubric.py',
            'human_evaluation_sampling.py', Path(__file__).name)},
    }
    if study == 'primary':
        result['repeat_consistency']['days_since_primary_lock_at_first_repeat'] = _repeat_interval(directory, locked)
    return {**progress, 'result': result}


def _repeat_interval(directory, locked):
    path = directory / 'repeat-start.json'
    if not path.exists():
        return None
    record, _ = read_checkpoint(path)
    start = datetime.fromisoformat(record['started_at'])
    return round((start - locked).total_seconds() / 86400, 2)


def export_results(directory, result, *, version='results-v1'):
    """Freeze an aggregate snapshot, leaving optional case-level examples private."""
    if not version.startswith('results-') or Path(version).name != version:
        raise ValueError('Use a dedicated results-* version name.')
    directory = Path(directory).expanduser()
    with _locked_directory(directory) as directory:
        manifest, files = read_packet(directory)
        primary, repeats, hashes, locked = _annotation_state(directory, manifest, files)
        if (not locked or result['provenance']['annotation_sha256'] != hashes
                or result['provenance']['study_manifest_sha256'] != fingerprint(manifest)):
            raise ValueError('Annotations changed before export.')
        output = directory / version
        with _locked_directory(output) as output:
            aggregate = output / 'aggregate.json'
            index_file = output / 'index.json'
            if aggregate.exists() and read_checkpoint(aggregate)[0] != result:
                raise ValueError('The export version contains a different snapshot. Choose a new results-* version.')
            if index_file.exists():
                index, _ = read_checkpoint(index_file)
                if any(Path(name).name != name or source_hash(output / name) != digest for name, digest in index.items()):
                    raise ValueError('Saved aggregate bundle failed checksum validation.')
                return output
            _save(aggregate, result)
            index = {aggregate.name: source_hash(aggregate)}
            import matplotlib.pyplot as plt

            for name, figure in build_figures(result).items():
                path = output / f'{name}.png'
                with open(path, 'wb', opener=_owner_file) as file:
                    figure.savefig(file, format='png', dpi=180)
                plt.close(figure)
                index[path.name] = source_hash(path)
            links = files['crosswalk.json']
            _save(output / 'private-example-ids.json', example_ids(_link(primary, links['primary']),
                                                                   _link(repeats, links.get('repeat', {}))))
            _save(index_file, index)
            return output


def _owner_file(path, flags):
    return os.open(path, flags | os.O_NOFOLLOW, 0o600)


def fictional_results():
    """Scripted software fixtures with paired ratings, no source files or model calls."""
    def rows(study):
        return [{'case_id': f'{i}-{variant}', 'cluster': str(i), 'condition': condition, 'variant': variant,
                 'group': ('low', 'medium', 'high')[i % 3] if study == 'primary' else ('renal-utilization', 'age-physiology')[i % 2],
                 'source': SOURCES[i % 3] if condition == 'C' else None, 'review_id': f'F-{i}-{variant}-{condition}',
                 'mathematical_faithfulness': 'faithful' if (i + j) % 4 else 'unfaithful',
                 'completeness': ('complete', 'partial', 'insufficient')[(i + j) % 3],
                 'clarity_1_to_5': 1 + (i + j) % 5, 'readability_1_to_5': 1 + (i + 2 * j) % 5,
                 'uncertainty': 'appropriate', 'error_categories': [],
                 'automated': {'structured_correct': bool(i % 2), 'narrative': 'no_flag',
                               'release': 'verified_complete' if condition == 'C' else None}}
                for i in range(6) for variant in ((None,) if study == 'primary' else ('intuitive', 'counterintuitive'))
                for j, condition in enumerate(CONDITIONS)]

    primary = rows('primary')
    repeats = [{**row, 'review_id': f'FR-{i}', 'clarity_1_to_5': 1 + (row['clarity_1_to_5'] + i % 2 - 1) % 5}
               for i, row in enumerate(primary[:12])]
    return analyze(primary, repeats), analyze(rows('synthetic'), study='synthetic')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('calibrate', 'setup', 'review', 'annotate', 'analyze', 'fictional'))
    parser.add_argument('--study', choices=('primary', 'synthetic'), default='primary')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--write-worksheets', action='store_true')
    parser.add_argument('--pass', dest='pass_name', choices=('primary', 'repeat'), default='primary')
    parser.add_argument('--export', metavar='RESULTS_VERSION')
    args = parser.parse_args()
    directory = args.directory or (DEFAULT_DIRECTORY if args.study == 'primary' else DEFAULT_DIRECTORY.with_name(
        'experiment5-synthetic-private-v1'))
    if args.export and args.mode != 'analyze':
        parser.error('--export requires analyze mode')
    if args.mode == 'calibrate':
        calibration_session()
        return
    if args.mode == 'annotate':
        if args.study == 'synthetic' and args.pass_name == 'repeat':
            parser.error('Repeat review is available for the primary MIMIC study.')
        review_session(directory, args.pass_name)
        return
    response = notebook_run(args.mode, directory=directory, study=args.study, checkpoint=args.checkpoint,
                            write_worksheets=args.write_worksheets)
    if args.export and 'result' in response:
        response['export_directory'] = str(export_results(directory, response['result'], version=args.export))
    json.dump(response, sys.stdout, indent=2, allow_nan=False)
    print()


if __name__ == '__main__':
    main()
