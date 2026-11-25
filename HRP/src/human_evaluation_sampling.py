"""Build condition-blinded packets from frozen explanation checkpoints."""

from collections import Counter, defaultdict
from hashlib import sha256
import importlib.metadata
import os
from pathlib import Path
import random
import sys

from adversarial_runner import DEFAULT_ROOT, _saved_run
from adversarial_scenarios import FAMILIES, explanation_cases, validate_manifest
from experiment2_analysis import ORIGINAL_CHECKPOINT_SHA256, _validate_run, read_checkpoint
from experiment2_inputs import MODEL_RUN_ID, load_explanation_cases
from experiment2_scoring import CONDITIONS, score_experiment
from llm_runner import ModelConfig, _locked_directory, _save
from llm_verification import expected_drivers, fingerprint
from modeling import DEFAULT_COHORT_ROOT

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_CHECKPOINT = DEFAULT_COHORT_ROOT / 'checkpoint.json'
SYNTHETIC_CHECKPOINT = DEFAULT_ROOT / 'qwen3-v1' / 'checkpoint.json'
SYNTHETIC_SHA = '52ecf7c17a9cfd9108c264af7e8f0886a1754be23edfb261597deb1cf6052a1b'
SCENARIO_SHA = '68387030b1bd39b0bc243a81704624a0bc9c1f19bb2bac8edce09420693a7478'
SEEDS = {'primary_sample': 50380, 'synthetic_sample': 50381, 'primary_order': 50382,
         'repeat_sample': 50383, 'repeat_order': 50384, 'synthetic_order': 50385}
VERSION = 'experiment5-sampling-v1'
RUBRIC_VERSION = 'experiment5-rubric-v1'
PROTOCOL_VERSION = 'experiment5-protocol-v1'
RISKS = ('low', 'medium', 'high')
SOURCES = ('grounded_draft', 'model_revision', 'deterministic_fallback')
VARIANTS = ('intuitive', 'counterintuitive')
REVIEWER = 'reviewer-01'


def source_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def study_versions():
    return {
        'protocol_version': PROTOCOL_VERSION,
        'rubric_version': RUBRIC_VERSION,
        'rubric_sha256': source_hash(ROOT / 'experiment5/rubric.md'),
        'protocol_sha256': source_hash(ROOT / 'experiment5/protocol.md'),
    }


def software_versions():
    return {'python': sys.version.split()[0], **{package: importlib.metadata.version(package)
            for package in ('numpy', 'pandas', 'pyarrow', 'matplotlib', 'scikit-learn')}}


def _pinned_checkpoint(path, expected):
    state, digest = read_checkpoint(path)
    if digest != expected:
        raise ValueError('Frozen source checkpoint checksum mismatch.')
    return state, digest


def load_source(study, checkpoint=None):
    if study == 'primary':
        import pandas as pd

        state, digest = _pinned_checkpoint(checkpoint or PRIMARY_CHECKPOINT, ORIGINAL_CHECKPOINT_SHA256)
        provenance, cases = load_explanation_cases()
        score = _validate_run(state, provenance, cases, completed=True)
        cohort_root = Path(os.environ.get('MIMIC_OUTPUT_DIR', DEFAULT_COHORT_ROOT))
        model_root = Path(os.environ.get('HRP_MODEL_OUTPUT_DIR', cohort_root / 'model-runs')).expanduser()
        strata = pd.read_parquet(model_root / MODEL_RUN_ID / 'explanation_cases.parquet',
                                 columns=['risk_stratum'])['risk_stratum'].tolist()
        if Counter(strata) != dict.fromkeys(RISKS, 24):
            raise ValueError('The primary source must have 24 cases in each frozen risk stratum.')
        metadata = {case.case_id: {'cluster': case.case_id, 'group': stratum}
                    for case, stratum in zip(cases, strata, strict=True)}
        hashes = {'checkpoint': digest, 'model_manifest': provenance['manifest_sha256']}
    elif study == 'synthetic':
        path = Path(checkpoint or SYNTHETIC_CHECKPOINT).expanduser()
        _, digest = _pinned_checkpoint(path, SYNTHETIC_SHA)
        manifest, manifest_hash = read_checkpoint(path.with_name('scenario_manifest.json'))
        if fingerprint(manifest) != SCENARIO_SHA:
            raise ValueError('Frozen synthetic scenario checksum mismatch.')
        validate_manifest(manifest)
        state, validated_digest, _ = _saved_run(path.parent, manifest, 'real', ModelConfig(),
                                                 request_timeout=None, completed=True)
        if digest != validated_digest:
            raise ValueError('Synthetic checkpoint changed during validation.')
        cases = explanation_cases(manifest)
        score = score_experiment(cases, state)
        metadata = {row['case_id']: {'cluster': row['profile_id'], 'group': row['family'],
                                    'variant': row['variant']} for row in manifest['records']}
        hashes = {'checkpoint': digest, 'scenario_manifest': manifest_hash, 'scenario_canonical': SCENARIO_SHA}
    else:
        raise ValueError('Choose primary or synthetic.')
    return extract_cases(cases, metadata, state, score), hashes


def _response(record):
    if not record:
        return 'absent', None
    if (record.get('call', {}).get('error') or {}).get('code') == 'truncated_generation':
        return 'truncated', None
    if record['status'] != 'ok':
        return ('malformed' if record['status'] == 'parse_error' else record['status']), None
    trace = record.get('verification')
    answer = trace['answer'] if trace else (record.get('parse') or {}).get('answer')
    if not isinstance(answer, dict) or not isinstance(answer.get('explanation'), str) or not answer['explanation'].strip():
        return 'malformed', None
    return 'usable', answer


def extract_cases(cases, metadata, state, score):
    records = {(row['case_id'], row['condition']): row for row in state['records'].values()}
    scores = {(row['case_id'], row['condition']): row['quality'] for row in score['per_case']}
    result = {}
    for case in cases:
        outputs = {}
        for condition in CONDITIONS:
            key = (case.case_id, condition)
            record = records.get(key, {})
            status, answer = _response(record)
            trace = record.get('verification') or {}
            quality = scores.get(key) or {}
            outputs[condition] = {
                'status': status, 'answer': answer, 'source': trace.get('source'),
                'automated': {
                    'structured_correct': quality.get('structured_correct'),
                    'narrative': ('contradiction' if quality.get('narrative_contradictions') else
                                  'ambiguous' if quality.get('narrative_claims_unparsed') else
                                  'no_flag' if quality else None),
                    'release': trace.get('status'),
                },
            }
            if condition == 'C' and answer:
                if trace.get('source') not in SOURCES:
                    raise ValueError('Unexpected Condition C delivery source.')
                outputs[condition]['derivation'] = {
                    'grounded_draft': trace['original_raw'],
                    'model_revision': (trace.get('revision_call') or {}).get('raw'),
                    'deterministic_fallback': answer if trace['source'] == 'deterministic_fallback' else None,
                    'final_explanation': answer,
                }
        reference = {
            'predicted_probability': case.predicted_risk, 'feature_values': list(case.features),
            'reference_shap': case.shap,
            'reference_top_three': [{'feature': driver.feature, 'direction': driver.direction}
                                    for driver in expected_drivers(case)],
        }
        result[case.case_id] = {**metadata[case.case_id], 'reference': reference, 'outputs': outputs}
    return result


def _selection(cases, study):
    grouped = defaultdict(list)
    for case_id, case in sorted(cases.items()):
        grouped[case['group'], case['cluster']].append(case_id)
    expected_groups, target = (RISKS, 8) if study == 'primary' else (tuple(FAMILIES), 4)
    rng = random.Random(SEEDS[f'{study}_sample'])
    selected, accounting, exclusions = [], {}, []
    for group in expected_groups:
        eligible = []
        for (stratum, cluster), case_ids in grouped.items():
            if stratum != group:
                continue
            variant_counts = Counter(cases[key].get('variant') for key in case_ids)
            expected = {None: 1} if study == 'primary' else dict.fromkeys(VARIANTS, 1)
            if variant_counts != expected:
                raise ValueError('A source cluster has incorrect case or variant pairing.')
            failures = [{'case_id': key, 'condition': condition, 'status': output['status']}
                        for key in case_ids for condition, output in cases[key]['outputs'].items()
                        if output['status'] != 'usable']
            if failures:
                exclusions.extend(failures)
            else:
                eligible.append(case_ids)
        chosen = rng.sample(eligible, min(target, len(eligible)))
        selected.extend(key for cluster in chosen for key in cluster)
        accounting[group] = {'source_clusters': sum(key[0] == group for key in grouped),
                             'eligible_clusters': len(eligible), 'target_clusters': target,
                             'selected_clusters': len(chosen)}
    return selected, accounting, exclusions


def prepare(cases, study, hashes):
    if study not in ('primary', 'synthetic') or any(set(row['outputs']) != set(CONDITIONS) for row in cases.values()):
        raise ValueError('Expected a supported study with exactly three condition slots per case.')
    selected, accounting, exclusions = _selection(cases, study)
    items = [(key, condition) for key in selected for condition in CONDITIONS]
    repeat = []
    if study == 'primary':
        rng = random.Random(SEEDS['repeat_sample'])
        for condition in CONDITIONS:
            pool = [(key, condition) for key in selected]
            repeat.extend(rng.sample(pool, min(4, len(pool))))

    crosswalk, files = {}, {}
    for pass_name, pairs, seed_name, prefix in (
        ('primary', items, f'{study}_order', 'P' if study == 'primary' else 'S'),
        ('repeat', repeat, 'repeat_order', 'R'),
    ):
        if pass_name == 'repeat' and study != 'primary':
            continue
        random.Random(SEEDS[seed_name]).shuffle(pairs)
        links, worksheet, references = {}, [], {}
        for number, (key, condition) in enumerate(pairs, 1):
            review_id = f'{prefix}{number:03d}'
            case = cases[key]
            output = case['outputs'][condition]
            answer = output['answer']
            worksheet.append({'review_id': review_id, 'explanation_text': answer['explanation'],
                              'structured_drivers': answer.get('drivers', [])})
            references[review_id] = case['reference']
            links[review_id] = {'case_id': key, 'condition': condition, 'cluster': case['cluster'],
                                'group': case['group'], 'variant': case.get('variant'),
                                'source': output['source'], 'automated': output['automated'],
                                'derivation': output.get('derivation')}
        crosswalk[pass_name] = links
        files[f'{pass_name}-worksheet.json'] = worksheet
        files[f'{pass_name}-references.json'] = references
    files['crosswalk.json'] = crosswalk
    files['selection.json'] = {'selected_cases': selected, 'excluded_outputs': exclusions}
    expected_primary = 72 if study == 'primary' else 48
    manifest = {
        'study': study, 'sampling_version': VERSION, 'seeds': SEEDS, 'source_sha256': hashes,
        **study_versions(), 'reviewers': 1, 'software': software_versions(),
        'accounting': accounting, 'primary_target': expected_primary, 'primary_selected': len(items),
        'repeat_target': 12 if study == 'primary' else 0, 'repeat_selected': len(repeat),
        'excluded_source_outputs': dict(Counter(row['status'] for row in exclusions)),
        'selected_by_condition': dict.fromkeys(CONDITIONS, len(selected)),
        'files': {name: fingerprint(body) for name, body in files.items()},
    }
    return manifest, files


def write_packet(directory, manifest, files):
    """Write once; an interrupted setup can resume with the exact same inputs."""
    with _locked_directory(directory) as directory:
        marker = directory / 'manifest.json'
        if marker.exists():
            saved, _ = read_checkpoint(marker)
            if saved != manifest:
                raise ValueError('The directory already contains a different frozen study.')
            read_packet(directory)
            return
        if set(path.name for path in directory.iterdir()) - set(files) - {'.run.lock'}:
            raise ValueError('Choose a dedicated private study directory.')
        for name, body in files.items():
            path = directory / name
            if path.exists() and fingerprint(read_checkpoint(path)[0]) != fingerprint(body):
                raise ValueError('An existing private file does not match the selected study.')
            if not path.exists():
                _save(path, body)
        _save(marker, manifest)


def read_packet(directory, names=None):
    directory = Path(directory).expanduser()
    manifest, _ = read_checkpoint(directory / 'manifest.json')
    if ({key: manifest.get(key) for key in study_versions()} != study_versions()
            or manifest.get('sampling_version') != VERSION or manifest.get('reviewers') != 1):
        raise ValueError('Review packet version or protocol checksum mismatch.')
    files = {}
    for name in manifest['files'] if names is None else names:
        body, _ = read_checkpoint(directory / name)
        if fingerprint(body) != manifest['files'][name]:
            raise ValueError(f'Private packet checksum mismatch for {name}.')
        files[name] = body
    return manifest, files
