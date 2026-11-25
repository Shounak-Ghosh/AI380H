"""Validate single-reviewer annotations and run the two-stage local review."""

from datetime import datetime, timezone
import json
from pathlib import Path

from experiment2_analysis import read_checkpoint
from human_evaluation_sampling import REVIEWER, ROOT, read_packet
from llm_runner import _locked_directory, _save
from llm_verification import fingerprint

FAITHFULNESS = ('faithful', 'unfaithful', 'indeterminate', 'invalid')
COMPLETENESS = ('insufficient', 'partial', 'complete', 'not_applicable')
UNCERTAINTY = ('appropriate', 'overconfident', 'excessively_vague', 'not_applicable')
ERRORS = ('incorrect_direction', 'incorrect_rank', 'incorrect_probability', 'wrong_shap_scale',
          'invented_feature', 'unsupported_clinical_inference', 'unsupported_causal_claim',
          'prose_json_contradiction', 'other')
SCALES = ('clarity_1_to_5', 'readability_1_to_5')
LABELS = {'mathematical_faithfulness': FAITHFULNESS, 'completeness': COMPLETENESS, 'uncertainty': UNCERTAINTY}
FIELDS = {'review_id', 'reviewer_id', 'annotation_status', 'review_notes', 'error_categories', *SCALES, *LABELS}


def validate_annotation(row):
    if (not isinstance(row, dict) or row.keys() != FIELDS or row['reviewer_id'] != REVIEWER
            or row['annotation_status'] != 'completed' or not isinstance(row['review_id'], str)):
        raise ValueError('Incomplete annotation or unexpected reviewer, status, or fields.')
    invalid = row['mathematical_faithfulness'] == 'invalid'
    for field in SCALES:
        value = row[field]
        if not ((invalid and value is None) or (type(value) is int and 1 <= value <= 5)):
            raise ValueError(f'{field} must be an integer from 1 to 5, or blank for invalid text.')
    for field, labels in LABELS.items():
        if row[field] not in labels:
            raise ValueError(f'Unknown label for {field}.')
    if invalid and any(row[field] != 'not_applicable' for field in ('completeness', 'uncertainty')):
        raise ValueError('Invalid explanations require not_applicable completeness and uncertainty.')
    categories = row['error_categories']
    if (not isinstance(categories, list) or any(value not in ERRORS for value in categories)
            or len(set(categories)) != len(categories)):
        raise ValueError('Use distinct, recognized unsupported-claim categories.')
    notes = row['review_notes']
    if not isinstance(notes, str) or ((invalid or row['mathematical_faithfulness'] == 'unfaithful' or categories)
                                      and not notes.strip()):
        raise ValueError('Record brief evidence for invalid text, unfaithfulness, or unsupported claims.')


def load_annotations(directory, manifest, pass_name, ids):
    directory = Path(directory)
    filepath = directory / f'{pass_name}-annotations.json'
    if not filepath.exists():
        return {}, None
    body, digest = read_checkpoint(filepath)
    if (body.keys() != {'pass', 'worksheet_sha256', 'annotations'} or body['pass'] != pass_name
            or body['worksheet_sha256'] != manifest['files'][f'{pass_name}-worksheet.json']):
        raise ValueError('Annotations do not belong to the frozen worksheet.')
    indexed = {}
    for row in body['annotations']:
        validate_annotation(row)
        key = row['review_id']
        if key not in ids or key in indexed:
            raise ValueError('Unexpected or duplicated annotation ID.')
        indexed[key] = row
    return indexed, digest


def primary_lock(directory, manifest, primary_hash, count):
    path = Path(directory) / 'primary-lock.json'
    if not path.exists():
        return None
    body, _ = read_checkpoint(path)
    if (body['manifest_sha256'] != fingerprint(manifest) or body['annotations_sha256'] != primary_hash
            or count != manifest['primary_selected'] or not primary_hash):
        raise ValueError('Locked primary annotations failed checksum or completion validation.')
    return datetime.fromisoformat(body['completed_at'])


def repeat_lock(directory, manifest, repeat_hash, count):
    path = Path(directory) / 'repeat-lock.json'
    if not path.exists():
        return False
    body, _ = read_checkpoint(path)
    if (body != {'manifest_sha256': fingerprint(manifest), 'annotations_sha256': repeat_hash,
                 'completed': count} or count != manifest['repeat_selected'] or not repeat_hash):
        raise ValueError('Locked repeat annotations failed checksum or completion validation.')
    return True


def _prompt(field, labels):
    while True:
        answer = input(f'{field} ({" / ".join(map(str, labels))}): ').strip()
        if answer == ':quit':
            raise KeyboardInterrupt
        if answer in map(str, labels):
            return answer


def _rate(worksheet, reference):
    print(f'\n{worksheet["review_id"]}\n\n{worksheet["explanation_text"]}\n')
    text_scores = {
        scale: None if (answer := _prompt(scale, ('unusable', 1, 2, 3, 4, 5))) == 'unusable' else int(answer)
        for scale in SCALES
    }
    print('\nModel reference and structured drivers\n')
    print(json.dumps(reference, indent=2))
    print('\nExplanation’s structured drivers')
    print(json.dumps(worksheet['structured_drivers'], indent=2))
    labels = {field: _prompt(field, choices) for field, choices in LABELS.items()}
    print('\nUnsupported claim codes: ' + ', '.join(ERRORS))
    categories = [code.strip() for code in input('Codes, comma-separated (blank for none): ').split(',') if code.strip()]
    notes = input('Brief evidence notes (required for errors, invalid or unfaithful labels): ').strip()
    row = {'review_id': worksheet['review_id'], 'reviewer_id': REVIEWER, 'annotation_status': 'completed',
           **text_scores, **labels, 'error_categories': categories, 'review_notes': notes}
    validate_annotation(row)
    return row


def calibration_session():
    """Practice the live rating procedure with fictional public examples."""
    practice = json.loads((ROOT / 'experiment5/calibration.json').read_text(encoding='utf-8'))
    examples = practice['examples']
    print(f'Fictional calibration • {len(examples)} items. Practice answers are temporary.')
    for number, example in enumerate(examples, 1):
        item = {'review_id': f'Practice {number}', 'explanation_text': example['explanation_text'],
                'structured_drivers': []}
        while True:
            try:
                _rate(item, practice['reference'])
                break
            except (KeyboardInterrupt, EOFError):
                print('\nCalibration ended. Rerun the command to start again.')
                return
            except ValueError as error:
                print(f'{error} Try this practice item again.')
    print('\nCalibration complete. Compare your judgments with these rubric notes.')
    for number, example in enumerate(examples, 1):
        print(f'{number}. {example["rubric_feedback"]}')


def review_session(directory, pass_name):
    worksheet_name = f'{pass_name}-worksheet.json'
    refs_name = f'{pass_name}-references.json'
    with _locked_directory(directory) as directory:
        manifest, files = read_packet(directory, dict.fromkeys((worksheet_name, refs_name, 'primary-worksheet.json')))
        primary_ids = {row['review_id'] for row in files['primary-worksheet.json']}
        primary, primary_hash = load_annotations(directory, manifest, 'primary', primary_ids)
        completed_at = primary_lock(directory, manifest, primary_hash, len(primary))
        if pass_name == 'repeat':
            if completed_at is None:
                raise ValueError('Complete and lock the primary pass before beginning repeat review.')
            if (datetime.now(timezone.utc) - completed_at).total_seconds() < 3 * 86400:
                print('The protocol prefers at least three days between passes.')
        worksheet, references = files[worksheet_name], files[refs_name]
        ids = {row['review_id'] for row in worksheet}
        if len(ids) != len(worksheet) or ids != references.keys():
            raise ValueError('Worksheet IDs or reference linkage are invalid.')
        annotations, annotation_hash = load_annotations(directory, manifest, pass_name, ids)
        if ((pass_name == 'primary' and completed_at is not None)
                or (pass_name == 'repeat' and repeat_lock(directory, manifest, annotation_hash, len(annotations)))):
            print(f'{pass_name.capitalize()} pass already locked. {len(annotations)} ratings saved.')
            return
        if pass_name == 'repeat' and not (directory / 'repeat-start.json').exists():
            _save(directory / 'repeat-start.json', {'started_at': datetime.now(timezone.utc).isoformat()})
        print(f'{len(annotations)} of {len(worksheet)} saved. Enter :quit at a rating prompt or Ctrl+C to pause.')
        for item in worksheet:
            key = item['review_id']
            if key in annotations:
                continue
            try:
                annotation = _rate(item, references[key])
            except (KeyboardInterrupt, EOFError):
                print(f'\nPaused. {len(annotations)} ratings saved.')
                return
            except ValueError as error:
                print(f'Row not saved. {error} Restart the session to rate this row again.')
                return
            annotations[key] = annotation
            _save(directory / f'{pass_name}-annotations.json', {'pass': pass_name,
                  'worksheet_sha256': manifest['files'][worksheet_name], 'annotations': list(annotations.values())})
        _, digest = load_annotations(directory, manifest, pass_name, ids)
        if pass_name == 'primary' and digest:
            _save(directory / 'primary-lock.json', {'manifest_sha256': fingerprint(manifest),
                  'annotations_sha256': digest, 'completed_at': datetime.now(timezone.utc).isoformat()})
        elif digest:
            _save(directory / 'repeat-lock.json', {'manifest_sha256': fingerprint(manifest),
                  'annotations_sha256': digest, 'completed': len(annotations)})
        print(f'{pass_name.capitalize()} pass complete. {len(annotations)} ratings saved.')
