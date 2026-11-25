"""Offline checks for the Experiment 5 notebook, packets and paired analysis."""

from collections import Counter
import io
import json
from pathlib import Path
import stat
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from human_evaluation_analysis import COMP, FAITH, _bootstrap, _weighted_kappa, analyze, repeat_consistency
from human_evaluation_rubric import REVIEWER, load_annotations, review_session, validate_annotation
from human_evaluation_sampling import CONDITIONS, RISKS, _pinned_checkpoint, _response, prepare, read_packet, write_packet
from human_explanation_evaluation import export_results, notebook_run
from llm_runner import _save

SECRET = 'PRIVATE_MIMIC_PLACEHOLDER_TEXT'


def case_fixture(study='primary'):
    groups = RISKS if study == 'primary' else ('renal-utilization', 'age-physiology')
    cases = {}
    for group in groups:
        for number in range(10 if study == 'primary' else 6):
            cluster = f'{group}-private-{number}'
            for variant in ((None,) if study == 'primary' else ('intuitive', 'counterintuitive')):
                cases[f'{cluster}-{variant}'] = {'cluster': cluster, 'group': group, 'variant': variant,
                    'reference': {'predicted_probability': .8 if variant == 'counterintuitive' else .2},
                    'outputs': {condition: {'status': 'usable', 'source': 'deterministic_fallback' if condition == 'C' else None,
                                            'answer': {'explanation': SECRET, 'drivers': []},
                                            'automated': {'structured_correct': condition != 'A',
                                                          'narrative': 'no_flag', 'release': None}}
                                for condition in CONDITIONS}}
    return cases


def annotation(key='P001', **fields):
    return {'review_id': key, 'reviewer_id': REVIEWER, 'annotation_status': 'completed',
            FAITH: 'faithful', COMP: 'complete', 'clarity_1_to_5': 4, 'readability_1_to_5': 5,
            'uncertainty': 'appropriate', 'error_categories': [], 'review_notes': '', **fields}


class HumanSamplingTests(unittest.TestCase):
    def test_primary_design_is_deterministic_blind_and_paired(self):
        cases = case_fixture()
        manifest, files = prepare(cases, 'primary', {'checkpoint': 'fixture'})
        self.assertEqual((manifest, files), prepare(dict(reversed(list(cases.items()))), 'primary', {'checkpoint': 'fixture'}))
        crosswalk = files['crosswalk.json']
        keys = lambda rows: {(row['case_id'], row['condition']) for row in rows.values()}
        self.assertEqual(len(keys(crosswalk['primary'])), 72)
        self.assertEqual(keys(crosswalk['repeat']) & keys(crosswalk['primary']), keys(crosswalk['repeat']))
        self.assertEqual(Counter(row['condition'] for row in crosswalk['repeat'].values()), dict.fromkeys(CONDITIONS, 4))
        self.assertEqual(Counter(row['group'] for row in crosswalk['primary'].values()), dict.fromkeys(RISKS, 24))
        self.assertFalse(crosswalk['primary'].keys() & crosswalk['repeat'].keys())
        for name in ('primary', 'repeat'):
            worksheet = files[f'{name}-worksheet.json']
            self.assertEqual(set(worksheet[0]), {'review_id', 'explanation_text', 'structured_drivers'})
            self.assertNotIn('deterministic_fallback', json.dumps(worksheet))
            self.assertEqual({row['review_id'] for row in worksheet}, files[f'{name}-references.json'].keys())
            for columns in files[f'{name}-references.json'].values():
                self.assertEqual(set(columns), {'predicted_probability'})

    def test_synthetic_variants_keep_their_model_specific_references(self):
        manifest, files = prepare(case_fixture('synthetic'), 'synthetic', {})
        self.assertEqual(manifest['primary_selected'], 48)
        self.assertEqual(manifest['repeat_target'], 0)
        links = files['crosswalk.json']['primary']
        self.assertEqual(set(Counter(row['cluster'] for row in links.values()).values()), {6})
        for key, link in links.items():
            expected = .8 if link['variant'] == 'counterintuitive' else .2
            self.assertEqual(files['primary-references.json'][key]['predicted_probability'], expected)

    def test_failed_sources_are_accounted_and_smaller_strata_remain_explicit(self):
        cases = case_fixture()
        for index in range(3):
            cases[f'low-private-{index}-None']['outputs']['A']['status'] = 'malformed'
        manifest, files = prepare(cases, 'primary', {})
        self.assertEqual(manifest['primary_target'], 72)
        self.assertEqual(manifest['primary_selected'], 69)
        self.assertEqual(manifest['excluded_source_outputs'], {'malformed': 3})
        self.assertEqual(len(files['selection.json']['excluded_outputs']), 3)
        self.assertEqual(_response({}), ('absent', None))
        self.assertEqual(_response({'status': 'ok', 'parse': {'answer': {'explanation': ' '}}}), ('malformed', None))
        self.assertEqual(_response({'status': 'inference_error', 'call': {'error': {'code': 'truncated_generation'}}}),
                         ('truncated', None))
        delivered = {'explanation': 'final fallback', 'drivers': []}
        self.assertEqual(_response({'status': 'ok', 'verification': {'answer': delivered, 'original_raw': 'draft'}}),
                         ('usable', delivered))

    def test_private_files_and_hashes(self):
        manifest, files = prepare(case_fixture(), 'primary', {})
        with TemporaryDirectory() as directory:
            path = Path(directory)
            write_packet(path, manifest, files)
            write_packet(path, manifest, files)
            self.assertEqual(read_packet(path), (manifest, files))
            self.assertTrue(all(stat.S_IMODE(file.stat().st_mode) == 0o600 for file in path.iterdir()))
            with self.assertRaisesRegex(ValueError, 'Frozen source checkpoint checksum'):
                _pinned_checkpoint(path / 'manifest.json', 'incorrect')
            _save(path / 'crosswalk.json', {})
            with self.assertRaisesRegex(ValueError, 'Private packet checksum'):
                read_packet(path)


class HumanRubricTests(unittest.TestCase):
    def test_annotation_validation(self):
        validate_annotation(annotation())
        for fields in ({'clarity_1_to_5': True}, {'readability_1_to_5': 6}, {FAITH: 'unfaithful'},
                       {'annotation_status': 'pending'}, {'reviewer_id': 'other'}, {'error_categories': ['wrong']}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_annotation(annotation(**fields))

    def test_annotation_file_rejects_duplicates_and_wrong_worksheet(self):
        manifest, files = prepare(case_fixture(), 'primary', {})
        with TemporaryDirectory() as directory:
            path = Path(directory)
            write_packet(path, manifest, files)
            ids = files['crosswalk.json']['primary'].keys()
            self.assertEqual(load_annotations(path, manifest, 'primary', ids), ({}, None))
            body = {'pass': 'primary', 'worksheet_sha256': 'wrong', 'annotations': [annotation()]}
            filepath = path / 'primary-annotations.json'
            _save(filepath, body)
            with self.assertRaisesRegex(ValueError, 'frozen worksheet'):
                load_annotations(path, manifest, 'primary', ids)
            body['worksheet_sha256'] = manifest['files']['primary-worksheet.json']
            for rows in ([annotation(), annotation()], [annotation('R001')]):
                _save(filepath, {**body, 'annotations': rows})
                with self.assertRaisesRegex(ValueError, 'Unexpected or duplicated'):
                    load_annotations(path, manifest, 'primary', ids)

    def test_review_shows_reference_after_both_text_questions(self):
        from human_evaluation_rubric import _rate

        output = io.StringIO()
        responses = iter(('4', '5', 'faithful', 'complete', 'appropriate', '', ''))
        transcript = []
        def answer(prompt):
            transcript.append((prompt, output.getvalue()))
            return next(responses)
        with patch('sys.stdout', output), patch('builtins.input', side_effect=answer):
            _rate({'review_id': 'P001', 'explanation_text': 'test', 'structured_drivers': []}, {'reference_marker': 1})
        self.assertNotIn('reference_marker', transcript[0][1])
        self.assertNotIn('reference_marker', transcript[1][1])
        self.assertIn('reference_marker', transcript[2][1])

    def test_calibration_uses_live_prompts_without_accessing_the_private_packet(self):
        from human_evaluation_rubric import calibration_session

        responses = ('4', '5', 'faithful', 'complete', 'appropriate', '', '') * 6
        with patch('sys.stdout', new_callable=io.StringIO) as output, \
                patch('builtins.input', side_effect=responses) as prompt, \
                patch('human_evaluation_rubric.read_packet') as read_packet, \
                patch('human_evaluation_rubric._save') as save:
            calibration_session()
        self.assertEqual(prompt.call_count, 42)
        transcript = output.getvalue()
        self.assertLess(transcript.index('Practice 6'), transcript.index('Calibration complete'))
        self.assertIn('The explanation reverses that direction.', transcript)
        read_packet.assert_not_called()
        save.assert_not_called()


class HumanAnalysisTests(unittest.TestCase):
    def test_whole_cluster_bootstrap_and_repeat_agreement(self):
        self.assertEqual(_bootstrap([(1, 2), (0, 1)], 10_000), [0.0, 0.5])
        self.assertIsNone(_bootstrap([(3, 6), (0, 0)], 100))
        primary = [{**annotation(f'P{i}'), 'case_id': str(i), 'condition': 'A',
                    'clarity_1_to_5': 1 + i % 5, 'readability_1_to_5': 1 + i % 5} for i in range(12)]
        repeats = [{**row, 'review_id': f'R{i}'} for i, row in enumerate(primary)]
        repeats[0].update({'clarity_1_to_5': 3, FAITH: 'unfaithful', COMP: 'partial'})
        result = repeat_consistency(primary, repeats)
        self.assertEqual(result[FAITH]['exact_matches'], 11)
        self.assertEqual(result[COMP]['exact_matches'], 11)
        self.assertAlmostEqual(result['clarity_1_to_5']['more_than_one_point_percent'], 100/12)
        self.assertEqual(result['readability_1_to_5']['weighted_kappa']['quadratic_weighted_cohens_kappa'], 1.0)
        self.assertIsNone(result[COMP]['weighted_kappa']['quadratic_weighted_cohens_kappa'])
        self.assertIn('85%', _weighted_kappa([(1, 1)] * 12, range(1, 6))['reason'])
        self.assertEqual(repeat_consistency(primary, repeats[:4])['missing_from_selected'], 8)

    def test_paired_main_analysis_preserves_human_labels_and_excludes_repeats(self):
        _, files = prepare(case_fixture(), 'primary', {})
        link = files['crosswalk.json']['primary']
        rows = [{**row, **annotation(key, **{FAITH: 'unfaithful' if row['condition'] == 'A' else 'faithful'})}
                for key, row in link.items()]
        result = analyze(rows, rows[:12], replicates=100)
        faith = result['paired']['B − A'][FAITH]
        self.assertEqual((faith['matched_n'], faith['improved']), (24, 24))
        self.assertEqual(faith['mean_change_ci95'], [1.0, 1.0])
        self.assertEqual(result['explanations'], 72)
        self.assertEqual(result['by_condition']['A']['faithfulness']['unfaithful'], 24)
        self.assertEqual(result['condition_c_by_final_source']['deterministic_fallback']['n'], 24)
        discordance = result['human_vs_frozen_verifier']['A']['discordance']['narrative']
        self.assertEqual(discordance['human_unfaithful_automated_clear'], 24)
        self.assertEqual(result['repeat_consistency']['completed'], 12)
        self.assertNotIn('review_id', json.dumps(result))

    def test_workflow_analysis_and_export_offline(self):
        source = (case_fixture(), {'checkpoint': 'fictional-checkpoint'})
        with TemporaryDirectory() as directory, patch('human_explanation_evaluation.load_source', return_value=source), \
                patch('llm_runner.Ollama.discover', side_effect=AssertionError('Model calls are disabled')), \
                patch('human_evaluation_rubric._rate', side_effect=lambda row, _: annotation(row['review_id'])), \
                patch('sys.stdout', new_callable=io.StringIO):
            path = Path(directory)
            self.assertEqual(notebook_run(directory=path)['status'], 'Awaiting human review')
            self.assertFalse((path / 'manifest.json').exists())
            notebook_run(directory=path, write_worksheets=True)
            with self.assertRaisesRegex(ValueError, 'primary pass'):
                review_session(path, 'repeat')
            review_session(path, 'primary')
            review_session(path, 'repeat')
            run = notebook_run('analyze', directory=path)
            self.assertEqual((run['primary_completed'], run['repeat_completed']), (72, 12))
            output = export_results(path, run['result'])
            self.assertEqual(len(list(output.glob('*.png'))), 7)
            self.assertEqual(output, export_results(path, run['result']))
            self.assertEqual(notebook_run('review', directory=path)['result'], run['result'])
            aggregate = (output / 'aggregate.json').read_text()
            for private in (SECRET, '-private-', 'P001', 'review_notes'):
                self.assertNotIn(private, aggregate)
            data = json.loads((path / 'repeat-annotations.json').read_text())
            data['annotations'][0]['clarity_1_to_5'] = 1
            _save(path / 'repeat-annotations.json', data)
            with self.assertRaisesRegex(ValueError, 'Locked repeat annotations'):
                notebook_run('analyze', directory=path)
            self.assertIn('Scripted fictional', notebook_run('fictional')['status'])


if __name__ == '__main__':
    unittest.main()
