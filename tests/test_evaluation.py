"""Offline regression checks for the released Judge/PaperQuiz protocol."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from dagent.core.config import Config, ROOT, dump_json
from dagent.core.evaluation.benchmark_eval import _ocr_proxy
from evaluation import judge_fill, judge_paperquiz
from evaluation.run import run, summarize


def partial_result():
    aspects = {key: {'score': score} for key, score in zip(
        ('element', 'layout', 'engagement', 'clarity', 'content', 'logic'),
        (1.0, 3.0, 5.0, 4.0, None, None))}
    judge = judge_fill.recompute_avgs(aspects)
    judge['status'] = 'incomplete'
    return dict(status='complete', metrics=dict(judge=judge, paperquiz=dict(
        status='ok', verbatim_raw=60.0, interpretive_raw=80.0, overall_raw=70.0,
        verbatim_aug=120.0, interpretive_aug=160.0, overall_aug=140.0)))


class AggregationTests(unittest.TestCase):
    def test_partial_judge_keeps_available_aspects_and_independent_quiz(self):
        summary = summarize([dict(status='fail', result=partial_result())])
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(summary['means']['aesthetic_avg'], 3.0)
        self.assertEqual(summary['means']['information_avg'], 4.0)
        # Comparison Overall averages the four available aspects directly.
        self.assertEqual(summary['means']['overall_judge'], 3.25)
        self.assertEqual(summary['means']['overall_raw'], 70.0)
        self.assertEqual(summary['valid_counts']['logic'], 0)
        self.assertEqual(summary['valid_counts']['overall_raw'], 1)

    def test_failed_quiz_does_not_discard_valid_judge(self):
        result = partial_result()
        result['metrics']['paperquiz'].update(status='incomplete', overall_raw=0.0)
        summary = summarize([dict(status='fail', result=result)])
        self.assertEqual(summary['means']['overall_judge'], 3.25)
        self.assertIsNone(summary['means']['overall_raw'])

    def test_failed_attempt_does_not_contribute_checkpoint_scores(self):
        rows = []
        for state in ('failed', 'interrupted', 'running'):
            result = partial_result()
            result['status'] = state
            rows.append(dict(status='fail', result=result))
        rows.append(dict(status='skipped'))
        self.assertTrue(all(n == 0 for n in summarize(rows)['valid_counts'].values()))

    def test_columns_have_independent_denominators(self):
        first, second = partial_result(), partial_result()
        second['metrics']['judge']['aspects']['logic']['score'] = 5.0
        second['metrics']['judge']['overall'] = 3.6
        second['metrics']['paperquiz']['overall_raw'] = 90.0
        summary = summarize([dict(status='fail', result=r) for r in (first, second)])
        self.assertAlmostEqual(summary['means']['overall_judge'], 3.425)
        self.assertEqual(summary['means']['overall_raw'], 80.0)
        self.assertEqual(summary['valid_counts']['logic'], 1)
        self.assertEqual(summary['valid_counts']['overall_judge'], 2)


class EvaluationFilesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='eval_test_', dir=ROOT / 'outputs')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        guard = patch('openai.OpenAI', side_effect=AssertionError('Real API forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_length_inputs_preserve_reference_formula(self):
        self.assertEqual(_ocr_proxy({'text_len_words_proxy': 1200}),
                         (1200.0, 'manifest_text_len_words_proxy'))
        self.assertEqual(_ocr_proxy({'ocr_chars': 6004}), (1200, 'manifest_ocr_chars//5'))
        self.assertEqual(_ocr_proxy({'ocr_chars': 0})[0], 1)
        self.assertEqual(_ocr_proxy({}), (400.0, 'reference_fallback_400'))
        cache = self.root / 'ocr.json'
        for rows in ([{'slug': 'legacy-name', 'ocr_chars': 6004}],
                     {'rows': [{'slug': 'legacy-name', 'ocr_chars': 6004}]}):
            dump_json(cache, rows)
            self.assertEqual(_ocr_proxy(dict(paper_id='new-name', ocr_cache_slug='legacy-name',
                ocr_cache_path=str(cache))), (1200, 'cached_ocr_chars//5'))
        self.assertAlmostEqual(judge_paperquiz.density_aug(60, 1200), 98.7)

    def test_explicit_bad_length_inputs_cannot_silently_fall_back(self):
        for value in (None, 0, -1, float('nan'), float('inf'), True, '400'):
            with self.subTest(length=value), self.assertRaises(ValueError):
                _ocr_proxy({'text_len_words_proxy': value})
        for value in (None, -1, True, '2000', 1.5):
            with self.subTest(chars=value), self.assertRaises(ValueError):
                _ocr_proxy({'ocr_chars': value})
        with self.assertRaises(FileNotFoundError):
            _ocr_proxy({'ocr_cache_path': str(self.root / 'missing.json')})
        cache = self.root / 'bad_ocr.json'
        for rows in ([], [{'slug': 'other', 'ocr_chars': 100}],
                     [{'slug': 'paper'}, {'slug': 'paper'}], {'rows': 'bad'}):
            dump_json(cache, rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                _ocr_proxy(dict(paper_id='paper', ocr_cache_path=str(cache)))

    def test_full_runner_and_report_keep_partial_judge_and_completed_quiz(self):
        folder = self.root / 'generation' / 'paper'
        folder.mkdir(parents=True)
        Image.new('RGB', (64, 48), 'white').save(folder / 'poster.png')
        qa = self.root / 'qa.json'
        bucket = dict(questions={'q1': dict(question='Which method?',
            options=['A. Proposed', 'B. Other'])}, answers={'q1': 'A'})
        dump_json(qa, dict(detail=bucket, understanding=deepcopy(bucket)))
        manifest = self.root / 'manifest.json'
        dump_json(manifest, dict(papers=[dict(paper_id='paper', qa_path=str(qa),
            text_len_words_proxy=1200)]))
        config = Config()
        config.eval_fill_attempts = 1
        requests = []

        def transport(request):
            requests.append(request)
            prompt = request['messages'][0]['content'][0]['text']
            if prompt.startswith('Answer the multiple-choice'):
                self.assertNotIn('answers', prompt)
                answer = 'A'
            elif "Score ONLY 'Logical Flow'" in prompt or not prompt.startswith('You are a scientific poster judge.'):
                answer = ''  # Fail the one detailed fill attempt too.
            else:
                answer = '{"score": 4, "reason": "Readable"}'
            return dict(choices=[dict(message=dict(content=answer))],
                        usage=dict(prompt_tokens=100, completion_tokens=10, total_tokens=110))

        with patch('time.sleep'):
            summary = run(self.root, manifest, config, ('judge', 'qa'), transport=transport)
        self.assertEqual(len(requests), 9)  # Six compact, one fill, two questions.
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(summary['means']['overall_judge'], 4.0)
        self.assertEqual(summary['valid_counts']['logic'], 0)
        self.assertEqual(summary['means']['overall_raw'], 100.0)
        self.assertAlmostEqual(summary['means']['overall_aug'], 164.5)
        report = json.loads((folder / 'evaluation' / 'metrics.json').read_text())
        self.assertEqual(report['metrics']['paperquiz']['length_source'], 'manifest_text_len_words_proxy')
        self.assertEqual(run(self.root, manifest, config, ('judge', 'qa'), report_only=True), summary)

        from dagent.agents import runner
        dump_json(self.root / 'protocol.json', dict(entries=[dict(paper_id='paper')], full_benchmark=False))
        dump_json(folder / 'status.json', dict(status='complete'))
        # Reporting must read the canonical report, rather than a stale copy.
        dump_json(folder / 'evaluation_result.json', dict(status='failed', metrics={}))
        generated_report = runner.report(self.root)
        self.assertEqual(generated_report['evaluation'], summary)
        self.assertEqual(generated_report['evaluation_states']['incomplete'], 1)

    def test_explicit_missing_inputs_fail_before_any_requests(self):
        folder = self.root / 'generation' / 'paper'
        folder.mkdir(parents=True)
        Image.new('RGB', (64, 48), 'white').save(folder / 'poster.png')
        qa = self.root / 'qa.json'
        dump_json(qa, dict(detail=dict(questions={}), understanding=dict(questions={})))
        manifest = self.root / 'manifest.json'
        for entry in (dict(paper_id='paper', qa_path=str(self.root / 'missing_qa.json')),
                      dict(paper_id='paper', qa_path=str(qa), ocr_cache_path=str(self.root / 'missing_ocr.json'))):
            dump_json(manifest, dict(papers=[entry]))
            with self.subTest(entry=entry):
                def forbidden(request):
                    self.fail('Input validation must precede paid requests')
                summary = run(self.root, manifest, Config(), ('judge', 'qa'), transport=forbidden)
                self.assertEqual(summary['failed'], 1)
                self.assertTrue(all(n == 0 for n in summary['valid_counts'].values()))
                report = json.loads((folder / 'evaluation' / 'metrics.json').read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['error_type'], 'FileNotFoundError')


class RequestProtocolTests(unittest.TestCase):
    def test_quiz_request_budget_and_answer_key_remain_reference_compatible(self):
        bucket = judge_paperquiz.merge_qa_bucket(dict(
            questions={'q': dict(question='Which result?', options=['A. One', 'B. Two'])},
            answers={'q': 'B'}))
        for model, expected_tokens in (('deepseek-flash', 1024), ('gpt-5.5', 256)):
            requests = []
            def create(**request):
                requests.append(request)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='B'))])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with patch('time.sleep'):
                result = judge_paperquiz.run_quiz_bucket(client, model, 'IMAGE', bucket, 50)
            self.assertEqual(result, (100.0, 1, 1))
            self.assertEqual(requests[0]['max_tokens'], expected_tokens)
            content = requests[0]['messages'][0]['content']
            self.assertEqual(content[0]['text'],
                'Answer the multiple-choice question using ONLY the poster image. '
                'Reply with a single letter A/B/C/D.\n\nQ: Which result?\nA. One\nB. Two')
            self.assertEqual(content[1]['image_url']['url'], 'data:image/jpeg;base64,IMAGE')


if __name__ == '__main__':
    unittest.main()
