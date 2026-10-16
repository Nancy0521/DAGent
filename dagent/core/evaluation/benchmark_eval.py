from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import hashlib
import time
import math

from dagent.core.evaluation.reference_client import ReferenceClient
from dagent.core.budget import BudgetLedger
from dagent.core.config import BudgetConfig,dump_json,writable_path
from dagent.core.document import sha256
from dagent.core.evaluation.records import (normalize_metrics, input_identity, evaluation_lock,
                                          aggregate_cost, cached_report)

ASPECTS=('element', 'layout', 'engagement', 'clarity', 'content', 'logic')
PROTOCOL_VERSION = 5


def json_finite(value):
    """The reference uses NaN for an empty bucket; serialize missing values as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(v) for v in value]
    return value


def evaluate(run_dir,entry,config,metrics=('judge','qa'),transport=None,*,retry_incomplete=False):
    config.validate()
    metrics=normalize_metrics(metrics)
    run_dir=writable_path(run_dir)
    status_path=run_dir/'status.json'
    if status_path.is_file():
        status=json.loads(status_path.read_text())
        if status.get('status')!='complete':
            raise ValueError('Only completed generations may be benchmarked')
        if entry.get('pdf_sha256') and status.get('pdf_sha256')!=entry['pdf_sha256']:
            raise ValueError('Run source does not match benchmark paper; refusing mismatched QA')
        if 'simulated_api' in status and status['simulated_api'] != (transport is not None):
            raise ValueError('Cannot mix simulated and real generation/evaluation')
    image=run_dir/'poster.png'
    if not image.is_file():
        raise FileNotFoundError('No generated poster image to evaluate')
    canonical=writable_path(run_dir/'evaluation')
    with evaluation_lock(canonical):
        # Reconcile old interrupted attempts even when preflight or cache lookup fails.
        aggregate_cost(canonical)
        sources=input_identity(run_dir,entry,config,metrics,ASPECTS)
        fingerprint=hashlib.sha256(json.dumps(dict(protocol_version=PROTOCOL_VERSION,config=config.to_dict(),
            metrics=metrics,sources=sources,entry=entry,simulated_api=transport is not None),sort_keys=True).encode()).hexdigest()
        previous=cached_report(canonical,fingerprint,retry_incomplete)
        if previous is not None:
            return previous
        out=writable_path(canonical/'attempts'/f'{time.time_ns()}_{fingerprint[:10]}')
        out.mkdir(parents=True,exist_ok=False)
        # Evaluation is a separate budget, never part of generation cost.
        evaluation_config=replace(config,budget=BudgetConfig(max_tokens=config.eval_max_tokens,max_calls=config.eval_max_calls,
            max_vlm_calls=config.eval_max_calls,content_fraction=.5,image_token_reservation=config.budget.image_token_reservation))
        ledger=BudgetLedger(evaluation_config.budget,out/'cost.json')
        api=ReferenceClient(evaluation_config,ledger,out/'logs',transport)
        result=dict(protocol='comparison Judge/PaperQuiz',protocol_version=PROTOCOL_VERSION,
                    judge_prompt_protocol=config.eval_judge_prompt,quiz_limit=config.eval_quiz_limit,
                    generation_cost_excludes_evaluation=True,fingerprint=fingerprint,simulated_api=transport is not None,
                    evaluation_attempt=str(out),status='running',started_at=time.time(),
                    requested_metrics=list(metrics),input_provenance=sources,metrics={})
        ledger.save()
        dump_json(out/'metrics.json',json_finite(result))  # Exists before the first possibly billed request.
        try:
            _compute_metrics(run_dir,entry,config,metrics,image,out,api,result)
            if input_identity(run_dir,entry,config,metrics,ASPECTS) != sources:
                raise ValueError('Evaluation inputs changed during this attempt')
            result['status']='complete'
        except BaseException as exc:
            # Also preserve Ctrl-C/SystemExit. A hard kill leaves the running checkpoint
            # and the ledger's conservative in-flight reservation for the next resume.
            result.update(status='failed' if isinstance(exc,Exception) else 'interrupted',error_type=type(exc).__name__)
            raise
        finally:
            result['finished_at']=time.time()
            ledger.save()
            dump_json(out/'metrics.json',json_finite(result))
            dump_json(canonical/'metrics.json',json_finite(result))
            aggregate_cost(canonical)
        return json_finite(result)


def _ocr_proxy(entry):
    # Portable manifests can retain the exact length recorded by the original
    # comparison, without depending on its private directory layout.
    if 'text_len_words_proxy' in entry:
        length = entry['text_len_words_proxy']
        if type(length) not in (int, float) or not math.isfinite(length) or length <= 0:
            raise ValueError('text_len_words_proxy must be a finite positive number')
        return float(length), 'manifest_text_len_words_proxy'
    if 'ocr_chars' in entry:
        return _chars_proxy(entry['ocr_chars']), 'manifest_ocr_chars//5'
    cache = entry.get('ocr_cache_path')
    if cache:
        data = json.loads(Path(cache).read_text())
        rows = data if isinstance(data, list) else data.get('rows') if isinstance(data, dict) else None
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError('OCR cache must be a list of rows or an object with a rows list')
        slug = entry.get('ocr_cache_slug', entry.get('paper_id'))
        matches = [row for row in rows if row.get('slug') == slug]
        if len(matches) != 1:
            raise ValueError(f'OCR cache must contain exactly one row for {slug!r}')
        return _chars_proxy(matches[0].get('ocr_chars')), 'cached_ocr_chars//5'
    return 400.0, 'reference_fallback_400'


def _chars_proxy(chars):
    if type(chars) is not int or chars < 0:
        raise ValueError('ocr_chars must be a nonnegative integer')
    return max(1, chars // 5)


def _compute_metrics(run_dir,entry,config,metrics,image,out,api,result):
    from evaluation import judge_paperquiz as reference
    # Validate existing QA before any potentially billed request.
    qa = entry.get('qa_path')
    buckets = reference.load_qa(qa) if 'qa' in metrics and qa else None
    length, length_source = _ocr_proxy(entry) if 'qa' in metrics else (None, None)
    b64 = reference.b64_jpeg(image) if 'judge' in metrics or 'qa' in metrics else None
    if 'judge' in metrics:
        from evaluation import judge_fill
        api.stage = 'eval_judge_compact'
        if config.eval_judge_prompt == 'detailed':
            judge = judge_fill.recompute_avgs({})
        else:
            judge = reference.judge_six(api, config.vision_model.name, b64)
        result['metrics']['judge'] = judge
        dump_json(out/'metrics.json', json_finite(result))
        if config.eval_judge_prompt in ('compact-fill', 'detailed'):
            original = judge_fill.png_b64(image)
            for key in judge_fill.missing_aspects(judge):
                api.stage = 'eval_judge_fill_' + key
                judge['aspects'][key] = judge_fill.score_one_aspect(api, config.vision_model.name,
                    original, key, max_attempts=config.eval_fill_attempts)
                judge = judge_fill.recompute_avgs(judge['aspects'])
                result['metrics']['judge'] = judge
                dump_json(out/'metrics.json', json_finite(result))
        missing = judge_fill.missing_aspects(judge)
        judge.update(status='incomplete' if missing else 'ok', model=config.vision_model.name,
                     missing_aspects=missing, prompt_protocol=config.eval_judge_prompt,
                     valid_aspect_count=6-len(missing))
    dump_json(out/'metrics.json', json_finite(result))
    if 'qa' in metrics:
        if buckets is None:
            quiz = dict(status='skipped', reason='matched_question_bank_missing')
        elif not any(buckets):
            quiz = dict(status='skipped', reason='empty_question_bank')
        else:
            api.stage = 'eval_paperquiz'
            quiz = reference.paperquiz(api, config.vision_model.name, b64, buckets,
                limit=config.eval_quiz_limit, text_len_words_proxy=length)
            quiz.update(length_source=length_source, qa_sha256=sha256(Path(qa)),
                        answer_key_sent_to_model=False, n_questions_sampled=config.eval_quiz_limit)
            if not all((quiz['verbatim_n'], quiz['interpretive_n'])):
                quiz.update(status='incomplete', reason='missing_quiz_bucket')
        result['metrics']['paperquiz'] = json_finite(quiz)
    dump_json(out/'metrics.json', json_finite(result))
