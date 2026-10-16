"""Offline only: bounded fault injection, source metadata and real local rendering."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import fitz
from dagent.core.api import APIError
from dagent.core.budget import BudgetLedger, BudgetExceeded
from dagent.core.config import Config, ROOT, dump_json
from dagent.core.document import SourceDocument, sha256
from dagent.agents import runner, metadata, evidence_agent, grounded_revision, writing_agent, repair_agent, titles
from tests.test_storyboard import example, MockAPI


class RecordingAPI(MockAPI):
    def __init__(self, responses):
        super().__init__(responses)
        self.prompts = []
    def json(self, stage, system, prompt, **kwargs):
        self.prompts.append(prompt)
        return super().json(stage, system, prompt, **kwargs)


class OfflineCase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT / 'outputs')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        guard = patch.object(runner.APIClient, 'json', side_effect=AssertionError('Real API forbidden'))
        guard.start(); self.addCleanup(guard.stop)

    def source(self, author_lines=None):
        pdf = self.root / 'fixture.pdf'
        doc, _ = example(2)
        with fitz.open() as source:
            page = source.new_page()
            page.insert_text((40, 60), 'Audited scientific title', fontsize=18)
            for y, text, font in author_lines or [(92, 'Alice Smith', 'hebo')]:
                page.insert_text((40, y), text, fontsize=11, fontname=font)
            page.insert_text((40, 145), 'Abstract', fontsize=12)
            page.insert_textbox(fitz.Rect(40, 165, 530, 335), doc['blocks'][0]['text']*2, fontsize=10)
            source.save(pdf)
        return dict(paper_id='fixture', pdf=str(pdf), pdf_sha256=sha256(pdf), parsed_json=None)

    def raw(self):
        entry = self.source()
        doc = evidence_agent.load_source_document(entry, self.root / 'input')
        metadata.recover_authors(doc)
        return doc

    def text_story(self):
        doc, draft = example(2)
        doc['assets'] = []
        for m in draft['modules']:
            m['visuals'] = []
            for b in m['bullets']: b['asset_ids'] = []
        return doc, draft


class MetadataTests(OfflineCase):
    def test_flattened_parser_blocks_do_not_hide_native_names(self):
        raw = self.raw()
        raw.authors = 'Unknown Name'
        raw.blocks = [dict(id='b0',page=1,text='Alice Smith author@example.test Some University')]
        audit = metadata.recover_authors(raw, self.root / 'audit')
        self.assertEqual(raw.authors, 'Alice Smith')
        self.assertEqual(audit['matches'][0]['pdf_sha256'], raw.pdf_sha256)
        self.assertFalse(audit['author_list_completeness_verified'])
        self.assertTrue(runner.read(self.root/'audit/required_author_metadata.json')['valid'])

    def test_pdf_ligatures_accents_and_hyphenated_names(self):
        self.assertEqual(metadata.clean('Gecia Bravo-Hermsdorﬀ∗'), 'Gecia Bravo-Hermsdorff')
        self.assertEqual(metadata.clean('Pierre-Andr´e Maugis'), 'Pierre-André Maugis')
        self.assertTrue(metadata.is_name('Scott Wen-tau Yih'))

    def test_wrapped_names_repaired_with_feedback(self):
        entry = self.source([(92,'Alice Smith','hebo'),(108,'Xuanren','hebo'),(121,'Song','hebo')])
        doc = evidence_agent.load_source_document(entry,self.root/'input')
        audit = metadata.recover_authors(doc,self.root/'audit')
        self.assertEqual(doc.authors,'Alice Smith, Xuanren Song')
        self.assertEqual(audit['attempts_used'],2)
        self.assertTrue(runner.read(self.root/'audit/metadata_attempts/attempt_2.json')['feedback'])

    def test_mixed_fonts_do_not_drop_second_author(self):
        entry=self.source([(92,'Alice Smith','hebo'),(108,'Bob Jones','heit')])
        doc=evidence_agent.load_source_document(entry,self.root/'input')
        audit=metadata.recover_authors(doc)
        self.assertEqual(doc.authors,'Alice Smith, Bob Jones')
        self.assertEqual(audit['attempts_used'],3)

    def test_institutions_and_smallcaps_titles_are_not_names(self):
        for text in ['Computer Vision Lab','AWS AI','UC Santa Barbara','ETH Zurich',
                     'Artificial Intelligence','Alibaba Group','UniDistance Suisse','NEURAL NETWORK ENSEMBLES']:
            with self.subTest(text=text):self.assertFalse(metadata.is_name(text))

    def test_hash_change_fails_closed(self):
        doc=self.raw();doc.pdf_sha256='wrong'
        with self.assertRaisesRegex(ValueError,'hash_mismatch'):metadata.recover_authors(doc)

    def test_no_authors_fails_after_three_local_strategies(self):
        entry=self.source([(92,'Example University','hebo')])
        with self.assertRaisesRegex(evidence_agent.InputFailure,'native_author_recovery_exhausted'):
            evidence_agent.prepare_evidence_input(entry,Config(),self.root/'prepared')
        self.assertEqual(len(list((self.root/'prepared/metadata_attempts').glob('attempt_*.json'))),3)
        self.assertEqual(runner.read(self.root/'prepared/status.json')['stage'],'metadata')

    def test_preflight_persists_authors_and_detects_tampering(self):
        entry=self.source(); folder=self.root/'prepared'
        evidence_agent.prepare_evidence_input(entry,Config(),folder)
        raw,pool=evidence_agent.load_preflight(folder)
        self.assertEqual(raw.authors,'Alice Smith');self.assertEqual(pool,{})
        (folder/'author_metadata_audit.json').write_text('{}')
        with self.assertRaisesRegex(evidence_agent.InputFailure,'changed'):evidence_agent.load_preflight(folder)


class TitleTests(unittest.TestCase):
    def blocks(self):
        return [dict(id='p1b0', page=1, text='SiT Dataset: Socially Interactive Pedestrian Trajectory Dataset for Social Navigation Robots'),
                dict(id='p1b1', page=1, text='Abstract'),
                dict(id='p1b2', page=1, text='The study examines pedestrian trajectories.')]

    def test_localized_application_default_is_not_a_paper_title(self):
        expected = self.blocks()[0]['text']
        for stale in ('PowerPoint 프레젠테이션', 'PowerPoint Presentation', 'Présentation PowerPoint',
                      'A Plausible But Unrelated Old Paper Title'):
            with self.subTest(stale=stale):
                title, audit = titles.select_title(dict(paper_id='SiT_Dataset'), stale, self.blocks())
                self.assertEqual(title, expected)
                self.assertEqual(audit['selected_source'], 'frontmatter_block')
                self.assertFalse(audit['metadata_source_matched'])
                self.assertIn('metadata_not_frontmatter_matched', [r['reason'] for r in audit['rejected']])

    def test_source_matched_non_english_title_is_not_rejected(self):
        text = '사회적으로 상호 작용하는 보행자 궤적 데이터셋'
        blocks = [dict(id='p1b0', page=1, text=text), dict(id='p1b1', page=1, text='Abstract')]
        title, audit = titles.select_title(dict(paper_id='fixture'), text, blocks)
        self.assertEqual(title, text)
        self.assertTrue(audit['metadata_source_matched'])

    def test_split_title_matches_source_without_using_manifest_as_evidence(self):
        text = 'Grounded Methods for Reliable Scientific Posters'
        blocks = [dict(id='p1b0', page=1, text='Grounded Methods for'),
                  dict(id='p1b1', page=1, text='Reliable Scientific Posters'),
                  dict(id='p1b2', page=1, text='Abstract')]
        title, audit = titles.select_title(dict(paper_id='fixture'), text, blocks)
        self.assertEqual(title, text)
        self.assertTrue(audit['metadata_source_matched'])

    def test_manifest_and_pdf_property_cannot_agree_on_an_unsupported_title(self):
        stale = 'Unrelated Old Paper Title'
        title, audit = titles.select_title(dict(paper_id='fixture', source_metadata='title='+stale),
                                           stale, self.blocks())
        self.assertEqual(title, self.blocks()[0]['text'])
        self.assertEqual(audit['selected_source'], 'frontmatter_block')


class EvidenceSelectionTests(OfflineCase):
    def response(self, doc):
        sid=next(b['id'] for b in doc.blocks if 'We evaluate' in b['text'])
        return dict(title='Abstract',authors='Invented Author',assets=[],nodes=[dict(
            id='n0',section='Method',role='method',source_ids=[sid])])

    def test_second_repair_third_success_with_prior_error(self):
        raw=self.raw(); api=RecordingAPI([dict(nodes='bad'),dict(nodes=[]),self.response(raw)])
        result=runner.select_evidence_with_audit(raw,api,Config(),self.root/'output')
        self.assertEqual(result.authors,'Alice Smith');self.assertEqual(result.title,raw.title)
        self.assertEqual(len(api.calls),3)
        self.assertIn('nodes must be a list',api.prompts[1])
        self.assertIn('no valid source-grounded candidate nodes',api.prompts[2])
        self.assertEqual(len(set(api.calls)),3)  # malformed cached response never silently reused
        self.assertEqual(raw.candidates,[])

    def test_first_success_does_not_request_repairs(self):
        raw=self.raw();api=RecordingAPI([self.response(raw)])
        runner.select_evidence_with_audit(raw,api,Config(),self.root/'output')
        self.assertEqual(len(api.calls),1)

    def test_three_failures_preserve_all_audits(self):
        raw=self.raw();api=MockAPI([dict(nodes=[])]*3)
        with self.assertRaisesRegex(ValueError,'no valid source-grounded'):
            runner.select_evidence_with_audit(raw,api,Config(),self.root/'output')
        self.assertEqual(len(runner.read(self.root/'output/understanding_history.json')),3)
        self.assertEqual(len(list((self.root/'output/understanding_attempts').glob('*/understanding_source_audit.json'))),3)

    def test_permanent_error_stops_and_transient_error_retries(self):
        raw=self.raw();api=MockAPI([APIError('Missing API credential environment variable: TEST')])
        with self.assertRaises(APIError):runner.select_evidence_with_audit(raw,api,Config(),self.root/'permanent')
        self.assertEqual(len(api.calls),1)
        api=MockAPI([APIError('temporary'),self.response(raw)])
        runner.select_evidence_with_audit(raw,api,Config(),self.root/'transient')
        self.assertEqual(len(api.calls),2)


class RepairCandidateTests(OfflineCase):
    def exercise(self,outcomes,responses=None):
        doc,draft=self.text_story();api=RecordingAPI(responses or [{},{},{}]);seen=[]
        def render(story,decision,document,pool,cfg,ledger,directory,**kwargs):
            seen.append(dict(decision=deepcopy(decision),**deepcopy(kwargs)))
            for name in runner.PRODUCTS:
                if name.endswith('.json'):dump_json(directory/name,dict(test_fixture=True))
                else:(directory/name).write_bytes(directory.name.encode())
            value=outcomes[len(seen)-1]
            if isinstance(value,Exception):raise value
            return dict(score=value)
        with patch.object(runner,'stage_api',return_value=(api,MagicMock())) as stages, patch.object(runner,'verify_repair_candidate',side_effect=render):
            if all(isinstance(o,Exception) for o in outcomes):
                with self.assertRaisesRegex(ValueError,'three attempts'):runner.repair_poster(draft,doc,{},Config(),self.root)
            else:runner.repair_poster(draft,doc,{},Config(),self.root)
        self.assertEqual(stages.call_args.args[-2:],(3,2))
        return runner.read(self.root/'candidate_selection.json'),api,seen

    def test_first_two_fail_third_succeeds(self):
        defect=runner.RenderedCandidateError([dict(category='rendered_textbox_overflow',target='m0_b0')])
        selection,api,seen=self.exercise([defect,defect,.8])
        self.assertEqual(selection['selected'],'candidate_2')
        self.assertEqual([s['text_safety'] for s in seen],[{},dict(m0_b0=1),dict(m0_b0=2)])
        self.assertIn('rendered_textbox_overflow',api.prompts[2])
        self.assertEqual(len(list((self.root/'feedback_attempts').glob('*/request.json'))),2)

    def test_three_failures_never_publish_final(self):
        selection,api,seen=self.exercise([ValueError('bad geometry')]*3)
        self.assertIsNone(selection['selected'])
        self.assertFalse((self.root/'poster.png').exists())
        self.assertEqual(len(seen),3);self.assertEqual(len(api.calls),3)
        self.assertTrue(all(r['status']=='failed' for r in selection['candidates']))

    def test_valid_incumbent_survives_second_failure_and_skips_third(self):
        selection,api,seen=self.exercise([.9,ValueError('later failed')])
        self.assertEqual(selection['selected'],'candidate_0')
        self.assertEqual(selection['candidates'][2]['status'],'skipped')
        self.assertEqual(len(seen),2);self.assertEqual(len(api.calls),2)

    def test_second_success_skips_third(self):
        selection,api,seen=self.exercise([ValueError('geometry'),.8])
        self.assertEqual(selection['selected'],'candidate_1')
        self.assertEqual(selection['candidates'][2]['status'],'skipped')
        self.assertEqual(len(seen),2)

    def test_rejected_feedback_gets_specific_validation_in_next_prompt(self):
        selection,api,seen=self.exercise([ValueError('bad geometry')]*2+[.8],[{},dict(module_order=[]),{}])
        self.assertIn('Modules must be an exact permutation',api.prompts[2])
        self.assertNotEqual(seen[0]['decision']['module_order'],seen[1]['decision']['module_order'])
        self.assertEqual(selection['selected'],'candidate_2')

    def test_compound_repairs_keep_prior_text_edit(self):
        _,draft=self.text_story();decision=repair_agent.validate_repair_decision({},draft)
        edit=dict(section_id='module:m0',bullet_index=0,text='We evaluate the proposed mechanism under the stated experimental assumptions.',source_ids=['p1b0'])
        defects=[dict(category='rendered_textbox_overflow',target='m0_b0')]
        first=runner._merge_feedback(dict(content_edits=[edit]),draft,decision,defects,True)
        second=runner._merge_feedback({},draft,first,[],False)
        self.assertEqual(second['content_edits'],first['content_edits'])

    def test_numbers_negation_scope_cannot_be_deleted(self):
        _,draft=self.text_story();draft['modules'][0]['bullets'][0]['text']='We do not improve by 12% under these assumptions.'
        for text in ['We improve by 12% under these assumptions.','We do not improve under these assumptions.','We do not improve by 12%.']:
            edit=dict(section_id='module:m0',bullet_index=0,text=text,source_ids=['p1b0'])
            with self.subTest(text=text), self.assertRaisesRegex(ValueError,'scope markers'):
                repair_agent.validate_repair_decision(dict(content_edits=[edit]),draft,allow_content_edits=True)

    def test_actual_three_pptx_pdf_png_renders_and_glyph_gates(self):
        doc,draft=self.text_story();api=MockAPI([{},{},{}])
        cfg=replace(Config(),layout=replace(Config().layout,render_dpi=25))
        ledger=BudgetLedger(replace(cfg.budget,max_calls=3,max_vlm_calls=2),self.root/'cost.json')
        real=runner.PosterRenderer.render
        def inject(renderer,layout,folder):
            result=real(renderer,layout,folder)
            if folder.name in ('candidate_0','candidate_1'):
                result['defects']=[dict(category='rendered_textbox_overflow',target='m0_b0')]
            return result
        with patch.object(runner,'stage_api',return_value=(api,ledger)),patch.object(runner.PosterRenderer,'render',new=inject),patch.object(runner,'audit_layout',wraps=runner.audit_layout) as geometry:
            runner.repair_poster(draft,doc,{},cfg,self.root)
        self.assertEqual(ledger.renderer_calls,3);self.assertEqual(geometry.call_count,3)
        self.assertTrue(runner.read(self.root/'render_checks.json')['valid'])
        self.assertEqual(runner.read(self.root/'candidate_selection.json')['selected'],'candidate_2')
        for name in ('poster.pptx','poster.pdf','poster.png'):self.assertTrue((self.root/name).stat().st_size>100)


class WritingTests(OfflineCase):
    def test_refinement_third_only_when_no_usable_candidate(self):
        doc,draft=self.text_story();api=RecordingAPI([dict(modules=[]),dict(storyboard={}),dict(storyboard=draft)])
        result,_=grounded_revision.revise_grounded_storyboard(doc,dict(modules=[]),{},api,self.root/'refinement')
        self.assertEqual(len(api.calls),3)
        self.assertIn('checks',api.prompts[2]);self.assertTrue(result['modules'])
        self.assertEqual(runner.read(self.root/'refinement/completion.json')['requests_used'],3)

    def test_refinement_keeps_valid_seed_after_errors_without_third(self):
        doc,draft=self.text_story();api=MockAPI([dict(modules=[]),dict(storyboard={})])
        result,_=grounded_revision.revise_grounded_storyboard(doc,draft,{},api,self.root/'refinement')
        self.assertEqual(len(api.calls),2);self.assertTrue(result['modules'])

    def test_draft_transport_error_receives_bounded_retry(self):
        doc,draft=self.text_story();api=RecordingAPI([APIError('temporary'),draft,dict(issues=[]),draft])
        result=writing_agent.construct_provenance_storyboard(doc,api,self.root/'draft')
        self.assertTrue(result['modules']);self.assertLessEqual(len(api.calls),4)
        self.assertIn('Previous request failed',api.prompts[1])

    def test_full_generate_metadata_lock_and_real_render(self):
        entry=self.source();doc,draft=self.text_story()
        raw=evidence_agent.load_source_document(entry,self.root/'inspect')
        sid=next(b['id'] for b in raw.blocks if 'We evaluate' in b['text'])
        for section in [draft['intro'],*draft['modules'],draft['conclusion']]:
            for b in section['bullets']:b['source_ids']=[sid]
        calls=[]
        def respond(api,stage,*args,**kwargs):
            calls.append(stage)
            if stage.startswith('content_understanding'):
                return dict(title='Abstract',authors='',assets=[],nodes=[dict(id='n0',section='Method',role='method',source_ids=[sid])])
            if stage=='grounded_review':return dict(issues=[])
            if stage=='layout_plan' or stage.startswith('visual_feedback'):return {}
            if stage.startswith('grounded_check_and_revision'):return dict(review={},storyboard=draft)
            return deepcopy(draft)
        cfg=replace(Config(),layout=replace(Config().layout,render_dpi=25))
        with patch.object(runner.APIClient,'json',new=respond):runner.generate_one(entry,cfg,self.root/'generation')
        status=runner.read(self.root/'generation/status.json')
        self.assertEqual(status['status'],'complete',status)
        self.assertEqual(runner.read(self.root/'generation/understood_document.json')['authors'],'Alice Smith')
        self.assertEqual(calls.count('content_understanding'),1)
        self.assertNotIn('visual_feedback_2',calls)
        self.assertLessEqual(len(calls),9)

    def test_budget_and_provider_permanent_categories(self):
        from dagent.agents.retry import failure_feedback
        self.assertFalse(failure_feedback(BudgetExceeded('limit'))['retryable'])
        api=MagicMock();api.ledger.events=[dict(status='error',error_category='quota_or_balance')]
        self.assertFalse(failure_feedback(APIError('wrapped'),api)['retryable'])



class PreflightRetryTests(OfflineCase):
    def test_timeout_then_success_retains_feedback_and_hash_envelope(self):
        import subprocess
        from types import SimpleNamespace
        entry=self.source(); directory=self.root/'run'
        dump_json(directory/'protocol.json',dict(entries=[entry]))
        dump_json(directory/'config.json',Config().to_dict())
        seen=[]
        def worker(command,**kwargs):
            seen.append(kwargs['timeout'])
            attempt=int(command[command.index('--attempt')+1])
            if attempt==1:raise subprocess.TimeoutExpired(command,kwargs['timeout'])
            runner.preflight_worker(directory,entry['paper_id'],attempt)
            return SimpleNamespace(returncode=0)
        with patch.object(runner,'check_input'),patch.object(runner.subprocess,'run',side_effect=worker):
            result=runner.ensure_preflight(directory,entry)
        self.assertEqual(seen,[180,360])
        status=runner.read(result/'status.json');self.assertEqual(status['status'],'ready')
        self.assertEqual(status['attempts_used'],2)
        self.assertEqual(evidence_agent.load_preflight(result)[0].authors,'Alice Smith')
        feedback=runner.read(result/'attempts/attempt_2/attempt_request.json')['feedback']
        self.assertEqual(feedback['failure_code'],'input_preflight_timeout')
        (result/'attempts/attempt_2/status.json').write_text('{}')
        with self.assertRaisesRegex(evidence_agent.InputFailure,'changed'):evidence_agent.load_preflight(result)

    def test_three_timeouts_stop_without_fourth_attempt(self):
        import subprocess
        entry=self.source()
        with patch.object(runner,'check_input'),patch.object(runner.subprocess,'run',side_effect=subprocess.TimeoutExpired('test',1)) as call:
            result=runner.ensure_preflight(self.root/'run',entry)
        self.assertEqual(call.call_count,3)
        self.assertEqual([c.kwargs['timeout'] for c in call.call_args_list],[180,360,540])
        self.assertEqual(runner.read(result/'status.json')['status'],'failed')
        self.assertEqual(len(runner.read(result/'attempt_history.json')),3)

    def test_permanent_input_error_is_not_blindly_retried(self):
        entry=self.source()
        with patch.object(runner,'check_input',side_effect=ValueError('hash changed')),patch.object(runner.subprocess,'run') as worker:
            result=runner.ensure_preflight(self.root/'run',entry)
        worker.assert_not_called();self.assertEqual(runner.read(result/'status.json')['status'],'failed')


class SourceReferenceTests(OfflineCase):
    def test_known_asset_alias_is_exact_and_unknown_reference_stays_invalid(self):
        from dagent.agents.source_refs import normalize_source_references
        doc,draft=example(2)
        draft['modules'][0]['bullets'][0]['source_ids']=['a0','p1b0','a999']
        result,changes=normalize_source_references(draft,doc)
        self.assertEqual(result['modules'][0]['bullets'][0]['source_ids'],['asset:a0','p1b0','a999'])
        self.assertEqual(draft['modules'][0]['bullets'][0]['source_ids'],['a0','p1b0','a999'])
        self.assertEqual(len(changes),1)
        packet,allowed=writing_agent.assemble_evidence_package(doc)
        self.assertIn('asset:a0',allowed)
        self.assertEqual(next(b['text'] for b in packet['source_blocks'] if b['id']=='asset:a0'),doc['assets'][0]['caption'])
        self.assertIn('section_1:bullet_0:unknown_source',writing_agent.validate_provenance_storyboard(result,doc,allowed)[0])

    def test_real_block_id_collision_keeps_original_meaning(self):
        from dagent.agents.source_refs import normalize_source_references
        doc,draft=example(2);doc['blocks'].append(dict(id='a0',page=1,text='A genuine block.'))
        draft['modules'][0]['bullets'][0]['source_ids']=['a0']
        result,changes=normalize_source_references(draft,doc)
        self.assertEqual(result['modules'][0]['bullets'][0]['source_ids'],['a0']);self.assertFalse(changes)

    def test_caption_without_evidence_is_not_invented(self):
        from dagent.agents.source_refs import normalize_source_references
        doc,draft=example(2);doc['assets'][0]['caption']=''
        draft['modules'][0]['bullets'][0]['source_ids']=['a0']
        result,changes=normalize_source_references(draft,doc)
        self.assertFalse(changes);self.assertEqual(result['modules'][0]['bullets'][0]['source_ids'],['a0'])

    def test_formula_fallback_glyph_top_inset_is_really_repaired(self):
        from dagent.core.layout.geometry import Element,Layout,text_height
        cfg=replace(Config(),layout=replace(Config().layout,render_dpi=25))
        ledger=BudgetLedger(cfg.budget,self.root/'cost.json')
        text='Gradient ∇ f(x) under the stated assumptions.'
        element=Element('formula','text',1,1,40,text_height(text,40,48)+.08,text,48)
        layout=Layout(48,36,[element],columns=1)
        first=runner.PosterRenderer(cfg.layout,ledger).render(layout,self.root/'first')
        self.assertTrue((self.root/'first/poster.pdf').is_file())
        self.assertTrue(all(d['category']=='rendered_textbox_overflow' for d in first['defects']))
        element.height+=.4
        second=runner.PosterRenderer(cfg.layout,ledger,text_safety={'formula':1}).render(layout,self.root/'second')
        self.assertFalse(second['defects'],second['defects'])


if __name__=='__main__':unittest.main()
