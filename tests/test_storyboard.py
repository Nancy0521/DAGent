import copy
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from dagent.agents.writing_agent import (
    validate_provenance_storyboard as validate_story,
    construct_provenance_storyboard as generate_story,
    word_count, body_bullets,
)
from dagent.core.api import APIError
from dagent.core.config import LayoutConfig
from dagent.core.layout.layout_planning import make_layout,audit_layout


def example(n=3):
    words='We evaluate the proposed mechanism under the stated experimental assumptions and compare its measured behavior against the reported reference systems.'
    block=dict(id='p1b0',page=1,text=words)
    doc=dict(title='A Source Grounded Adaptive Poster',authors='A. Author, B. Researcher',blocks=[block],
             candidates=[dict(id='n0',section='Method',role='method',text=words,source_ids=['p1b0'],details=[])],
             assets=[dict(id=f'a{i}',kind='figure',page=1,caption='Figure '+str(i),pixel_width=900,pixel_height=450) for i in range(n)])
    def bullet(a=None):
        return dict(text=words,source_ids=['p1b0'],**({'asset_ids':[a]} if a else {}))
    story=dict(intro=dict(heading='Introduction',bullets=[bullet(),bullet()]),
               conclusion=dict(heading='Conclusion & Scope',bullets=[bullet(),bullet()]),
               modules=[dict(id=f'm{i}',heading='Source Grounded Scientific Finding',role='method' if i==0 else 'results',
                             bullets=[bullet(f'a{i}'),bullet(f'a{i}')],
                             visuals=[dict(asset_id=f'a{i}',caption='Evidence from the original scientific experiment',source_ids=['p1b0'])]) for i in range(n)])
    return doc,story


class MockAPI:
    def __init__(self,responses):self.responses=iter(responses);self.calls=[]
    def json(self,stage,*args,**kwargs):
        self.calls.append(stage);value=next(self.responses)
        if isinstance(value,Exception):raise value
        return copy.deepcopy(value)


class ProvenanceStoryboardTests(unittest.TestCase):
    def test_sources_and_bindings(self):
        doc,story=example();self.assertFalse(validate_story(story,doc)[0])
        bad=copy.deepcopy(story);bad['modules'][0]['bullets'][0]['source_ids']=['invented']
        self.assertTrue(validate_story(bad,doc)[0])
        bad=copy.deepcopy(story);bad['modules'][0]['visuals'][0]['asset_id']='unknown'
        self.assertTrue(validate_story(bad,doc)[0])
        bad=copy.deepcopy(story);bad['modules'][0]['bullets'][0]['asset_ids']=['a1']
        self.assertTrue(validate_story(bad,doc)[0])

    def test_variable_modules_and_actual_layout(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as tmp:
            image=Path(tmp)/'asset.png';Image.new('RGB',(2800,1400),'white').save(image)
            cfg=LayoutConfig(margin_inches=.65,design='editorial',minimum_image_dpi=65)
            for n in (2,3,4,5):
                doc,story=example(n);assets={a['id']:dict(path=str(image),aspect=2,pixels=[2800,1400]) for a in doc['assets']}
                layout,search=make_layout(story,doc,assets,cfg);audit=audit_layout(story,layout,assets,cfg)
                self.assertEqual(sum(e.kind=='figure' for e in layout.elements),n)
                self.assertFalse(audit['missing_atoms']);self.assertEqual(audit['hard_violations'],0,audit['defects'])
                self.assertGreater(search['min_figure_dimension'],2.5)
                self.assertEqual(sum(e.kind=='text' for e in layout.elements),len(list(body_bullets(story))))
                for e in (e for e in layout.elements if e.kind=='figure'):
                    c=layout.get(e.id.removesuffix('_image')+'_caption')
                    self.assertLess(abs(c.y-e.y-e.height-.12),1e-8)

    def test_bounded_last_usable_candidate(self):
        doc,story=example();new=copy.deepcopy(story)
        new['modules'][0]['heading']='Revised Scientific Finding'
        api=MockAPI([story,{'issues':[{'target':'m0','fix':'shorten'}]},new,APIError('hidden')])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as tmp:
            result=generate_story(doc,api,tmp)
            self.assertEqual(result['modules'][0]['heading'],'Revised Scientific Finding')
            self.assertLessEqual(len(api.calls),4)
            self.assertTrue((Path(tmp)/'candidate_0.json').exists())

    def test_malformed_repair_preserves_previous(self):
        doc,story=example();api=MockAPI([story,{'issues':[{'fix':'test'}]}, {'oops':1}, {'oops':2}])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as tmp:
            self.assertEqual(generate_story(doc,api,tmp),story)
            self.assertEqual(len(api.calls),4)

    def test_review_after_initial_invalid_candidate(self):
        doc,story=example();api=MockAPI([{'oops':1},story,{'issues':[{'fix':'check'}]},story])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as tmp:
            generate_story(doc,api,tmp);self.assertEqual(len(api.calls),4)

    def test_no_usable_candidate_is_not_fabricated(self):
        doc,_=example();api=MockAPI([{}, {}, {}])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as tmp:
            with self.assertRaises(ValueError): generate_story(doc,api,tmp)
            self.assertFalse((Path(tmp)/'storyboard.json').exists())

if __name__=='__main__':unittest.main()
