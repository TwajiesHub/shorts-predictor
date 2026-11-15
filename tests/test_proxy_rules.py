import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.proxy_rules import adjust_relative_score,midpoint_percentile,poor_thumbnail,ProxyRuleContext
from scripts.predict import load_bundle,predict_scores,parse_when
from src.config import ROOT


class RuleMathTests(unittest.TestCase):
    def test_thresholds_are_inclusive_and_do_not_cross(self):
        self.assertAlmostEqual(adjust_relative_score(0,{'humor':.7},channel_prior_percentile=70)['adjusted_score'],.08)
        self.assertAlmostEqual(adjust_relative_score(0,{'humor':.3},channel_prior_percentile=30)['adjusted_score'],-.06)
        self.assertEqual(adjust_relative_score(0,{'humor':.69999},channel_prior_percentile=70)['adjusted_score'],0)
        self.assertEqual(adjust_relative_score(0,{'humor':.30001},channel_prior_percentile=30)['adjusted_score'],0)
        self.assertEqual(adjust_relative_score(0,{'humor':.7},channel_prior_percentile=69.999)['adjusted_score'],0)
        self.assertEqual(adjust_relative_score(0,{'humor':.3},channel_prior_percentile=30.001)['adjusted_score'],0)

    def test_independent_rules_stack_before_clipping(self):
        result = adjust_relative_score(.5,{'curiosity_gap':.7,'controversy_outrage':.7,'humor':.1},
            channel_prior_percentile=70,in_sweet_spot=True,high_outrage_style=False,thumbnail_poor=True)
        self.assertEqual(len(result['rules_fired']),4)
        self.assertAlmostEqual(result['adjusted_score'],.5+.08+.05-.04-.03)

    def test_unknown_context_does_not_become_negative_evidence(self):
        result = adjust_relative_score(0,{'controversy_outrage':.9})
        self.assertEqual(result['rules_fired'],[])
        self.assertEqual(adjust_relative_score(0,{'curiosity_gap':.7},in_sweet_spot=True)['rule_delta'],.05)
        self.assertEqual(adjust_relative_score(0,{'controversy_outrage':.7},high_outrage_style=False)['rule_delta'],-.04)
        self.assertEqual(adjust_relative_score(0,{'controversy_outrage':.7},high_outrage_style=True)['rule_delta'],0)

    def test_three_triggers_and_no_trigger_repetition_rule(self):
        e={'a':.1,'b':.2,'c':.3}
        self.assertEqual(adjust_relative_score(0,e,thumbnail_poor=True)['rule_delta'],-.03)
        self.assertEqual(adjust_relative_score(0,{'a':.1,'b':.2},thumbnail_poor=True)['rule_delta'],0)
        self.assertAlmostEqual(adjust_relative_score(0,{'a':0},channel_prior_percentile=30,very_similar_recent=True)['rule_delta'],-.10)
        self.assertEqual(adjust_relative_score(0,{'a':.01},very_similar_recent=True)['rule_delta'],0)

    def test_clipping_and_invalid_inputs(self):
        result=adjust_relative_score(2.99,{'humor':.7},channel_prior_percentile=70)
        self.assertEqual(result['adjusted_score'],3)
        self.assertAlmostEqual(result['unclipped_rule_delta'],.08)
        self.assertAlmostEqual(result['rule_delta'],.01)
        self.assertTrue(result['rule_score_clipped'])
        self.assertEqual(adjust_relative_score(-10,{})['adjusted_score'],-3)
        with self.assertRaises(ValueError):adjust_relative_score(float('nan'),{})
        with self.assertRaises(ValueError):adjust_relative_score(0,{'x':1.1})
        with self.assertRaises(ValueError):adjust_relative_score(0,{},channel_prior_percentile=101)

    def test_tied_priors_are_neutral_and_missing_data_stays_unknown(self):
        self.assertEqual(midpoint_percentile(0,np.zeros(10)),50)
        self.assertIsNone(midpoint_percentile(0,[0,1]))
        self.assertIsNone(poor_thumbnail({}))
        self.assertFalse(poor_thumbnail(dict(thumb_brightness=.15,thumb_contrast=.10,thumb_edge_density=.02)))
        self.assertTrue(poor_thumbnail(dict(thumb_brightness=.1,thumb_contrast=.09,thumb_edge_density=.2)))


class ProxyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.when=parse_when('2026-11-16 10:00')
        cls.bundle=load_bundle('v5')

    def test_opt_in_adds_fields_without_modifying_any_original_value(self):
        titles=['Why is your phone slow?','3 secret battery tricks']
        plain=predict_scores(self.bundle,titles,30,self.when)
        adjusted=predict_scores(self.bundle,titles,30,self.when,apply_rules=True)
        pd.testing.assert_frame_equal(plain,adjusted[plain.columns],check_exact=True)
        self.assertNotIn('adjusted_score',plain)
        self.assertIn('adjusted_score',adjusted)
        self.assertIn('adjusted_rank',adjusted)

    def test_duplicate_titles_keep_their_thumbnail_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            dark=Path(tmp)/'dark.jpg';good=Path(tmp)/'gradient.jpg'
            Image.new('RGB',(320,180),(8,8,8)).save(dark)
            pixels=np.broadcast_to(np.linspace(60,220,320,dtype=np.uint8)[None,:,None],(180,320,3)).copy()
            Image.fromarray(pixels).save(good)
            out=predict_scores(self.bundle,['Why your 3 secret tricks?']*2,30,self.when,
                channel_id='unseen-rule-test',thumb_paths=[dark,good],apply_rules=True).set_index('thumbnail')
            self.assertTrue(out.loc[str(dark)].proxy_rule_context['thumbnail_poor'])
            self.assertFalse(out.loc[str(good)].proxy_rule_context['thumbnail_poor'])
            self.assertAlmostEqual(out.loc[str(dark)].unclipped_rule_delta,-.03)
            self.assertAlmostEqual(out.loc[str(good)].unclipped_rule_delta,0)

    def test_query_outcomes_are_not_used_for_rule_context(self):
        frame=pd.read_parquet(ROOT/'data/videos.parquet')
        row=frame.loc[frame.video_id==self.bundle['test_ids'][0]].copy()
        engine=ProxyRuleContext(self.bundle)
        before=engine.contexts(row)
        row['views']=1e15;row['age_days']=1e9
        self.assertEqual(before,ProxyRuleContext(self.bundle).contexts(row))
        row['channel_id']='unseen-rule-channel'
        missing=engine.contexts(row)[0]
        self.assertIsNone(missing['channel_prior_percentile'])
        self.assertIsNone(missing['high_outrage_style'])
        self.assertIsNone(missing['in_sweet_spot'])

    def test_v4_and_cli_optional_fields(self):
        out=predict_scores(load_bundle('v4'),['Example'],30,self.when,apply_rules=True)
        self.assertIn('adjusted_score',out)
        command=[sys.executable,str(ROOT/'scripts/predict.py'),'Example','--model','v5','--apply-rules','--json']
        result=subprocess.run(command,capture_output=True,text=True,check=True)
        def reject(x):raise AssertionError(f'Invalid JSON: {x}')
        data=json.loads(result.stdout,parse_constant=reject)
        self.assertIn('adjusted_score',data['drafts'][0])
        bad=subprocess.run(command[:-4]+['--model','v3','--apply-rules','--json'],capture_output=True,text=True)
        self.assertNotEqual(bad.returncode,0)


if __name__=='__main__':unittest.main()
