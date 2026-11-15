import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.palette_features import palette_from_rgb,extract_palette,PIXEL_NAMES,cached_color_proxies
from scripts.analyze_palettes import palette_arrays,qvalues,distribution,OnlyThumbnailRedirects
from urllib.error import URLError
from urllib.request import Request


class PaletteExtractorTests(unittest.TestCase):
    def test_uniform_red_palette_is_finite_and_red(self):
        rgb=np.full((64,64,3),[255,0,0],dtype=np.uint8)
        features,colors=palette_from_rgb(rgb)
        self.assertEqual(set(features),set(PIXEL_NAMES))
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertEqual(colors[0]['rgb'],[255,0,0])
        self.assertEqual(colors[0]['fraction'],1)
        self.assertEqual(features['pal_red_accent_fraction'],1)
        self.assertEqual(features['pal_high_saturation_fraction'],1)
        self.assertAlmostEqual(features['pal_effective_color_count'],1)
        self.assertEqual(features['pal_complementary'],0)

    def test_complementary_vs_analogous_pairs(self):
        rgb=np.zeros((64,64,3),np.uint8)
        rgb[:,:32]=[255,0,0];rgb[:,32:]=[0,255,255]
        features,colors=palette_from_rgb(rgb)
        self.assertEqual(features['pal_complementary'],1)
        self.assertEqual(features['pal_analogous'],0)
        self.assertAlmostEqual(features['pal_chromatic_pair_mass'],.5)
        rgb[:,32:]=[255,80,0]
        analogous,_=palette_from_rgb(rgb)
        self.assertGreater(analogous['pal_analogous'],.5)
        self.assertEqual(analogous['pal_complementary'],0)

    def test_neutral_hue_and_black_white_contrast(self):
        gray=np.full((64,64,3),120,np.uint8)
        features,_=palette_from_rgb(gray)
        self.assertEqual(features['pal_hue_concentration'],0)
        self.assertEqual(features['pal_red_accent_present'],0)
        grid=np.indices((64,64)).sum(axis=0)//8%2
        # Pattern has black/white in each row and column, not an edge letterbox.
        rgb=np.repeat((grid*255).astype(np.uint8)[:,:,None],3,axis=2)
        contrast,_=palette_from_rgb(rgb)
        self.assertEqual(contrast['pal_chromatic_pair_mass'],0)
        self.assertGreater(contrast['pal_max_pair_brightness_contrast'],.95)

    def test_deterministic_no_global_rng_mutation_and_jpeg_decode(self):
        rgb=np.random.default_rng(3).integers(0,256,(192,108,3),dtype=np.uint8)
        state=np.random.get_state()
        first=palette_from_rgb(rgb)
        second=palette_from_rgb(rgb)
        self.assertEqual(first,second)
        after=np.random.get_state()
        self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frame.jpg';Image.fromarray(rgb).save(path)
            values,swatches=extract_palette(path)
            self.assertEqual(len(values),50)
            self.assertTrue(1<=len(swatches)<=5)

    def test_invalid_shape_and_all_black_do_not_fabricate_harmony(self):
        with self.assertRaises(ValueError):palette_from_rgb(np.zeros((5,5),np.uint8))
        black,_=palette_from_rgb(np.zeros((32,32,3),np.uint8))
        self.assertTrue(np.isfinite(list(black.values())).all())
        self.assertEqual(black['pal_retained_fraction'],1)
        self.assertEqual(black['pal_complementary'],0)


class PaletteAnalysisTests(unittest.TestCase):
    def test_fitting_only_imputation_and_readonly_frame_support(self):
        table=pd.DataFrame({'video_id':['a','b','test'],'x':[1.,3.,1000.]})
        matrix,medians=palette_arrays(pd.DataFrame({'video_id':['a','b']}),table,['x'])
        self.assertEqual(medians.tolist(),[2])
        missing,_=palette_arrays(pd.DataFrame({'video_id':['missing']}),table,['x'],medians)
        self.assertEqual(missing.tolist(),[[2.,0.]])
        dropped,_=palette_arrays(pd.DataFrame({'video_id':['test']}),table,['x'],medians,dropout=np.array([True]))
        self.assertEqual(dropped.tolist(),[[2.,0.]])

    def test_bh_correction_and_within_channel_quartiles(self):
        np.testing.assert_allclose(qvalues([.01,.04,.03]),[.03,.04,.04])
        self.assertEqual(qvalues([None]),[None])
        frame=pd.DataFrame([dict(channel_id=str(c),target=float(i)) for c in range(8) for i in range(12)])
        values=pd.DataFrame({'signal':frame.target.to_numpy(),'constant':np.ones(len(frame))})
        result=distribution(frame,values)
        self.assertGreater(result[0]['mean_channel_difference'],0)
        self.assertLess(result[0]['fdr_q'],.05)
        self.assertEqual(result[1]['channel_signed_rank_p'],1)

    def test_redirect_scope_stays_on_thumbnail_host(self):
        handler=OnlyThumbnailRedirects()
        with self.assertRaises(URLError):
            handler.redirect_request(Request('https://i.ytimg.com/x'),None,302,'',{},'https://example.com/x')

    def test_cached_proxies_do_not_claim_pixel_palette_information(self):
        frame=pd.DataFrame(dict(thumb_dom_h=[0.],thumb_dom_s=[1.],thumb_dom_v=[1.],thumb_dom_frac=[.5],
            thumb_saturation_mean=[.5],thumb_saturation_std=[.2],thumb_warm_frac=[.3],
            thumb_dark_frac=[.1],thumb_bright_frac=[.1]))
        proxies=cached_color_proxies(frame)
        self.assertEqual(proxies.shape,(1,12))
        self.assertFalse(any('harmony' in name or 'complementary' in name or 'high_saturation_fraction' in name for name in proxies))
        self.assertEqual(proxies.cached_dom_red_mass_proxy.iloc[0],.5)


if __name__=='__main__':unittest.main()
