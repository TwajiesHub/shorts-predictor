#!/usr/bin/env python3
"""Paired off/on CPU benchmark of optional rules using the same frozen model."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scripts.predict import load_bundle,predict_scores,parse_when
from src.proxy_rules import ProxyRuleContext


def main():
    bundle=load_bundle('v5')
    start=time.perf_counter();context=ProxyRuleContext(bundle)
    initialization=time.perf_counter()-start
    titles=['Why Does My 128GB iPhone Only Show 113GB Free?',
            'Your 128GB iPhone Is Lying About Its Storage',"Why 128GB on iPhone Isn't Really 128GB"]
    when=parse_when('2026-11-16 10:00')
    result=dict(rule_context_init_seconds=initialization,timings={})
    with tempfile.TemporaryDirectory() as tmp:
        paths=[Path(tmp)/f'frame_{i}.jpg' for i in range(3)]
        pixels=np.random.default_rng(3).integers(0,256,(1920,1080,3),dtype=np.uint8)
        for path in paths:Image.fromarray(pixels).save(path)
        for name,images in [('titles_only',None),('three_thumbnails',paths)]:
            timings={False:[],True:[]}
            for trial in range(11):
                # Alternate order and discard the first pair of warm-up requests.
                outputs={}
                for on in ([False,True] if trial%2==0 else [True,False]):
                    start=time.perf_counter()
                    outputs[on]=predict_scores(bundle,titles,30,when,thumb_paths=images,
                        apply_rules=on,rules_context=context if on else None)
                    elapsed=time.perf_counter()-start
                    if trial:timings[on].append(elapsed)
                np.testing.assert_allclose(outputs[False].score,outputs[True].score,atol=0,rtol=0)
            off=float(np.median(timings[False]));on=float(np.median(timings[True]))
            result['timings'][name]=dict(off_median_seconds=off,on_median_seconds=on,
                overhead_seconds=on-off,overhead_percent=100*(on/off-1),paired_requests=10)
        fresh=[]
        for _ in range(5):
            start=time.perf_counter()
            predict_scores(bundle,titles,30,when,apply_rules=True)
            fresh.append(time.perf_counter()-start)
        result['titles_with_fresh_rule_context_median_seconds']=float(np.median(fresh))
    result['torch_or_clip_loaded']='torch' in sys.modules or 'open_clip' in sys.modules
    assert not result['torch_or_clip_loaded']
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
