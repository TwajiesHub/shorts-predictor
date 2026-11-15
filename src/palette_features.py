"""Deterministic, bounded-resolution thumbnail palette features; no learned vision.

The separate cached-color proxies never pretend to reconstruct unobserved pixels.
Production v1–v5 feature builders do not import this research module.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.thumb_features import _letterbox_masks, _colorfulness

CONFIG = dict(max_side=128,rgb_bits_per_channel=4,dominant_colors=5,
    chromatic_saturation=.25,chromatic_value=.20,high_saturation=.65,accent_value=.50,
    accent_presence_fraction=.02,red_degrees=[0,20,340,360],yellow_degrees=[40,75],
    warm_degrees=[0,60,330,360],cool_degrees=[90,270],
    complementary_tolerance_degrees=30,analogous_tolerance_degrees=45,
    pair_min_fraction=.02,pair_min_saturation=.35,minimum_crop_fraction=.5)

PIXEL_NAMES = ['pal_mean_hue_sin','pal_mean_hue_cos','pal_hue_concentration',
    'pal_mean_saturation','pal_mean_value','pal_mean_brightness','pal_colorfulness',
    'pal_warm_fraction','pal_cool_fraction','pal_warm_cool_ratio','pal_high_saturation_fraction',
    'pal_vivid_fraction','pal_red_accent_fraction','pal_yellow_accent_fraction',
    'pal_red_accent_present','pal_yellow_accent_present','pal_effective_color_count',
    'pal_top5_coverage','pal_retained_fraction']
for _i in range(1,6):
    PIXEL_NAMES += [f'pal_color{_i}_{s}' for s in ['hue_sin','hue_cos','saturation','value','fraction']]
PIXEL_NAMES += ['pal_complementary','pal_analogous','pal_chromatic_pair_mass',
                'pal_max_pair_brightness_contrast','pal_max_pair_rgb_distance','pal_high_contrast_pair_mass']


def palette_from_rgb(rgb):
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<2:
        raise ValueError('Expected a nonempty RGB image with at least two pixels per side')
    if rgb.dtype!=np.uint8:raise ValueError('RGB image must use uint8 values')
    h,w=rgb.shape[:2]
    scale=min(1.,128/max(h,w))
    if scale<1:rgb=cv2.resize(rgb,(max(2,round(w*scale)),max(2,round(h*scale))),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    rowbars,colbars,_=_letterbox_masks(gray)
    ys,xs=np.flatnonzero(~rowbars),np.flatnonzero(~colbars)
    retained=1.
    if len(ys)>=8 and len(xs)>=8:
        crop=rgb[ys[0]:ys[-1]+1,xs[0]:xs[-1]+1]
        fraction=crop.shape[0]*crop.shape[1]/(rgb.shape[0]*rgb.shape[1])
        if fraction>=.5:rgb=crop;retained=fraction
    pixels=rgb.reshape(-1,3)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV).reshape(-1,3).astype(float)
    hue=hsv[:,0]*2; saturation=hsv[:,1]/255;value=hsv[:,2]/255
    radians=np.deg2rad(hue);weights=saturation*value;weight_sum=max(weights.sum(),1e-12)
    sine=float(np.dot(weights,np.sin(radians))/weight_sum)
    cosine=float(np.dot(weights,np.cos(radians))/weight_sum)
    chromatic=(saturation>=.25)&(value>=.20)
    warm=chromatic&((hue<60)|(hue>=330));cool=chromatic&(hue>=90)&(hue<270)
    vivid=(saturation>=.65)&(value>=.5)
    red=vivid&((hue<=20)|(hue>=340));yellow=vivid&(hue>=40)&(hue<=75)
    # 4 bits/channel => 4,096 fixed bins. No k-means seeds or global RNG mutation.
    q=(pixels.astype(np.int32)>>4)
    codes=(q[:,0]<<8)+(q[:,1]<<4)+q[:,2]
    counts=np.bincount(codes,minlength=4096)
    occupied=np.flatnonzero(counts)
    ordered=occupied[np.lexsort((occupied,-counts[occupied]))][:5]
    sums=[np.bincount(codes,weights=pixels[:,i],minlength=4096) for i in range(3)]
    colors=np.array([[sums[c][k]/counts[k] for c in range(3)] for k in ordered])
    swatch_hsv=cv2.cvtColor(np.rint(colors).astype(np.uint8)[None,:,:],cv2.COLOR_RGB2HSV)[0].astype(float)
    fractions=counts[ordered]/len(pixels)
    p=counts[occupied]/len(pixels)
    values=dict(zip(PIXEL_NAMES[:19],[sine,cosine,float(np.hypot(sine,cosine)),
        float(saturation.mean()),float(value.mean()),float(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).mean()/255),
        _colorfulness(rgb[:,:,::-1]),float(warm.mean()),float(cool.mean()),
        float((warm.mean()+.01)/(cool.mean()+.01)),float((saturation>=.65).mean()),
        float(vivid.mean()),float(red.mean()),float(yellow.mean()),float(red.mean()>=.02),
        float(yellow.mean()>=.02),float(np.exp(-np.sum(p*np.log(p)))),float(fractions.sum()),retained]))
    swatches=[]
    for i in range(5):
        terms=np.zeros(5)
        if i<len(ordered):
            H,S,V=swatch_hsv[i]*np.array([2.,1/255,1/255])
            terms=[np.sin(np.deg2rad(H))*S,np.cos(np.deg2rad(H))*S,S,V,fractions[i]]
            color=np.rint(colors[i]).astype(int).tolist()
            swatches.append(dict(rgb=color,hex='#'+''.join(f'{v:02x}' for v in color),
                                 hue_degrees=float(H),saturation=float(S),value=float(V),fraction=float(fractions[i])))
        for name,term in zip(['hue_sin','hue_cos','saturation','value','fraction'],terms):
            values[f'pal_color{i+1}_{name}']=float(term)
    complementary=analogous=mass=contrast=distance=contrast_mass=0.
    for i in range(len(swatches)):
        for j in range(i+1,len(swatches)):
            a,b=swatches[i],swatches[j]
            if min(a['fraction'],b['fraction'])<.02:continue
            pair_weight=2*a['fraction']*b['fraction']
            luma_a=np.dot(a['rgb'],[.299,.587,.114])/255
            luma_b=np.dot(b['rgb'],[.299,.587,.114])/255
            diff=abs(luma_a-luma_b);rgbdiff=np.linalg.norm(np.array(a['rgb'])-b['rgb'])/(255*np.sqrt(3))
            contrast=max(contrast,float(diff));distance=max(distance,float(rgbdiff))
            if diff>=.5 or rgbdiff>=.65:contrast_mass+=pair_weight
            if min(a['saturation'],b['saturation'])<.35 or min(a['value'],b['value'])<.2:continue
            delta=abs(a['hue_degrees']-b['hue_degrees']);delta=min(delta,360-delta)
            mass+=pair_weight
            complementary+=pair_weight*max(0,1-abs(delta-180)/30)
            analogous+=pair_weight*max(0,1-delta/45)
    values.update(pal_complementary=complementary/mass if mass else 0.,
        pal_analogous=analogous/mass if mass else 0.,pal_chromatic_pair_mass=mass,
        pal_max_pair_brightness_contrast=contrast,pal_max_pair_rgb_distance=distance,
        pal_high_contrast_pair_mass=contrast_mass)
    if not np.isfinite(list(values.values())).all():raise ValueError('Nonfinite palette features')
    return values,swatches


def extract_palette(path):
    with Image.open(Path(path)) as image:
        image.draft('RGB',(256,256))
        image=image.convert('RGB')
        image.thumbnail((128,128),Image.Resampling.BOX)
        return palette_from_rgb(np.asarray(image))


def cached_color_proxies(frame):
    """Derived encodings only: no invented top-five colors or harmony scores."""
    h=frame['thumb_dom_h'].to_numpy(float)
    s=frame['thumb_dom_s'].to_numpy(float)
    v=frame['thumb_dom_v'].to_numpy(float)
    f=frame['thumb_dom_frac'].to_numpy(float)
    mean=frame['thumb_saturation_mean'].to_numpy(float)
    std=frame['thumb_saturation_std'].to_numpy(float)
    warm=frame['thumb_warm_frac'].to_numpy(float)
    red=(h<=20/360)|(h>=340/360)
    yellow=(h>=40/360)&(h<=75/360)
    return pd.DataFrame(dict(cached_dom_hue_sin=np.sin(2*np.pi*h)*s,
        cached_dom_hue_cos=np.cos(2*np.pi*h)*s,cached_dom_chroma=s*v,
        cached_dom_visual_mass=f*s*v,cached_dom_red_mass_proxy=f*s*red,
        cached_dom_yellow_mass_proxy=f*s*yellow,cached_warm_saturation_product=warm*mean,
        cached_saturation_dispersion=std/(mean+.05),cached_vivid_dark_product=mean*frame['thumb_dark_frac'].to_numpy(float),
        cached_vivid_bright_product=mean*frame['thumb_bright_frac'].to_numpy(float),
        cached_dom_neutral_mass=f*(1-s),cached_dom_bright_mass=f*v),index=frame.index)
