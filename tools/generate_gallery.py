"""Rebuild the public DIFFRUX figures from redistributable bundled fixtures.

Run from a Python 3.12 environment with the normal application dependencies:
    python tools/generate_gallery.py
No private CIF, experimental result, AI image, or external renderer is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afruz_pxrd.version import APP_RELEASE
from afruz_pxrd.crystallography import load_cif, calculate_powder_pattern
from afruz_pxrd.crystal_scene import (
    SCIENTIFIC_STYLE, RenderSettings, build_scene,
    model_from_structure, scene_document, load_scene_document,
)
from afruz_pxrd.crystal_render import render_crystal
from afruz_pxrd.phase_identification import reference_from_cif_pattern
from afruz_pxrd.whole_pattern_refinement import WholePatternPhaseSpec, refine_whole_pattern


def font(size, bold=False):
    for candidate in (Path('C:/Windows/Fonts') / ('segoeuib.ttf' if bold else 'segoeui.ttf'),
                      'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def model(relative):
    structure = load_cif(ROOT / relative)
    # Public sidecars identify repository-relative inputs, never local user paths.
    structure['source_path'] = relative
    return structure, model_from_structure(structure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/figures')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    source_nacl = 'Afruz_PXRD_Statistical_Validation_Set/NaCl_validation_P1.cif'
    nacl, nacl_model = model(source_nacl)
    figures = []

    def crystal(name, crystal_model, settings, source, status):
        scene = build_scene(crystal_model, settings)
        render_crystal(scene, 2400, 1800).save(out/name, dpi=(300, 300), optimize=True)
        document = scene_document(scene, 2400, 1800, 300)
        document.update(software=APP_RELEASE, source_status=status)
        dump(out/Path(name).with_suffix('.crystal.json'), document)
        figures.append(dict(file=name, source=source, status=status,
            geometry=document['geometry'], settings=asdict(settings)))
        print(name, document['geometry'], flush=True)
        return scene

    crystal('nacl-crystal.png', nacl_model,
        RenderSettings(style=SCIENTIFIC_STYLE, repeats=(1, 1, 1),
            center_element='Cl', complete_boundaries=False,
            azimuth=33, elevation=23, atom_scale=1.05, bond_radius=.06,
            polyhedron_opacity=.16, polyhedron_edges=True,
            cell_grid=False, caption=True, axes=True), source_nacl,
        'Idealized NaCl rock-salt fixture; reference geometry, not a refinement result.')

    # Preserve the exact PNG explicitly selected by Ali Afruz, not a newly
    # generated substitute. Its portable scene retains the saved fitted cell.
    selected = ROOT/'docs/figures/nacl-polyhedra.png'
    selected_scene = ROOT/'docs/figures/nacl-polyhedra.crystal.json'
    expected_sha = '2d14c7212457546defbeda21ff5ec994d0c14ec1c35cf1d247803a8a3a015eaa'
    if hashlib.sha256(selected.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('The user-selected NaCl PNG changed; do not silently replace it.')
    for path in (selected, selected_scene):
        if path.resolve() != (out/path.name).resolve():
            shutil.copyfile(path, out/path.name)
    selected_doc = json.loads(selected_scene.read_text(encoding='utf-8'))
    selected_model, selected_settings = load_scene_document(selected_scene)
    selected_geometry = build_scene(selected_model, selected_settings)
    if len(selected_geometry.positions) != selected_doc['geometry']['displayed_atoms']:
        raise ValueError('Selected NaCl scene no longer matches its recorded geometry.')
    figures.append(dict(file='nacl-polyhedra.png',source='docs/figures/nacl-polyhedra.crystal.json',
        status='User-selected original NaCl Rietveld visualization from 23.5.0. '
        'Saved synthetic fitted cell; CIF fractional coordinates/occupancies remain fixed. '
        'Distinct run from the Le Bail plot. PNG preserved byte-for-byte.',
        original_sha256=expected_sha, geometry=selected_doc['geometry'],
        settings=selected_doc['settings']))

    # A typographic banner with an exact, transparently rendered crystal.
    width, height = 2400, 1000
    yy, xx = np.mgrid[:height, :width]
    mix = np.exp(-((xx-1900)/900)**2-((yy-400)/800)**2)[...,None]
    base = np.array([8, 16, 35.])[None,None,:]
    glow = np.array([32, 31, 63.])[None,None,:]
    hero = Image.fromarray(np.uint8(np.clip(base+mix*glow,0,255))).convert('RGBA')
    scene = build_scene(nacl_model, RenderSettings(style='Cinematic • midnight', repeats=(2,2,2),
        azimuth=31, elevation=20, center_element='Cl', complete_boundaries=False,
        polyhedra=False, cell_edges=False, cell_grid=False, axes=False, caption=False,
        transparent=True, atom_scale=1.16, bond_radius=.06))
    crystal_image = render_crystal(scene, 1160, 940)
    hero.alpha_composite(crystal_image, (1230,25))
    draw = ImageDraw.Draw(hero)
    draw.rounded_rectangle((115,110,185,120), radius=5, fill='#5de0ce')
    draw.text((115,165),'DIFFRUX',font=font(182,True),fill='#f4f7ff')
    draw.text((124,390),'CRYSTALS. PROFILES. EVIDENCE.',font=font(32,True),fill='#6fe3d3')
    draw.text((124,485),'Powder X-ray diffraction',font=font(49),fill='#e3e8f4')
    draw.text((124,555),'from measured patterns to crystal figures.',font=font(36),fill='#aab9d0')
    for x, label in ((124,'REFINE'),(400,'COMPARE'),(710,'VISUALIZE')):
        draw.rounded_rectangle((x,685,x+240,752),radius=18,fill='#14293c',outline='#345a6a',width=2)
        draw.text((x+120,716),label,font=font(25,True),fill='#cae8ed',anchor='mm')
    draw.text((124,865),f'{APP_RELEASE}  /  OPEN-SOURCE SCIENTIFIC WORKBENCH',font=font(23),fill='#859bb8')
    draw.text((1830,953),'Idealized NaCl · CIF-derived geometry',font=font(21),fill='#aab9d0',anchor='mm')
    hero.convert('RGB').save(out/'diffrux-hero.png',dpi=(200,200),optimize=True)
    figures.append(dict(file='diffrux-hero.png', source=source_nacl,
        status='DIFFRUX branding; idealized NaCl, exact renderer geometry; atom display sizes illustrative.'))

    data_path = ROOT/'Afruz_PXRD_Statistical_Validation_Set/01_NaCl_raw_counts_with_sigma.txt'
    data = np.loadtxt(data_path, skiprows=8)
    x, y, sigma = data.T
    reference = reference_from_cif_pattern(nacl, calculate_powder_pattern(nacl,1.5406,4,80),1.5406)
    fit = refine_whole_pattern(x,y,[WholePatternPhaseSpec(reference,refine_cell=True)],
        mode='Le Bail-style extraction', observed_sigma=sigma, intensity_provenance='raw_counts',
        profile_model='TCH pseudo-Voigt', refine_lorentzian_width=True,
        extraction_cycles=12, use_staged_refinement=True,
        maximum_nonlinear_evaluations=120, maximum_optimization_points=1800)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig, (ax, residual) = plt.subplots(2,1,figsize=(12,6.5),sharex=True,
        gridspec_kw=dict(height_ratios=[3.4,1],hspace=.12))
    fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.85)
    fig.text(.09,.94,'DIFFRUX  /  Le Bail whole-pattern extraction',fontsize=19,weight='bold',color='#17243c')
    fig.text(.09,.89,'Bundled synthetic NaCl · Cu Kα₁, λ = 1.5406 Å · not an experimental validation standard',fontsize=11,color='#53637a')
    ax.plot(x,y/1000,'.',color='#9ca8b6',ms=2,label='Synthetic observations')
    ax.plot(x,np.asarray(fit['calculated_y'])/1000,color='#6941b8',lw=1.2,label='Le Bail calculated')
    ax.plot(x,np.asarray(fit['background_y'])/1000,color='#de9d31',lw=1,label='Fitted background')
    ax.set_ylabel('Intensity / 10³ counts')
    ax.legend(frameon=False,loc='upper right')
    ax.text(.98,.65,f"Rwp = {fit['rwp_percent']:.2f}%\n12 extraction cycles",ha='right',va='top',transform=ax.transAxes,color='#53637a')
    residual.plot(x,np.asarray(fit['difference_y'])/1000,color='#168e91',lw=.7)
    residual.axhline(0,color='#9ca8b6',lw=.6)
    residual.set(xlabel='2θ / degrees',ylabel='Difference\n/ 10³ counts',xlim=(4,80))
    for axis in (ax,residual):
        axis.grid(axis='y',color='#e4e9ef',lw=.5)
        axis.set_axisbelow(True)
    fig.savefig(out/'nacl-le-bail.png',dpi=240,facecolor='white')
    fig.savefig(out/'nacl-le-bail.svg',facecolor='white',metadata={'Creator':APP_RELEASE,'Date':None})
    plt.close(fig)
    np.savetxt(out/'nacl-le-bail-profile.txt',
        np.column_stack([fit[k] for k in ('observed_x','observed_y','calculated_y','background_y','difference_y')]),
        delimiter='\t',fmt='%.9g',header='two_theta_deg\tobserved_counts\tcalculated_counts\tbackground_counts\tdifference_counts')
    figures.append(dict(file='nacl-le-bail.png',source=str(data_path.relative_to(ROOT)).replace('\\','/'),
        status='Synthetic software regression demonstration; not experimental ground truth.',
        rwp_percent=fit['rwp_percent'], refined_cell=fit['phases'][0]['refined_cell'],
        extraction=fit['extraction_diagnostics']))
    for row in figures:
        row['sha256'] = hashlib.sha256((out/row['file']).read_bytes()).hexdigest()
        row['source_sha256'] = hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()
    dump(out/'gallery-manifest.json',dict(software=APP_RELEASE,license='MIT',
        conventions='CIF-derived coordinates; orthographic projection; illustrative radii/colors; no AI-generated scientific geometry.',
        figures=figures))
    print('Gallery complete:',out,flush=True)


if __name__ == '__main__':
    main()
