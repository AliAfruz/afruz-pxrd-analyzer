"""Public identity, backwards compatibility and scientific figure provenance."""
import hashlib
import json
import re
import tomllib

import numpy as np
import pytest
from PIL import Image

from afruz_pxrd.version import APP_NAME, APP_VERSION, APP_RELEASE
from afruz_pxrd.plot_data_export import _core_xml, _app_xml
from afruz_pxrd.text_export import write_table_txt
from afruz_pxrd.crystal_scene import load_scene_document, build_scene
from afruz_pxrd.gsasii_settings_dialog import SETTINGS_APPLICATION


def test_diffrux_name_launchers_and_metadata_keep_legacy_compatibility(project_root):
    assert APP_NAME == 'DIFFRUX'
    assert APP_VERSION == '23.6.0'
    package = tomllib.loads((project_root/'pyproject.toml').read_text(encoding='utf-8'))['project']
    assert package['name'] == 'afruz-pxrd-analyzer'
    assert package['gui-scripts']['diffrux'] == package['gui-scripts']['afruz-pxrd'] == 'afruz_pxrd.app:run'
    assert SETTINGS_APPLICATION == 'AfruzPXRDAnalyzer'
    zenodo = json.loads((project_root/'.zenodo.json').read_text())
    assert zenodo['title'].startswith('DIFFRUX:')
    assert zenodo['version'] == APP_VERSION
    cff = (project_root/'CITATION.cff').read_text(encoding='utf-8')
    assert f'version: {APP_VERSION}' in cff
    assert '10.5281/zenodo.23119993' not in cff  # Not the DOI of this newer source snapshot.
    assert zenodo['creators'][0]['name'] == 'Afruz, Ali'
    assert zenodo['creators'][1]['name'] == 'Kaffash Jamshid, Maryam'


def test_export_headers_use_diffrux(tmp_path):
    assert '<dc:creator>DIFFRUX</dc:creator>' in _core_xml()
    assert '<Application>DIFFRUX</Application>' in _app_xml(1)
    target = tmp_path/'table.txt'
    write_table_txt(target, [{'x':1,'y':2}], columns=['x','y'])
    assert '# DIFFRUX clean text export' in target.read_text(encoding='utf-8-sig')


def test_public_gallery_hashes_and_relative_sources(project_root):
    folder = project_root/'docs/figures'
    manifest = json.loads((folder/'gallery-manifest.json').read_text(encoding='utf-8'))
    assert manifest['software'] == APP_RELEASE
    assert len(manifest['figures']) == 4
    assert not (folder/'hypothetical-pore.png').exists()
    for row in manifest['figures']:
        assert hashlib.sha256((folder/row['file']).read_bytes()).hexdigest() == row['sha256']
        assert hashlib.sha256((project_root/row['source']).read_bytes()).hexdigest() == row['source_sha256']
        assert not re.search(r'[A-Za-z]:[\\/]', row['source'])
        with Image.open(folder/row['file']) as picture:
            assert picture.width >= 2000 and picture.height >= 1000
    for link in re.findall(r'\]\((docs/figures/[^)]+)\)', (project_root/'README.md').read_text(encoding='utf-8')):
        assert (project_root/link).is_file()


@pytest.mark.parametrize('name', ['nacl-crystal','nacl-polyhedra'])
def test_gallery_scenes_reopen_with_recorded_legacy_schema_and_geometry(project_root, name):
    path = project_root/f'docs/figures/{name}.crystal.json'
    raw = path.read_text(encoding='utf-8')
    assert not re.search(r'[A-Za-z]:[\\/]', raw)
    document = json.loads(raw)
    assert document['schema'] == 'afruz.crystal-scene.v1'
    model, settings = load_scene_document(path)
    scene = build_scene(model, settings)
    assert len(scene.positions) == document['geometry']['displayed_atoms']
    assert len(scene.bonds) == document['geometry']['displayed_contacts']
    assert np.all(np.isfinite(scene.positions))
    assert not scene.pore_radii
    if name == 'nacl-polyhedra':
        assert model['provenance']['kind'] == 'rietveld_result'
        assert model['provenance']['atoms_fixed'] is True
        assert hashlib.sha256((path.parent/'nacl-polyhedra.png').read_bytes()).hexdigest() == '2d14c7212457546defbeda21ff5ec994d0c14ec1c35cf1d247803a8a3a015eaa'


def test_public_plot_metric_matches_full_unmasked_exported_profile(project_root):
    manifest = json.loads((project_root/'docs/figures/gallery-manifest.json').read_text())
    fit = next(row for row in manifest['figures'] if row['file'] == 'nacl-le-bail.png')
    profile = np.loadtxt(project_root/'docs/figures/nacl-le-bail-profile.txt')
    original = np.loadtxt(project_root/fit['source'], skiprows=8)
    np.testing.assert_allclose(profile[:,:2], original[:,:2], rtol=1e-8)
    np.testing.assert_allclose(profile[:,1]-profile[:,2], profile[:,4], atol=1e-4)
    weights = 1/original[:,2]**2
    rwp = 100*np.sqrt(np.sum(weights*(profile[:,1]-profile[:,2])**2)/np.sum(weights*profile[:,1]**2))
    assert rwp == pytest.approx(fit['rwp_percent'], abs=2e-6)
    assert fit['extraction']['algorithm_revision'] == 2
