import json
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from catalyst.tools import (
    add_adsorbate_to_slab_text,
    choose_adsorption_site_text,
    choose_slab_text,
    create_slab_from_structure_text,
    enumerate_slabs_text,
    get_adsorption_sites_text,
    get_bulk_polymorphs_data,
    get_mp_thermo_data,
    get_structure_from_mp_text,
    sort_and_get_first_from_json,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pymatgen.core import Molecule, Structure

from corral.utils.code_tools import execute_python_code

MOCK_DATA_DIR = Path(Path(__file__).parent / "mock_data")


def skip_if_no_api_key():
    """Skip test if MP_API_KEY is not available"""
    return pytest.mark.skipif(
        not os.getenv("MP_API_KEY"), reason="MP_API_KEY not available in environment"
    )


def rehydrate_docs(docs_as_dicts):
    """Converts a list of stored doc dicts into a list of mock MP documents.

    Copies each source dict so the cached pickle payload is never mutated
    (fixtures are reused across tests). Fields are normalised to what the tools
    expect: ``structure`` is a pymatgen ``Structure`` (already hydrated in the
    fixtures, so only converted when still a dict), ``symmetry`` is exposed as
    an object with a ``.symbol`` attribute, and ``decomposes_to`` entries as
    objects with ``.material_id`` / ``.formula`` / ``.amount``.
    """
    rehydrated = []
    for source in docs_as_dicts:
        doc_dict = dict(source)
        mock_obj = MagicMock()
        structure = doc_dict.get("structure")
        if isinstance(structure, dict):
            doc_dict["structure"] = Structure.from_dict(structure)
        symmetry = doc_dict.get("symmetry")
        if isinstance(symmetry, dict):
            symmetry_obj = MagicMock()
            symmetry_obj.configure_mock(**symmetry)
            doc_dict["symmetry"] = symmetry_obj
        if doc_dict.get("decomposes_to"):
            decomp_list = []
            for item_dict in doc_dict["decomposes_to"]:
                decomp_obj = MagicMock()
                decomp_obj.configure_mock(**item_dict)
                decomp_list.append(decomp_obj)
            doc_dict["decomposes_to"] = decomp_list
        mock_obj.configure_mock(**doc_dict)
        rehydrated.append(mock_obj)
    return rehydrated


@pytest.fixture()
def mock_mp_rester(mocker):
    """Mocks the MPRester class to return rehydrated mock data.

    Also guarantees an ``MP_API_KEY`` is present so the tools' key check passes;
    the Rester itself is mocked, so no live request is made and the value is
    irrelevant.
    """
    mocker.patch.dict(os.environ, {"MP_API_KEY": "test-key"})
    with (MOCK_DATA_DIR / "si_summary_docs.pkl").open("rb") as f:
        si_summary_dicts = pickle.load(f)
    with (MOCK_DATA_DIR / "tio2_polymorph_docs.pkl").open("rb") as f:
        tio2_polymorph_dicts = pickle.load(f)
    with (MOCK_DATA_DIR / "si_thermo_docs.pkl").open("rb") as f:
        si_thermo_dicts = pickle.load(f)

    def mock_summary_search(**kwargs):
        if "mp-149" in kwargs.get("material_ids", []) or "mp-149" in kwargs.get(
            "formula", ""
        ):
            return rehydrate_docs(si_summary_dicts)
        if "TiO2" in kwargs.get("formula", ""):
            return rehydrate_docs(tio2_polymorph_dicts)
        return []

    def mock_thermo_search(**kwargs):
        if "mp-149" in kwargs.get("material_ids", []):
            return rehydrate_docs(si_thermo_dicts)
        return []

    mock_mpr_instance = MagicMock()
    mock_mpr_instance.materials.summary.search.side_effect = mock_summary_search
    mock_mpr_instance.thermo.search.side_effect = mock_thermo_search

    mock_mpr_class = MagicMock()
    mock_mpr_class.return_value.__enter__.return_value = mock_mpr_instance

    mocker.patch("catalyst.tools.MPRester", mock_mpr_class)


@pytest.fixture(scope="module")
def silicon_cif():
    with (MOCK_DATA_DIR / "mp-149.cif").open("r") as f:
        return f.read()


@pytest.fixture(scope="module")
def co_cif():
    co_mol = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.1]])
    struct = co_mol.get_boxed_structure(10, 10, 10)
    return struct.to(fmt="cif")


@skip_if_no_api_key()
def test_get_structure_from_mp_text():
    cif_str = get_structure_from_mp_text.execute(mp_id="mp-149")
    assert isinstance(cif_str, str)
    assert "data_Si" in cif_str
    struct = Structure.from_str(cif_str, fmt="cif")
    assert struct.composition.reduced_formula == "Si"


@settings(deadline=5000, suppress_health_check=[HealthCheck.too_slow])
@given(
    miller_index=st.tuples(
        st.integers(0, 2), st.integers(0, 2), st.integers(0, 2)
    ).filter(lambda x: x != (0, 0, 0)),
    min_slab_size=st.integers(8, 15),
    min_vacuum_size=st.integers(5, 10),
)
def test_create_slab_from_structure_text(
    silicon_cif, miller_index, min_slab_size, min_vacuum_size
):
    slab_cif = create_slab_from_structure_text.execute(
        structure_cif=silicon_cif,
        miller_index=miller_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
    )
    assert isinstance(slab_cif, str)
    slab_struct = Structure.from_str(slab_cif, fmt="cif")
    assert slab_struct.composition.reduced_formula == "Si"


def test_enumerate_slabs_text(silicon_cif):
    slabs_json = enumerate_slabs_text.execute(
        bulk_cif=silicon_cif, miller_index=(1, 1, 1)
    )
    assert isinstance(slabs_json, str)
    slabs_dict = json.loads(slabs_json)
    assert "slab_0" in slabs_dict
    Structure.from_str(slabs_dict["slab_0"], fmt="cif")


def test_choose_slab_text(silicon_cif):
    slabs_json = enumerate_slabs_text.execute(
        bulk_cif=silicon_cif, miller_index=(1, 1, 1)
    )
    slabs_dict = json.loads(slabs_json)

    chosen_cif = choose_slab_text.execute(slabs_json=slabs_json, index=0)
    assert chosen_cif == slabs_dict["slab_0"]

    with pytest.raises(ValueError, match="Slab index 100 not found."):
        choose_slab_text.execute(slabs_json=slabs_json, index=100)


def test_adsorption_workflow(silicon_cif, co_cif):
    slab_cif = create_slab_from_structure_text.execute(
        structure_cif=silicon_cif, miller_index=(1, 0, 0)
    )
    sites_json = get_adsorption_sites_text.execute(slab_cif=slab_cif)
    sites_dict = json.loads(sites_json)
    site_type = "top" if sites_dict.get("top") else next(iter(sites_dict.keys()))
    site_coords = choose_adsorption_site_text.execute(
        adsorption_sites_json=sites_json, site_type=site_type, index=0
    )
    site_coords_list = json.loads(site_coords)
    combined_cif = add_adsorbate_to_slab_text.execute(
        slab_cif=slab_cif, adsorbate_cif=co_cif, site=site_coords_list
    )

    slab_struct = Structure.from_str(slab_cif, fmt="cif")
    adsorbate_struct = Structure.from_str(co_cif, fmt="cif")
    combined_struct = Structure.from_str(combined_cif, fmt="cif")
    assert len(combined_struct) == len(slab_struct) + len(adsorbate_struct)


def test_get_bulk_polymorphs_data(mock_mp_rester):
    polymorphs_json = get_bulk_polymorphs_data.execute(composition="TiO2")
    data = json.loads(polymorphs_json)
    assert isinstance(data, list)
    assert len(data) > 1
    assert data[0]["energy_above_hull"] <= data[1]["energy_above_hull"]


def test_sort_and_get_first_from_json(mock_mp_rester):
    polymorphs_json = get_bulk_polymorphs_data.execute(composition="TiO2")
    stable_id = sort_and_get_first_from_json.execute(
        json_data=polymorphs_json,
        sort_key="energy_above_hull",
        return_key="material_id",
    )
    data = json.loads(polymorphs_json)
    assert stable_id == data[0]["material_id"]


def test_execute_python_code():
    code = "result = 5 * 10"
    output_json = execute_python_code.execute(python_code=code)
    output = json.loads(output_json)
    assert output["success"]
    assert output["execution_result"]["result"] == 50


def test_get_mp_thermo_data(mock_mp_rester):
    thermo_json = get_mp_thermo_data.execute(material_id="mp-149")
    data = json.loads(thermo_json)
    assert isinstance(data, list)
    assert data[0]["material_id"] == "mp-149"
    assert "formation_energy_per_atom" in data[0]
