import unittest
from unittest.mock import patch, MagicMock
import json
from typing import Callable
import sys
import pathlib
from md_simulations.score import check_numerical, check_potential_file, check_structure
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

class TestCheckNumerical(unittest.TestCase):

    def setUp(self):
        self.target = 10.0
        self.tolerance = 0.1  # 10%
        self.score_fn = check_numerical(self.target, self.tolerance)

    def test_exact_value(self):
        self.assertEqual(self.score_fn("10.0"), 1.0)

    def test_within_tolerance(self):
        self.assertEqual(self.score_fn("9.1"), 1.0)
        self.assertEqual(self.score_fn("10.9"), 1.0)

    def test_outside_tolerance(self):
        self.assertEqual(self.score_fn("8.9"), 0.0)
        self.assertEqual(self.score_fn("11.1"), 0.0)

    def test_json_answer_field(self):
        self.assertEqual(self.score_fn(json.dumps({"answer": 10.0})), 1.0)
        self.assertEqual(self.score_fn(json.dumps({"C11": 10.0})), 1.0)
        self.assertEqual(self.score_fn(json.dumps({"C12": 10.0})), 1.0)
        self.assertEqual(self.score_fn(json.dumps({"C13": 10.0})), 1.0)
        self.assertEqual(self.score_fn(json.dumps({"C23": 10.0})), 1.0)
        self.assertEqual(self.score_fn(json.dumps({"C22": 10.0})), 1.0)
        self.assertEqual(self.score_fn(json.dumps({"C33": 10.0})), 1.0)

    def test_json_bulk_energy(self):
        self.assertEqual(self.score_fn(json.dumps({"BULK ENERGY": 10.0})), 1.0)
        self.assertEqual(self.score_fn(json.dumps({"BULK ENERGY": 10.0, "path to bulk structure" : 1.0})), 1.0)
        self.assertEqual(self.score_fn("{\"BULK ENERGY\" : 10.0, \"path to bulk structure\" : 1.0}"), 1.0) 
        self.assertEqual(self.score_fn("{\"BULK ENERGY\" : \"10.0\", \"path to bulk structure\" : \"1.0\"}"), 1.0) 
        
        

    def test_json_slab_energy(self):
        self.assertEqual(self.score_fn(json.dumps({"SLAB ENERGY": 10.0})), 1.0)
        self.assertEqual(self.score_fn(json.dumps({"SLAB ENERGY": 10.0, "path to slab structure" : 1.0})), 1.0)

    def test_string_with_extra_text(self):
        self.assertEqual(self.score_fn("The answer is 10.0."), 1.0)
        self.assertEqual(self.score_fn("The total: 9.5."), 1.0)

    def test_invalid_json(self):
        self.assertEqual(self.score_fn('{"invalid":'), 0.0)

    def test_non_numeric_string(self):
        self.assertEqual(self.score_fn("Ten"), 0.0)

    def test_empty_string(self):
        self.assertEqual(self.score_fn(""), 0.0)

    def test_none_input(self):
        self.assertEqual(self.score_fn(None), 0.0)

    def test_numeric_input_direct(self):
        self.assertEqual(self.score_fn(10.0), 1.0)
        self.assertEqual(self.score_fn(9.0), 1.0)
        self.assertEqual(self.score_fn(8.0), 0.0)

    def test_dict_without_answer(self):
        self.assertEqual(self.score_fn({"not_answer": 10.0}), 0.0)

class TestCheckPotentialFile(unittest.TestCase):

    def setUp(self):
        self.target_filename = "potential_file.eam"
        self.score_fn = check_potential_file(self.target_filename)

    def test_exact_match(self):
        self.assertEqual(self.score_fn("potential_file.eam"), 1.0)

    def test_full_path_match(self):
        self.assertEqual(self.score_fn("/some/path/to/potential_file.eam"), 1.0)

    def test_different_filename(self):
        self.assertEqual(self.score_fn("wrong_file.eam"), 0.0)

    def test_similar_name(self):
        self.assertEqual(self.score_fn("potential_file.eam.txt"), 0.0)

    def test_empty_string(self):
        self.assertEqual(self.score_fn(""), 0.0)

    def test_none_input(self):
        self.assertEqual(self.score_fn(None), 0.0)

    def test_invalid_type_input(self):
        self.assertEqual(self.score_fn(12345), 0.0)

    def test_directory_path_only(self):
        self.assertEqual(self.score_fn("/some/path/"), 0.0)

class TestLocalCheckStructure(unittest.TestCase):
    def setUp(self):
        self.target = "../ground_truth/energy_minimisation/Aluminum/Al_minimised_structure.dat"
        self.atom_style = "atomic"
        self.use_modal = False
        self.score_fn = check_structure(self.target, self.atom_style, self.use_modal)

    def test_same_file(self):
        self.assertEqual(self.score_fn("../ground_truth/energy_minimisation/Aluminum/Al_minimised_structure.dat"), 1.0)
        self.assertEqual(self.score_fn("../ground_truth/npt/Aluminum/minimized_aluminum_structure.dat"), 1.0)
        self.assertEqual(self.score_fn("../ground_truth/energy_minimisation/Silicon/Si_minimised_structure.dat"), 0.0)

class TestLocalCheckStructureFull(unittest.TestCase):
    def setUp(self):
        self.target = "../ground_truth/energy_minimisation/Silicon/Si_minimised_structure.data"
        self.atom_style = "full"
        self.use_modal = False
        self.score_fn = check_structure(self.target, self.atom_style, self.use_modal)

    def test_same_file(self):
        self.assertEqual(self.score_fn("../ground_truth/energy_minimisation/Silicon/Si_minimised_structure.data"), 1.0)
        self.assertEqual(self.score_fn("../ground_truth/energy_minimisation/SiO2/SiO2_minimised_structure.data"), 0.0)
        # self.assertEqual(self.score_fn("../ground_truth/energy_minimisation/Silicon/Si_minimised_structure.dat"), 0.0)


class TestCheckStructure(unittest.TestCase):
    def setUp(self):
        self.target = "../ground_truth/energy_minimisation/Aluminum/Al_minimised_structure.dat"
        self.atom_style = "atomic"
        self.score_fn = check_structure(self.target, self.atom_style)

    def test_relaxation_file(self):
        self.assertEqual(self.score_fn("/results/23_July_2025/test/gpt_4o/subtask/aluminum_energy_minimisation_subtask_npt_trial_0/relaxed_structure.data"), 1.0)
        self.assertEqual(self.score_fn(None), 0.0)
        self.assertEqual(self.score_fn("/results/23_July_2025/test/gpt_4o/subtask/aluminum_energy_minimisation_subtask_npt_trial_0/relaxed_structure.in"), 0.0)


# class TestNPTCheckStructureLocal(unittest.TestCase):
#     def setUp(self):
#         self.target = "../ground_truth/melting/Aluminum/heatedsystem_3000_reax.dat"
#         self.atom_style = "atomic"
#         self.use_modal = False
#         self.score_fn = check_structure(self.target, self.atom_style, self.use_modal)

#     def test_relaxation_file(self):
#         self.assertEqual(self.score_fn("../ground_truth/melting/Aluminum/heatedsystem_3000_reax.dat"), 1.0)
#         self.assertEqual(self.score_fn("../../md_simulations_2/md_simulations_2/md_tasks/ground_truth/melting/Al/heatedsystem_3000_reax.dat"), 0.0)

if __name__ == "__main__":
    unittest.main()