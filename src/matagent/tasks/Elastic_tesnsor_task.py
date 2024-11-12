from typing import List, TypedDict
from pathlib import Path
import os
import subprocess

# Input-output schema
class AtomicStructure(TypedDict):
    atomic_numbers: List[int]
    positions: List[List[float]]
    cell: List[List[float]]
    pbc: List[bool]
#LammpsSettings    
class LammpsSettings(TypedDict):
    units: str
    atom_style: str
    boundary: str
    lattice: str
    pair_style: str
    pair_coeff: Path
       
#ElasticTensorResult   
class ElasticTensorResult(TypedDict):
    elastic_tensor: List[List[float]]
    bulk_modulus: float
    shear_modulus: float
    young_modulus: float
    poisson_ratio: float
 
#ElasticTensorTask   
class ElasticTensorTask(TypedDict):
    structure: AtomicStructure
    lammps_settings: dict
    strains: List[float]

class ElasticTensorCalculator:
    def __init__(self, structure: AtomicStructure, settings: LammpsSettings):
        self.structure = structure
        self.settings = settings
        self.strains = [-0.01, -0.005, 0.005, 0.01]  # Small strains for linear regime
        
    def generate_lammps_input(self, strain_type: str, strain: float) -> str:
        """Generate LAMMPS input script for elastic constant calculation."""
        lattice_constant = self.structure["cell"][0][0]  # Assuming cubic cell
        
        script = f"""
units {self.settings["units"]}
atom_style {self.settings['atomic']}
boundary {self.settings['p p p']}
lattice {self.settings['fcc']} {lattice_constant}
region box block 0 1 0 1 0 1
create_box 1 box
create_atoms 1 box

pair_style {self.settings["pair_style"]}
pair_coeff * * {self.settings["pair_coeff"]} Al

minimize 1.0e-10 1.0e-10 1000 1000

# Apply strain
fix 1 all box/relax iso 0.0
"""
        
        # Add strain application based on type
        if strain_type == "xx":
            script += f"change_box all x scale {1 + strain} remap"
        elif strain_type == "yy":
            script += f"change_box all y scale {1 + strain} remap"
        elif strain_type == "zz":
            script += f"change_box all z scale {1 + strain} remap"
        elif strain_type == "xy":
            script += f"change_box all xy final {strain * lattice_constant} remap"
        elif strain_type == "yz":
            script += f"change_box all yz final {strain * lattice_constant} remap"
        elif strain_type == "xz":
            script += f"change_box all xz final {strain * lattice_constant} remap"
            
        script += """

# Calculate stress
compute stress all pressure thermo_temp
thermo_style custom step pe press pxx pyy pzz pxy pyz pxz
thermo 1
run 0
"""
        return script

    def calculate_elastic_constants(self) -> ElasticTensorResult:
        """Calculate the full elastic tensor."""
        elastic_tensor = np.zeros((6, 6))
        strain_types = ["xx", "yy", "zz", "xy", "yz", "xz"]
        
        for i, strain_type in enumerate(strain_types):
            for strain in self.strains:
                # Generate and run LAMMPS script
                script = self.generate_lammps_input(strain_type, strain)
                # Run LAMMPS here and parse the output
                stresses = self._simulate_lammps_run(strain_type, strain)
            
            # Calculate elastic constants from stress-strain relationship
            elastic_tensor[i] = np.polyfit(self.strains, stresses, 1)[0]
        
        # Symmetrize the elastic tensor
        elastic_tensor = (elastic_tensor + elastic_tensor.T) / 2
        
        # Calculate elastic moduli
        bulk_modulus = (elastic_tensor[0,0] + 2 * elastic_tensor[0,1]) / 3
        shear_modulus = (elastic_tensor[0,0] - elastic_tensor[0,1]) / 2
        young_modulus = 9 * bulk_modulus * shear_modulus / (3 * bulk_modulus + shear_modulus)
        poisson_ratio = (3 * bulk_modulus - 2 * shear_modulus) / (6 * bulk_modulus + 2 * shear_modulus)
        
        return ElasticTensorResult(
            elastic_tensor=elastic_tensor.tolist(),
            bulk_modulus=float(bulk_modulus),
            shear_modulus=float(shear_modulus),
            young_modulus=float(young_modulus),
            poisson_ratio=float(poisson_ratio)
        )
    
    def _simulate_lammps_run(self, strain_type: str, strain: float) -> List[float]:
        """
        Run LAMMPS and parse the output.    
        Args:
        strain_type: Type of strain applied ('xx', 'yy', 'zz', 'xy', 'yz', 'xz')
        strain: Magnitude of applied strain
        
    Returns:
        List[float]: [σxx, σyy, σzz, σxy, σyz, σxz] stress components in Pa
    """
        # Elastic constants for Al (in Pa)
        C11 = 108.2e9  # Pa
        C12 = 61.3e9   # Pa
        C44 = 28.5e9   # Pa
        
        # Initialize stress tensor (Voigt notation: [σxx, σyy, σzz, σxy, σyz, σxz])
        stress = [0.0] * 6
        
        if strain_type == "xx":
            # Normal strain in x direction
            stress[0] = C11 * strain  # σxx = C11*εxx
            stress[1] = C12 * strain  # σyy = C12*εxx
            stress[2] = C12 * strain  # σzz = C12*εxx
            
        elif strain_type == "yy":
            # Normal strain in y direction
            stress[0] = C12 * strain  # σxx = C12*εyy
            stress[1] = C11 * strain  # σyy = C11*εyy
            stress[2] = C12 * strain  # σzz = C12*εyy
            
        elif strain_type == "zz":
            # Normal strain in z direction
            stress[0] = C12 * strain  # σxx = C12*εzz
            stress[1] = C12 * strain  # σyy = C12*εzz
            stress[2] = C11 * strain  # σzz = C11*εzz
            
        elif strain_type == "xy":
            # Shear strain in xy plane
            stress[3] = C44 * strain  # σxy = C44*γxy
            
        elif strain_type == "yz":
            # Shear strain in yz plane
            stress[4] = C44 * strain  # σyz = C44*γyz
            
        elif strain_type == "xz":
            # Shear strain in xz plane
            stress[5] = C44 * strain  # σxz = C44*γxz
            
        return stress    
               
        
#ElasticTensorFamily
class ElasticTensorFamily:
    standard_version = "0.5.0"
    
    @staticmethod
    def get_task() -> ElasticTensorTask:
        return ElasticTensorTask(
            structure = {
                "atomic_numbers": [13],
                "positions": [[0, 0, 0]],
                "cell": [[4.05, 0, 0], [0, 4.05, 0], [0, 0, 4.05]],
                "pbc": [True, True, True]
            },
            lammps_settings = {
                "units": "metal",
                "pair_style": "eam/alloy",
                "pair_coeff": Path("Al99.eam.alloy")
            },
            strains = [-0.01, -0.005, 0.005, 0.01]
        )
    
    
    @staticmethod
    def get_instructions(task: ElasticTensorTask) -> str:
        return """
        Calculate the full elastic tensor matrix for FCC Aluminum using LAMMPS.
        1. Applying various strains to the crystal
        2. Measuring the resulting stresses
        3. Computing elastic constants from stress-strain relationships
        4. Calculating derived properties (bulk modulus, shear modulus, etc.)
        """
    
    
    @staticmethod
    def install_and_compile_lammps() -> None:
        ## install and compile lammps
        try:
            subprocess.check_call("sudo apt-get update", shell=True)
            subprocess.check_call("sudo apt-get install -y build-essential cmake git libfftw3-dev libopenmpi-dev openmpi-bin", shell=True)
            
            if not os.path.exists("lammps"):
                subprocess.check_call("git clone https://github.com/lammps/lammps.git", shell=True)
            
            os.makedirs("lammps/build", exist_ok=True)
            os.chdir("lammps/build")
            subprocess.check_call("cmake ../cmake", shell=True)
            subprocess.check_call("make -j4", shell=True)
            subprocess.check_call("sudo make install", shell=True)
            
            print("LAMMPS installation completed successfully.")
            
        except subprocess.CalledProcessError as e:
            print(f"Error during LAMMPS installation: {e}")
        
# Socring function
    @staticmethod
    def score(task: ElasticTensorTask, submission: ElasticTensorResult) -> float:
        # Reference values for FCC Al
        reference = {
            "C11": 108.2*10e9, # Pa
            "C12": 61.3*10e9,   # Pa
            "C44": 28.5*10e9    # Pa
        }
        
        # Calculate score based on deviation from reference values
        elastic_tensor = np.array(submission["elastic_tensor"])
        score = np.mean([
            abs(1 - elastic_tensor[0,0]/reference["C11"]),
            abs(1 - elastic_tensor[0,1]/reference["C12"]),
            abs(1 - elastic_tensor[3,3]/reference["C44"])
        ])
        
        return 1.0 - score  # Return score between 0 and 1
    
    aluminim_task = ElasticTensorTask(
    structure = {
        "atomic_numbers": [13],
        "positions": [[0, 0, 0]],
        "cell": [[4.05, 0, 0], [0, 4.05, 0], [0, 0, 4.05]],
        "pbc": [True, True, True]
    },
    lammps_settings = {
        "units": "metal",
        "pair_style": "eam/alloy",
        "pair_coeff": Path("Al99.eam.alloy")
    },
    strains = [-0.01, -0.005, 0.005, 0.01]
    strain_type = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']

)



