import json
import os
from abc import (
    ABC,
    abstractmethod,
)
from pathlib import Path

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from llamp_tools import (
    MaterialsDielectric,
    MaterialsElasticity,
    MaterialsElectronic,
    MaterialsMagnetism,
    MaterialsPiezoelectric,
    MaterialsStructureText,
    MaterialsSummary,
    MaterialsSynthesis,
    MaterialsThermo,
)
from loguru import logger
from promptstore import PromptStore

from corral.agents.utils import (
    LiteLLMMessage,
    llm_call,
)
from corral.base import Tool, ToolArgument
from corral.io import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/temp")
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())


class MPAgent(ABC):
    def __init__(
        self,
        max_iterations: int = 3,
        temperature: float = 0.0,
        prompt_store: PromptStore | None = None,
        **kwargs,
    ):
        self.model = "openai/gpt-4o"
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.prompt_store = PromptStore("./prompts")
        if prompt_store:
            self.prompt_store = prompt_store
        else:
            current_dir = Path(__file__).parent
            prompts_path = current_dir / "prompts"
            self.prompt_store = PromptStore(prompts_path)
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def description(self) -> str | None:
        return self.__doc__

    @property
    @abstractmethod
    def tools(self):
        return []

    def as_tool(self) -> Tool:
        def execute(**args):
            try:
                input_question = args.get("input_question")
                logger.info(
                    f"Running {self.__class__.__name__} with input: {input_question}"
                )
                result, _ = self.run_agent(input_question)
                return result
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__}: {e}")
                raise RuntimeError(
                    f"Error in {self.__class__.__name__}: {e}. "
                    "Please decompose the request into multiple smaller requests "
                    "or specify 'limit' in request."
                ) from e

        tool = Tool(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "input_question",
                    "str",
                    "Complete question to ask the assistant agent. Should include all the context and details needed to answer the question holistically.",
                ),
            ],
        )

        tool.execute = execute
        return tool

    def run_agent(self, input_question: str) -> str:
        logger.info(f"Running {self.__class__.__name__} with input: {input_question}")
        env_tools = {"tools": []}

        for tool in self.tools:
            env_tools["tools"].append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "arguments": tool.arguments,
                }
            )

        system_prompt = self.prompt_store.get("fb46ddea-eca3-458a-805f-aa6344780929")
        user_prompt = self.prompt_store.get("9f8a74c1-cd5e-4fc5-b50e-a2eebaffb409")
        tool_names = [tool.name for tool in self.tools]
        system = system_prompt.fill(
            {"tools": json.dumps(str(env_tools)), "tool_names": tool_names}
        )
        user = user_prompt.fill({"input": input_question, "agent_scratchpad": ""})
        messages: list[LiteLLMMessage] = []
        messages.append(LiteLLMMessage(role="system", content=system))
        messages.append(LiteLLMMessage(role="user", content=user))

        for _i in range(self.max_iterations):
            logger.info(f"Iteration {_i + 1} of {self.max_iterations}")
            try:
                response = llm_call(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    **self.kwargs,
                )
                try:
                    if hasattr(response, "content"):
                        response_dict = json.loads(response.content)
                        messages.append(
                            LiteLLMMessage(role="assistant", content=response.content)
                        )
                    else:
                        response_dict = response
                        messages.append(
                            LiteLLMMessage(
                                role="assistant", content=json.dumps(response)
                            )
                        )
                except Exception:
                    response_dict = response
                    messages.append(
                        LiteLLMMessage(role="assistant", content=str(response))
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Error in {self.__class__.__name__} response parsing: {e}\n{response}"
                ) from e

            action = response_dict.get("action", "")
            if action:
                if action == "Final Answer":
                    return response_dict.get("action_input"), messages
                else:
                    function_name = action
                    function_args = response_dict.get("action_input")
                    try:
                        tool = next(
                            (t for t in self.tools if t.name == function_name), None
                        )
                        if tool:
                            function_call = str(tool.execute(**function_args))
                        else:
                            function_call = f"Error: Tool '{function_name}' not found"
                    except Exception as e:
                        function_call = f"Error: {e}"
                    messages.append(
                        LiteLLMMessage(
                            role="assistant",
                            content=f"Observation: {function_call!s}",
                            name=function_name,
                        )
                    )

            else:
                messages.append(
                    LiteLLMMessage(
                        role="user",
                        content="Incorrect output format. Please follow the correct format.",
                    )
                )

        return f"No final answer found after {self.max_iterations}", messages


class MPSummaryExpert(MPAgent):
    r"""[BRIEF] Materials Project summary expert that provides high-level materials properties and information via the Materials Project database. [\BRIEF]

    [DETAILED] This tool is a specialized agent that has access to the Materials Project summary endpoint, which provides calculated and derived materials properties. It can perform filtering on chemical systems, sorting on materials properties, and retrieve high-level information about materials including material IDs that can be used for further detailed queries. The tool uses GPT-4o with iterative reasoning (up to 3 iterations) to process complex materials science queries and provide comprehensive answers. It integrates with the Materials Project database to access a vast collection of computed materials data. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need basic materials properties like formation energy, band gap, crystal structure information, or material IDs
    - When you need to search for materials by chemical formula or composition
    - When you need to filter materials by specific criteria or sort by properties
    - As a starting point for materials research before using more specialized expert tools
    - When you need high-level overview information about materials for further detailed analysis [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have a clear materials science question about general properties, composition, or structure. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with a complete question including all context and details needed. [\CURRENT]
    3. [FOLLOW_UP] Use the material IDs or information obtained to query more specialized expert tools like elasticity, magnetism, or electronic structure experts for detailed properties. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Utilizes OpenAI GPT-4o model with temperature 0.0 for consistent results
    - Runs up to 3 iterations of reasoning to solve complex queries
    - Accesses Materials Project summary endpoint via LLAMP tools integration
    - Can search by formula, elements, material properties, and various filtering criteria
    - Returns structured materials data including material IDs, formulas, and basic properties
    - Handles errors gracefully and suggests decomposing complex requests into smaller ones [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mp_summary_expert("What is the formation energy of NaCl?")`,
        `mp_summary_expert("Find all materials containing Li and O with band gap greater than 2 eV")`,
        `mp_summary_expert("What are the material IDs for different polymorphs of TiO2?")`,
        `mp_summary_expert("List materials with formula Li2CO3 and their basic properties")`,
        `mp_summary_expert("Find oxide materials with density less than 3 g/cm3")`
    ]
    [\SYNTACTICAL]

    Args:
        input_question (str):
                        [BRIEF] Complete question to ask the Materials Project summary expert [\BRIEF]
                        [DETAILED] A comprehensive question that includes all context and details needed to answer the materials science query. Should be specific about what materials properties or information is needed, and include any filtering criteria or constraints. [\DETAILED]
                        [SYNTACTICAL] Format: "Clear, descriptive question about materials properties or composition" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What is the formation energy of NaCl?", "Find all materials containing Li and O with band gap greater than 2 eV", "What are the material IDs for different polymorphs of TiO2?" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Comprehensive answer with materials properties, data, and relevant information from the Materials Project database [\BRIEF]
                        [DETAILED] A detailed response containing the requested materials information, including numerical values, material IDs, chemical formulas, and explanatory context. May include multiple materials if the query resulted in a list, formatted in a clear and readable manner. [\DETAILED]
                        [EXAMPLES] Examples: "NaCl (mp-22862) has a formation energy of -4.02 eV/atom and crystallizes in the Fm-3m space group...", "Found 15 Li-O containing materials with band gap > 2 eV: Li2O (mp-1960, band gap 2.45 eV)..." [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If there are issues with the Materials Project API, authentication, or if the query is too complex/broad and times out. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception occurs when the Materials Project API is unavailable, the API key is invalid, or when queries are too broad and exceed processing limits. The tool suggests decomposing requests into smaller, more specific queries. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try breaking down the request into smaller, more specific questions, or ensure MP_API_KEY environment variable is properly set. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires valid MP_API_KEY environment variable to access Materials Project database
        - Limited to 3 iterations of reasoning, may not solve extremely complex queries
        - Performance depends on Materials Project API availability and response times
        - Large queries may timeout - should be decomposed into smaller requests
        - Only provides summary-level information, not detailed property-specific data
        - Temperature set to 0.0 may limit creative problem-solving approaches
    [/LIMITATIONS]
    """

    @property
    def tools(self):
        return [MaterialsSummary()]


class MPStructureRetriever(MPAgent):
    r"""[BRIEF] Materials Project structure expert that retrieves detailed crystal structures and structural information from Materials Project. [\BRIEF]

    [DETAILED] This specialized agent has access to the Materials Project structure endpoint and can retrieve pymatgen structures as JSON text for structure generation, manipulation, and analysis. It provides detailed crystallographic information including lattice parameters, atomic positions, space groups, and symmetry information. The tool is optimized for structural queries and can handle requests for specific polymorphs, structure types, and crystallographic analysis. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need detailed crystal structure information including atomic positions and lattice parameters
    - When you need pymatgen structure objects for computational materials science workflows
    - When you need crystallographic data like space groups, symmetry operations, or unit cell information
    - For structure comparison, manipulation, or generation tasks
    - When you need to export structures in various formats (CIF, POSCAR, etc.) [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Have the material formula, material ID, or specific structural requirements identified. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with a detailed question about the crystal structure needed. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved structure data for computational analysis, structure manipulation, or input to other materials modeling tools. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Accesses Materials Project structure endpoint via specialized API wrapper
    - Returns pymatgen Structure objects serialized as JSON text
    - Can retrieve structures by formula, material ID, or structural criteria
    - Supports filtering by space group, crystal system, or other structural parameters
    - Provides detailed crystallographic information in machine-readable format
    - Integrates with pymatgen library for structure manipulation capabilities [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mp_structure_retriever("Get the crystal structure of rutile TiO2")`,
        `mp_structure_retriever("Retrieve structure data for material ID mp-2657")`,
        `mp_structure_retriever("Find structures of all SiO2 polymorphs with space group information")`,
        `mp_structure_retriever("Get the unit cell parameters and atomic positions for diamond carbon")`,
        `mp_structure_retriever("Retrieve structures of perovskite materials containing Ba and Ti")`
    ]
    [\SYNTACTICAL]

    Args:
        input_question (str):
                        [BRIEF] Complete question about crystal structures or structural properties [\BRIEF]
                        [DETAILED] A comprehensive question specifying what structural information is needed, including material identification (formula, material ID) and specific structural details required (atomic positions, lattice parameters, space group, etc.). [\DETAILED]
                        [SYNTACTICAL] Format: "Specific question about crystal structure, lattice, or atomic arrangement" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "Get the crystal structure of rutile TiO2", "Retrieve structure data for material ID mp-2657", "Find structures of all SiO2 polymorphs" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Detailed structural information including pymatgen structure data, crystallographic parameters, and atomic arrangements [\BRIEF]
                        [DETAILED] Comprehensive structural data including JSON-serialized pymatgen structures, lattice parameters, atomic positions, space group information, symmetry data, and other crystallographic details formatted for computational use. [\DETAILED]
                        [EXAMPLES] Examples: "TiO2 rutile structure (mp-2657): Space group P42/mnm, lattice parameters a=4.65 Å, c=2.95 Å, atomic positions Ti: (0,0,0), O: (0.31,0.31,0)...", "Retrieved 3 SiO2 structures: quartz (mp-7000), cristobalite (mp-546794)..." [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If there are Materials Project API issues, invalid material identifiers, or structure data unavailable. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception occurs when the API cannot retrieve structural data, material IDs don't exist, or when structural information is not available in the Materials Project database. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify material formula or ID exists in Materials Project database, check API key validity, or try alternative material identifiers. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires valid MP_API_KEY environment variable
        - Limited to structures available in Materials Project database
        - Structure data quality depends on DFT calculations performed
        - Large structure queries may timeout and need decomposition
        - Some experimental structures may not be available
        - Pymatgen structure format may require additional processing for some applications
    [/LIMITATIONS]
    """

    @property
    def tools(self):
        return [MaterialsStructureText()]


class MPThermoExpert(MPAgent):
    r"""[BRIEF] Thermodynamics expert with access to Materials Project thermodynamic properties including formation energies, decomposition enthalpies, and phase stability data. [\BRIEF]

    [DETAILED] This specialized agent provides comprehensive thermodynamic analysis using the Materials Project thermo endpoint. It can retrieve and analyze thermodynamic properties such as formation energy per atom, decomposition enthalpy, energy above hull, equilibrium reaction energies, mass density, atomic density, and raw DFT-calculated energies. The tool is particularly useful for phase stability analysis, materials comparison, and thermodynamic feasibility studies. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need thermodynamic properties like formation energies or decomposition enthalpies
    - When analyzing phase stability or materials thermodynamic feasibility
    - For comparing thermodynamic properties between different materials
    - When you need energy above hull calculations for phase diagram analysis
    - For density calculations or raw DFT energy data analysis [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify the materials and specific thermodynamic properties needed for analysis. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with detailed thermodynamic questions including material specifications. [\CURRENT]
    3. [FOLLOW_UP] Use thermodynamic data for phase diagram construction, stability analysis, or materials selection criteria. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Accesses Materials Project thermodynamic database with extensive computed properties
    - Retrieves formation energies calculated from DFT ground-state energies
    - Provides decomposition pathways and reaction energies
    - Calculates energy above hull for phase stability assessment
    - Returns mass and atomic density data from optimized structures
    - Integrates thermodynamic analysis with materials informatics approaches [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mp_thermo_expert("What is the formation energy per atom of Li2CO3?")`,
        `mp_thermo_expert("Compare the thermodynamic stability of different TiO2 polymorphs")`,
        `mp_thermo_expert("Calculate the decomposition energy of LiCoO2 battery material")`,
        `mp_thermo_expert("Find the energy above hull for all materials in the Li-Ni-Co-O system")`,
        `mp_thermo_expert("What is the density and formation enthalpy of perovskite LaFeO3?")`
    ]
    [\SYNTACTICAL]

    Args:
        input_question (str):
                        [BRIEF] Complete question about thermodynamic properties or phase stability [\BRIEF]
                        [DETAILED] A comprehensive question specifying the materials of interest and the thermodynamic properties needed, such as formation energies, decomposition pathways, phase stability, or density calculations. Include any comparative analysis requirements. [\DETAILED]
                        [SYNTACTICAL] Format: "Specific question about thermodynamic properties, stability, or energy calculations" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What is the formation energy per atom of Li2CO3?", "Compare thermodynamic stability of TiO2 polymorphs", "Calculate decomposition energy of LiCoO2" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Comprehensive thermodynamic analysis including formation energies, stability data, and relevant thermodynamic properties [\BRIEF]
                        [DETAILED] Detailed thermodynamic information including numerical values for formation energies, decomposition enthalpies, energy above hull, density values, and interpretive analysis of phase stability and thermodynamic feasibility. [\DETAILED]
                        [EXAMPLES] Examples: "Li2CO3 (mp-3054) has formation energy -2.89 eV/atom, energy above hull 0.02 eV/atom indicating metastability...", "TiO2 polymorphs stability: rutile (most stable), anatase (+0.05 eV/atom), brookite (+0.09 eV/atom)..." [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If thermodynamic data is unavailable, API access issues, or computational errors in thermodynamic calculations. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception occurs when Materials Project lacks thermodynamic data for requested materials, API authentication fails, or when thermodynamic calculations cannot be completed due to data limitations. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify materials exist in MP database, check API key, or request alternative thermodynamic properties that may be available. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires valid MP_API_KEY environment variable
        - Limited to materials with computed DFT data in Materials Project
        - Thermodynamic properties are based on 0K ground-state calculations
        - Temperature and pressure effects not included in standard calculations
        - Some metastable phases may have limited thermodynamic data
        - Accuracy depends on DFT functional and computational settings used
    [/LIMITATIONS]
    """

    @property
    def tools(self):
        return [MaterialsThermo()]


class MPElasticityExpert(MPAgent):
    r"""[BRIEF] Elasticity expert with access to Materials Project elastic properties including bulk modulus, shear modulus, elastic tensors, and mechanical anisotropy data. [\BRIEF]

    [DETAILED] This specialized agent provides comprehensive mechanical and elastic property analysis using the Materials Project elasticity endpoint. It can retrieve and analyze elastic properties such as bulk modulus, shear modulus, Young's modulus, Poisson ratio, universal anisotropy index, elastic tensors, compliance tensors, and mechanical stability criteria. The tool is essential for materials selection in structural applications and mechanical property optimization. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need mechanical properties like bulk modulus, shear modulus, or Young's modulus
    - When analyzing elastic anisotropy or mechanical stability of materials
    - For materials selection in structural or mechanical applications
    - When you need complete elastic tensor or compliance tensor data
    - For comparing mechanical properties between different materials or phases [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify materials and specific elastic/mechanical properties needed for analysis. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with detailed questions about elastic properties or mechanical behavior. [\CURRENT]
    3. [FOLLOW_UP] Use elastic property data for materials selection, mechanical design, or further computational mechanics analysis. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Accesses Materials Project elasticity database with computed elastic tensors
    - Provides bulk, shear, and Young's moduli calculated from elastic constants
    - Calculates Poisson ratio and universal anisotropy index for mechanical characterization
    - Returns complete 6x6 elastic and compliance tensors in Voigt notation
    - Analyzes mechanical stability using Born stability criteria
    - Supports filtering and sorting by various elastic property ranges [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mp_elasticity_expert("What is the bulk modulus and shear modulus of diamond?")`,
        `mp_elasticity_expert("Compare the elastic anisotropy of different steel alloys")`,
        `mp_elasticity_expert("Find materials with Young's modulus greater than 300 GPa")`,
        `mp_elasticity_expert("Get the complete elastic tensor for silicon carbide")`,
        `mp_elasticity_expert("Which ceramic materials have the highest Poisson ratio?")`
    ]
    [\SYNTACTICAL]

    Args:
        input_question (str):
                        [BRIEF] Complete question about elastic properties or mechanical behavior [\BRIEF]
                        [DETAILED] A comprehensive question specifying the materials and elastic properties of interest, such as moduli, elastic tensors, anisotropy indices, or mechanical stability criteria. Include any filtering or comparison requirements. [\DETAILED]
                        [SYNTACTICAL] Format: "Specific question about elastic moduli, tensors, or mechanical properties" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What is the bulk modulus of diamond?", "Compare elastic anisotropy of steel alloys", "Find materials with Young's modulus > 300 GPa" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Comprehensive elastic property analysis including moduli, tensors, anisotropy data, and mechanical characterization [\BRIEF]
                        [DETAILED] Detailed elastic property information including numerical values for bulk, shear, and Young's moduli, Poisson ratios, anisotropy indices, complete elastic/compliance tensors, and interpretive analysis of mechanical behavior and stability. [\DETAILED]
                        [EXAMPLES] Examples: "Diamond (mp-66) has bulk modulus 442 GPa, shear modulus 535 GPa, Young's modulus 1050 GPa, Poisson ratio 0.069...", "Found 23 materials with Young's modulus > 300 GPa: Diamond (1050 GPa), Lonsdaleite (941 GPa)..." [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If elastic property data is unavailable, API access issues, or materials lack computed elastic tensors. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception occurs when Materials Project lacks elastic tensor calculations for requested materials, API authentication fails, or when elastic property computations cannot be completed. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify materials have elastic data in MP database, check API key validity, or request alternative materials with available elastic properties. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires valid MP_API_KEY environment variable
        - Limited to materials with computed elastic tensors in Materials Project
        - Elastic properties calculated at 0K and ambient pressure
        - Some materials may lack complete elastic tensor data
        - Accuracy depends on DFT computational parameters used
        - Temperature and pressure dependence not included in standard calculations
    [/LIMITATIONS]
    """

    @property
    def tools(self):
        return [MaterialsElasticity()]


class MPMagnetismExpert(MPAgent):
    r"""[BRIEF] Magnetism expert with access to Materials Project magnetic properties including magnetic moments, ordering, exchange interactions, and magnetic site analysis. [\BRIEF]

    [DETAILED] This specialized agent provides comprehensive magnetic property analysis using the Materials Project magnetism endpoint. It can retrieve and analyze magnetic properties such as magnetic moments (magmoms), magnetic ordering and symmetry, number of magnetic sites, types of magnetic species, total magnetization, magnetization normalized by volume and formula units, exchange symmetry, and magnetic ground state information. The tool is essential for understanding magnetic materials and their applications in spintronics, data storage, and magnetic devices. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need magnetic moment data or magnetic ordering information
    - When analyzing magnetic materials for spintronics or magnetic device applications
    - For identifying ferromagnetic, antiferromagnetic, or ferrimagnetic materials
    - When you need magnetic site analysis or magnetic species information
    - For comparing magnetic properties between different materials or magnetic phases [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify materials and specific magnetic properties needed for analysis. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with detailed questions about magnetic behavior or magnetic materials selection. [\CURRENT]
    3. [FOLLOW_UP] Use magnetic property data for spintronics design, magnetic device development, or further magnetic materials analysis. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Accesses Materials Project magnetism database with computed magnetic properties
    - Provides magnetic moments calculated from spin-polarized DFT calculations
    - Analyzes magnetic ordering including ferromagnetic, antiferromagnetic, and ferrimagnetic states
    - Returns information about magnetic sites and magnetic species in crystal structures
    - Calculates total magnetization and normalized magnetization values
    - Supports filtering by magnetic properties and magnetic ground states [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mp_magnetism_expert("What is the magnetic moment of iron metal?")`,
        `mp_magnetism_expert("Find all antiferromagnetic materials containing Mn and O")`,
        `mp_magnetism_expert("Compare the magnetic ordering of different iron oxide phases")`,
        `mp_magnetism_expert("Which materials have the highest magnetic moments per formula unit?")`,
        `mp_magnetism_expert("Get magnetic site information for the perovskite LaFeO3")`
    ]
    [\SYNTACTICAL]

    Args:
        input_question (str):
                        [BRIEF] Complete question about magnetic properties or magnetic materials [\BRIEF]
                        [DETAILED] A comprehensive question specifying the materials and magnetic properties of interest, such as magnetic moments, magnetic ordering, magnetic sites, or magnetic material selection criteria. Include any filtering or comparison requirements. [\DETAILED]
                        [SYNTACTICAL] Format: "Specific question about magnetic moments, ordering, or magnetic behavior" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What is the magnetic moment of iron?", "Find antiferromagnetic Mn-O materials", "Compare magnetic ordering of iron oxides" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Comprehensive magnetic property analysis including magnetic moments, ordering information, and magnetic site data [\BRIEF]
                        [DETAILED] Detailed magnetic property information including magnetic moment values, magnetic ordering type (ferromagnetic/antiferromagnetic/ferrimagnetic), number and types of magnetic sites, total magnetization data, and interpretive analysis of magnetic behavior and potential applications. [\DETAILED]
                        [EXAMPLES] Examples: "Iron (mp-13) has magnetic moment 2.22 μB/atom, ferromagnetic ordering, 1 magnetic site...", "Found 45 antiferromagnetic Mn-O materials: MnO (mp-19395, moment 4.9 μB), Mn2O3 (mp-18759)..." [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If magnetic property data is unavailable, API access issues, or materials lack computed magnetic data. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception occurs when Materials Project lacks magnetic calculations for requested materials, API authentication fails, or when magnetic property computations cannot be completed due to data limitations. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify materials have magnetic data in MP database, check API key validity, or request alternative magnetic materials with available data. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires valid MP_API_KEY environment variable
        - Limited to materials with computed magnetic properties in Materials Project
        - Magnetic properties calculated using specific DFT magnetic settings
        - Some materials may lack complete magnetic site analysis
        - Temperature effects on magnetic properties not included
        - Accuracy depends on DFT treatment of magnetic interactions
    [/LIMITATIONS]
    """

    @property
    def tools(self):
        return [MaterialsMagnetism()]


class MPDielectricExpert(MPAgent):
    r"""[BRIEF] Dielectric expert with access to Materials Project dielectric properties including dielectric constants, dielectric tensors, and refractive indices. [\BRIEF]

    [DETAILED] This specialized agent provides comprehensive dielectric property analysis using the Materials Project dielectric endpoint. It can retrieve and analyze dielectric properties such as total dielectric constant, ionic and electronic contributions to dielectricity, dielectric tensors, refractive index (n), and ferroelectric potential. The dielectric tensors are calculated from first-principles Density Functional Perturbation Theory (DFPT) and include both electronic and ionic contributions. This tool is essential for applications in electronics, photonics, and energy storage devices. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need dielectric constants or dielectric tensor data
    - When analyzing materials for electronic or photonic applications
    - For finding materials with specific refractive indices
    - When you need to separate ionic and electronic contributions to dielectric properties
    - For identifying potential ferroelectric materials or high-k dielectric materials [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify materials and specific dielectric properties needed for electronic or photonic applications. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with detailed questions about dielectric behavior or materials selection for electronic devices. [\CURRENT]
    3. [FOLLOW_UP] Use dielectric property data for electronic device design, photonic applications, or further optoelectronic materials analysis. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Accesses Materials Project dielectric database with DFPT-calculated properties
    - Provides total dielectric constants and separates ionic/electronic contributions
    - Returns complete dielectric tensors with symmetry information
    - Calculates refractive indices from electronic dielectric constants
    - Analyzes potential for ferroelectricity based on dielectric properties
    - Supports filtering by dielectric constant ranges and optical properties [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mp_dielectric_expert("What is the dielectric constant of silicon?")`,
        `mp_dielectric_expert("Find materials with high refractive index greater than 2.5")`,
        `mp_dielectric_expert("Compare ionic and electronic contributions to BaTiO3 dielectric constant")`,
        `mp_dielectric_expert("Which oxide materials have dielectric constants above 100?")`,
        `mp_dielectric_expert("Get the complete dielectric tensor for quartz SiO2")`
    ]
    [\SYNTACTICAL]

    Args:
        input_question (str):
                        [BRIEF] Complete question about dielectric properties or optical behavior [\BRIEF]
                        [DETAILED] A comprehensive question specifying the materials and dielectric properties of interest, such as dielectric constants, dielectric tensors, refractive indices, or materials selection for electronic/photonic applications. Include any filtering criteria or comparison requirements. [\DETAILED]
                        [SYNTACTICAL] Format: "Specific question about dielectric constants, tensors, or optical properties" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What is the dielectric constant of silicon?", "Find materials with refractive index > 2.5", "Compare BaTiO3 dielectric contributions" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Comprehensive dielectric property analysis including dielectric constants, tensors, refractive indices, and ionic/electronic contributions [\BRIEF]
                        [DETAILED] Detailed dielectric property information including total, ionic, and electronic dielectric constants, complete dielectric tensors, refractive index values, symmetry information, and interpretive analysis of dielectric behavior and potential applications in electronics and photonics. [\DETAILED]
                        [EXAMPLES] Examples: "Silicon (mp-149) has total dielectric constant 13.1 (electronic: 12.9, ionic: 0.2), refractive index 3.42...", "Found 12 materials with refractive index > 2.5: TiO2 rutile (n=2.89), ZnS (n=2.67)..." [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If dielectric property data is unavailable, API access issues, or materials lack computed dielectric tensors. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception occurs when Materials Project lacks dielectric calculations for requested materials, API authentication fails, or when dielectric property computations cannot be completed due to convergence issues or data limitations. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify materials have dielectric data in MP database, check API key validity, or request alternative materials with available dielectric properties. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires valid MP_API_KEY environment variable
        - Limited to materials with computed dielectric tensors in Materials Project
        - Dielectric properties calculated at 0K and ambient conditions
        - Some materials may lack complete ionic/electronic separation
        - Accuracy depends on DFPT computational parameters and convergence
        - Temperature and frequency dependence not included in standard calculations
    [/LIMITATIONS]
    """

    @property
    def tools(self):
        return [MaterialsDielectric()]


class MPPiezoelectricExpert(MPAgent):
    r"""[BRIEF] Piezoelectric expert with access to Materials Project piezoelectric properties including piezoelectric tensors, moduli, and strain constants. [\BRIEF]

    [DETAILED] This specialized agent provides comprehensive piezoelectric property analysis using the Materials Project piezoelectric endpoint. Piezoelectricity is a reversible physical process where an electric moment is generated upon stress application (direct effect) or strain is generated upon electric field application (indirect effect). The tool can retrieve piezoelectric tensors, piezoelectric moduli, strain constants, maximum piezoelectric response directions, and the strain required for maximum response. This data is essential for applications in sensors, actuators, energy harvesting, and electromechanical devices. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need piezoelectric tensor or modulus data for material selection
    - When designing sensors, actuators, or energy harvesting devices
    - For analyzing electromechanical coupling in materials
    - When you need maximum piezoelectric response directions and strain information
    - For comparing piezoelectric performance between different materials [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify materials and specific piezoelectric properties needed for electromechanical applications. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with detailed questions about piezoelectric behavior or materials selection for electromechanical devices. [\CURRENT]
    3. [FOLLOW_UP] Use piezoelectric property data for sensor/actuator design, energy harvesting optimization, or further electromechanical analysis. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Accesses Materials Project piezoelectric database with computed tensors
    - Provides complete piezoelectric tensors describing electromechanical coupling
    - Calculates piezoelectric moduli and strain constants from tensor data
    - Identifies maximum response directions and required strain conditions
    - Returns both total and separated ionic/electronic contributions
    - Supports filtering by piezoelectric response magnitude and other criteria [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mp_piezoelectric_expert("What is the piezoelectric tensor of quartz?")`,
        `mp_piezoelectric_expert("Find materials with the highest piezoelectric modulus")`,
        `mp_piezoelectric_expert("Compare piezoelectric response of different PZT compositions")`,
        `mp_piezoelectric_expert("Which direction gives maximum piezoelectric response in BaTiO3?")`,
        `mp_piezoelectric_expert("Get ionic and electronic contributions to LiNbO3 piezoelectricity")`
    ]
    [\SYNTACTICAL]

    Args:
        input_question (str):
                        [BRIEF] Complete question about piezoelectric properties or electromechanical behavior [\BRIEF]
                        [DETAILED] A comprehensive question specifying the materials and piezoelectric properties of interest, such as piezoelectric tensors, moduli, strain constants, response directions, or materials selection for electromechanical applications. Include any performance criteria or comparison requirements. [\DETAILED]
                        [SYNTACTICAL] Format: "Specific question about piezoelectric tensors, moduli, or electromechanical coupling" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What is the piezoelectric tensor of quartz?", "Find materials with highest piezoelectric modulus", "Compare PZT piezoelectric response" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Comprehensive piezoelectric property analysis including tensors, moduli, response directions, and electromechanical coupling data [\BRIEF]
                        [DETAILED] Detailed piezoelectric property information including complete piezoelectric tensors, moduli values, strain constants, maximum response directions, strain conditions, ionic/electronic contributions, and interpretive analysis of electromechanical behavior and device applications. [\DETAILED]
                        [EXAMPLES] Examples: "Quartz (mp-7000) has piezoelectric modulus d11 = 2.31 pC/N, maximum response in [100] direction...", "Found materials with highest piezoelectric response: LiNbO3 (d33 = 27.2 pC/N), BaTiO3 (d33 = 17.5 pC/N)..." [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If piezoelectric property data is unavailable, API access issues, or materials lack computed piezoelectric tensors. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception occurs when Materials Project lacks piezoelectric calculations for requested materials (many materials are not piezoelectric), API authentication fails, or when piezoelectric property computations cannot be completed. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify materials are piezoelectric and have data in MP database, check API key validity, or request alternative piezoelectric materials with available data. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires valid MP_API_KEY environment variable
        - Limited to materials with computed piezoelectric tensors in Materials Project
        - Only non-centrosymmetric materials can exhibit piezoelectricity
        - Piezoelectric properties calculated at 0K and ambient conditions
        - Some piezoelectric materials may lack complete tensor data
        - Accuracy depends on DFPT computational parameters and convergence
    [/LIMITATIONS]
    """

    @property
    def tools(self):
        return [MaterialsPiezoelectric()]


class MPElectronicExpert(MPAgent):
    r"""[BRIEF] Electronic structure expert with access to Materials Project electronic properties including band gaps, band structures, density of states, and Fermi energies. [\BRIEF]

    [DETAILED] This specialized agent provides comprehensive electronic structure analysis using the Materials Project electronic structure endpoint. It can retrieve and analyze electronic properties such as band gaps (direct and indirect), Fermi energies, band structures, density of states, conduction band minimum and valence band maximum positions, and electronic band topology. This tool is essential for understanding semiconductor properties, conductor/insulator classification, and electronic material applications in photovoltaics, electronics, and optoelectronics. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need band gap information for semiconductor applications
    - When analyzing electronic band structures or density of states
    - For materials selection in photovoltaic or electronic device applications
    - When you need Fermi energy or electronic transport property estimates
    - For classifying materials as metals, semiconductors, or insulators [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify materials and specific electronic properties needed for electronic or optoelectronic applications. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with detailed questions about electronic structure or materials selection for electronic devices. [\CURRENT]
    3. [FOLLOW_UP] Use electronic property data for device design, photovoltaic optimization, or further electronic materials analysis. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Accesses Materials Project electronic structure database with DFT-calculated properties
    - Provides band gap values calculated using various DFT functionals
    - Returns band structure and density of states data for detailed analysis
    - Calculates conduction band minimum and valence band maximum energies
    - Analyzes electronic band topology and identifies band gap nature (direct/indirect)
    - Supports filtering by band gap ranges and electronic property criteria [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mp_electronic_expert("What is the band gap of silicon?")`,
        `mp_electronic_expert("Find semiconductors with band gaps between 1-3 eV for solar cells")`,
        `mp_electronic_expert("Compare the electronic band structures of GaAs and InP")`,
        `mp_electronic_expert("Which materials are metals with high density of states at Fermi level?")`,
        `mp_electronic_expert("Get the valence and conduction band positions for TiO2")`
    ]
    [\SYNTACTICAL]

    Args:
        input_question (str):
                        [BRIEF] Complete question about electronic structure or electronic properties [\BRIEF]
                        [DETAILED] A comprehensive question specifying the materials and electronic properties of interest, such as band gaps, band structures, density of states, Fermi energies, or materials selection for electronic/optoelectronic applications. Include any performance criteria or comparison requirements. [\DETAILED]
                        [SYNTACTICAL] Format: "Specific question about band gaps, electronic structure, or electronic behavior" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What is the band gap of silicon?", "Find semiconductors with band gaps 1-3 eV", "Compare GaAs and InP band structures" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Comprehensive electronic structure analysis including band gaps, band structure data, density of states, and electronic classification [\BRIEF]
                        [DETAILED] Detailed electronic property information including band gap values, direct/indirect nature, band structure plots, density of states data, Fermi energy levels, conduction/valence band positions, and interpretive analysis of electronic behavior and device applications. [\DETAILED]
                        [EXAMPLES] Examples: "Silicon (mp-149) has indirect band gap 0.61 eV (PBE), conduction band minimum at X-point, valence band maximum at Γ-point...", "Found 156 semiconductors with band gaps 1-3 eV: GaAs (1.42 eV), CdTe (1.44 eV)..." [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If electronic structure data is unavailable, API access issues, or materials lack computed band structure data. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception occurs when Materials Project lacks electronic structure calculations for requested materials, API authentication fails, or when electronic property computations cannot be completed due to convergence issues. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify materials have electronic data in MP database, check API key validity, or request alternative materials with available electronic structure data. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires valid MP_API_KEY environment variable
        - Limited to materials with computed electronic structure in Materials Project
        - Band gaps may be underestimated due to DFT functional limitations
        - Some materials may lack complete band structure or DOS data
        - Accuracy depends on DFT functional choice and computational parameters
        - Temperature effects on electronic properties not included
    [/LIMITATIONS]
    """

    @property
    def tools(self):
        return [MaterialsElectronic()]


class MPSynthesisExpert(MPAgent):
    r"""[BRIEF] Materials synthesis expert with access to Materials Project synthesis recipes extracted from scientific literature through text mining and natural language processing. [\BRIEF]

    [DETAILED] This specialized agent provides comprehensive materials synthesis information using the Materials Project synthesis endpoint. It accesses synthesis recipes that have been extracted from scientific literature using advanced text mining and natural language processing approaches. The tool can retrieve synthesis reactions, precursors, target materials, operational conditions, required devices, processing steps, and literature references. This data is invaluable for experimental materials synthesis planning, process optimization, and understanding synthesis-structure-property relationships. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need experimental synthesis procedures for specific materials
    - When planning materials synthesis experiments or process development
    - For finding precursors, reaction conditions, and synthesis equipment requirements
    - When you need literature references for materials synthesis methods
    - For comparing different synthesis routes for the same material [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify target materials and synthesis information needed for experimental work. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with detailed questions about synthesis procedures, conditions, or precursors. [\CURRENT]
    3. [FOLLOW_UP] Use synthesis information for experimental design, process optimization, or literature review for materials preparation. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Accesses Materials Project synthesis database with literature-extracted recipes
    - Provides synthesis reactions with balanced chemical equations where available
    - Returns precursor materials, target products, and intermediate phases
    - Lists operational conditions including temperature, pressure, time, and atmosphere
    - Identifies required synthesis equipment and processing steps
    - Includes literature references and DOIs for original synthesis reports [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `mp_synthesis_expert("How do I synthesize lithium cobalt oxide LiCoO2?")`,
        `mp_synthesis_expert("What are the precursors and conditions for making perovskite solar cell materials?")`,
        `mp_synthesis_expert("Find synthesis routes for graphene or carbon nanotube production")`,
        `mp_synthesis_expert("What equipment is needed to synthesize high-temperature superconductors?")`,
        `mp_synthesis_expert("Compare different synthesis methods for titanium dioxide nanoparticles")`
    ]
    [\SYNTACTICAL]

    Args:
        input_question (str):
                        [BRIEF] Complete question about materials synthesis procedures, conditions, or precursors [\BRIEF]
                        [DETAILED] A comprehensive question specifying the target materials and synthesis information needed, such as synthesis procedures, reaction conditions, precursors, equipment requirements, or synthesis route comparisons. Include any specific constraints or requirements for the synthesis process. [\DETAILED]
                        [SYNTACTICAL] Format: "Specific question about synthesis procedures, precursors, or processing conditions" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "How do I synthesize LiCoO2?", "What precursors are needed for perovskite materials?", "Find synthesis routes for graphene" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Comprehensive synthesis information including recipes, precursors, conditions, equipment, and literature references [\BRIEF]
                        [DETAILED] Detailed synthesis information including complete synthesis procedures, precursor materials, reaction conditions (temperature, pressure, time, atmosphere), required equipment, processing steps, target products, and literature references with DOIs for experimental validation and further reading. [\DETAILED]
                        [EXAMPLES] Examples: "LiCoO2 synthesis: React Li2CO3 and Co3O4 at 900°C for 12h in air, precursors in 1:1 molar ratio, requires high-temperature furnace (DOI: 10.1016/...)...", "Found 15 synthesis routes for perovskite materials: solid-state reaction (800-1000°C), sol-gel (400-600°C), hydrothermal (150-200°C)..." [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If synthesis data is unavailable, API access issues, or materials lack literature-extracted synthesis information. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception occurs when Materials Project lacks synthesis recipes for requested materials, API authentication fails, or when synthesis information cannot be retrieved due to limited literature coverage or text mining limitations. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try alternative material names or formulations, check API key validity, or request synthesis information for related materials that may have available data. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires valid MP_API_KEY environment variable
        - Limited to materials with literature-reported synthesis procedures
        - Synthesis information quality depends on literature coverage and text mining accuracy
        - Some specialized or novel synthesis methods may not be captured
        - Extraction accuracy may vary depending on literature source quality
        - Not all synthesis conditions may be fully captured from literature text
    [/LIMITATIONS]
    """

    @property
    def tools(self):
        return [MaterialsSynthesis()]


def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    Path(BASE_WORK_DIR).mkdir(parents=True, exist_ok=True)
    fs_manager = FSManager("file", base_path=BASE_WORK_DIR)
    return {
        "list_files": ListFilesTool(fs_manager),
        "read_file": ReadFileTool(fs_manager),
        "write_file": WriteFileTool(fs_manager),
        "file_info": FileInfoTool(fs_manager),
        "cat_files": CatFilesTool(fs_manager),
        "copy_file": CopyFileTool(fs_manager),
        "MP_thermo_expert": MPThermoExpert().as_tool(),
        "MP_elasticity_expert": MPElasticityExpert().as_tool(),
        "MP_dielectric_expert": MPDielectricExpert().as_tool(),
        "MP_magnetism_expert": MPMagnetismExpert().as_tool(),
        "MP_electronic_expert": MPElectronicExpert().as_tool(),
        "MP_piezoelectronic_expert": MPPiezoelectricExpert().as_tool(),
        "MP_summary_expert": MPSummaryExpert().as_tool(),
        "MP_synthesis_expert": MPSynthesisExpert().as_tool(),
        "MP_structure_retriever": MPStructureRetriever().as_tool(),
        "arxiv": arxiv,
        "wikipedia": wikipedia,
        "Python_REPL_tool": PythonREPLTool(),
    }
