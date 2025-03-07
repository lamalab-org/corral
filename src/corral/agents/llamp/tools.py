"""
Materials Project API tools for agent-based materials science operations.

This module provides a set of specialized tools for querying the Materials Project API
to retrieve various materials properties and data. It implements a framework for converting
Pydantic schemas into tool arguments that can be used by agents.

This code is adapted from the LLAMP (Large Language model Agent for Materials Project) repository:
https://github.com/chiang-yuan/llamp

Only modifications to fit our environment were made while preserving the original functionality.
"""

from __future__ import annotations

import json
import os
import re

from corral.agents.llamp.schemas import (
    BondsSchema,
    DielectricSchema,
    ElasticitySchema,
    ElectronicSchema,
    MagnetismSchema,
    OxidationSchema,
    PiezoSchema,
    RobocrysSchema,
    SimilaritySchema,
    StructureSchema,
    SummarySchema,
    SynthesisSchema,
    TasksSchema,
    ThermoSchema,
)
from corral.agents.llamp.utilities import MPAPIWrapper
from corral.base import Tool, ToolArgument


def pydantic_schema_to_tool_arguments(schema_class) -> list[ToolArgument]:
    """
    Convert a Pydantic schema class to a list of ToolArgument objects.

    Args:
        schema_class: A Pydantic BaseModel class

    Returns:
        list[ToolArgument]: A list of ToolArgument objects representing the schema fields
    """
    arguments = []

    model_fields = (
        schema_class.model_fields
        if hasattr(schema_class, "model_fields")
        else schema_class.__fields__
    )

    for field_name, field in model_fields.items():
        field_type = "str"  # Default type

        if hasattr(field, "annotation"):
            annotation = field.annotation
            if annotation == int:
                field_type = "int"
            elif annotation == float:
                field_type = "float"
            elif annotation == bool:
                field_type = "bool"
            elif annotation == list:
                field_type = "list"
            elif annotation == dict:
                field_type = "dict"
        elif hasattr(field, "type_"):
            try:
                if field.type_ == int:
                    field_type = "int"
                elif field.type_ == float:
                    field_type = "float"
                elif field.type_ == bool:
                    field_type = "bool"
                elif field.type_ == list:
                    field_type = "list"
                elif field.type_ == dict:
                    field_type = "dict"
            except AttributeError:
                if hasattr(field, "outer_type_"):
                    outer_type = field.outer_type_
                    if outer_type == int:
                        field_type = "int"
                    elif outer_type == float:
                        field_type = "float"
                    elif outer_type == bool:
                        field_type = "bool"
                    elif outer_type == list:
                        field_type = "list"
                    elif outer_type == dict:
                        field_type = "dict"

        description = ""
        if hasattr(field, "description"):
            description = field.description or ""
        elif hasattr(field, "field_info") and hasattr(field.field_info, "description"):
            description = field.field_info.description or ""

        required = True
        if hasattr(field, "is_required"):
            required = field.is_required
        elif hasattr(field, "required"):
            required = field.required

        default = None
        if not required:
            if hasattr(field, "default"):
                default = field.default
            elif hasattr(field, "field_info") and hasattr(field.field_info, "default"):
                default = field.field_info.default

        choices = None
        if hasattr(field, "choices") and field.choices:
            choices = field.choices
        elif (
            hasattr(field, "field_info")
            and hasattr(field.field_info, "choices")
            and field.field_info.choices
        ):
            choices = field.field_info.choices

        tool_arg = ToolArgument(
            name=field_name,
            type=field_type,
            description=description,
            required=required,
            default=default,
            choices=choices,
        )

        arguments.append(tool_arg)

    return arguments


class MPTool(Tool):
    name: str | None = None
    api_wrapper = MPAPIWrapper(mpApiKey=os.getenv("MP_API_KEY"))

    def __init__(self, *args, **kwargs):
        arguments = pydantic_schema_to_tool_arguments(self.args_schema)
        super().__init__(
            name=self.name,
            description=self.description,
            arguments=arguments,
        )
        mp_api_key = os.getenv("MP_API_KEY", kwargs.get("mp_api_key"))
        self.api_wrapper.set_api_key(mp_api_key)

    def execute(self, **query_params):
        _res = self.api_wrapper.run(
            function_name=self.name,
            function_args=json.dumps(query_params),
        )
        return _res


class MaterialsSummary(MPTool):
    name: str = "search_materials_summary__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need calulated or derived materials properties, also useful
        when you need to perform filtering on chemical systems or sorting on materials
        properties, also useful when you need high-level information about materials
        (such as material_id) and use the results to perform further queries using
        other tools.

        TIPS:
        - If pure elemental structure is desired, use `formula` instead of `elements`
        """,
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[SummarySchema] = SummarySchema


class MaterialsStructureText(MPTool):
    name: str = "search_materials_structure__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need to get the pymatgen structures from Materials
            Project as JSON text for structure generation or manipulation.

            TIPS:
            - If pure elemental structure is desired, use `formula` instead of `elements`
            """,
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[StructureSchema] = StructureSchema(return_mode="file")


class MaterialsElasticity(MPTool):
    name: str = "search_materials_elasticity__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need elasticity properties like bulk modulus, shear modulus,
        elastic anisotropy, Poisson ratio, elastic/compliance tensors, also useful when
        you need to perform filtering and sorting elastic properties and retrieve the
        qualified materials""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[ElasticitySchema] = ElasticitySchema


class MaterialsSynthesis(MPTool):
    name: str = "search_materials_synthesis__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need materials synthesis recipes and list out the reactions,
        precursors, targets, operations, conditions, required devices and references""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[SynthesisSchema] = SynthesisSchema

    def execute(self, **query_params):
        _res = self.api_wrapper.run(
            function_name=self.name,
            function_args=json.dumps(query_params),
        )
        return _res


class MaterialsThermo(MPTool):
    name: str = "search_materials_thermo__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need thermodynamic properties like mass density, atomic
        density, formation energy, decomposition enthalpy, energy above hull,
        equilibrium reaction energy, and raw DFT calculated energy, also useful when
        you want to compare the thermodynamic properties of materials with
        material_ids""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[ThermoSchema] = ThermoSchema


class MaterialsMagnetism(MPTool):
    name: str = "search_materials_magnetism__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need magnetic properties like magnetic moment (magmoms),
        magnetic ordering and symmetry, magnetic sites, magnetic type, also useful when
        you need to perform filtering and sorting magnetic properties and retrieve the
        qualified materials.""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[MagnetismSchema] = MagnetismSchema


class MaterialsDielectric(MPTool):
    name: str = "search_materials_dielectric__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            r"""useful when you need dielectric properties like dielectric constant,
        dielectric tensor, refractive index (n), also useful when you
        need to perform filtering and sorting dielectric properties and retrieve the
        qualified materials. A dielectric is a material that can be polarized by an
        applied electric field. This limits the dielectric effect to materials with
        a non-zero band gap. Formally, the dielectric tensor ε relates the externally
        applied electric field to the field within the material and can be defined as:
        $$E_i=\sum_j \epsilon^{-1}_{ij}E_{0j}$$ where $$E$$ is the electric field
        inside the material and $$E_{0}$$ is the externally applied electric field.
        The dielectric tensors from the Materials Project are calculated from first
        principles Density Functional Perturbation Theory (DFPT) and are approximated
        as the superimposed effect of an electronic and ionic contribution $$
        \epsilon_{ij}=\epsilon_{ij}^0+\epsilon_{ij}^\infty$$
        From the full piezoelectric tensor, several properties are derived such as the
        refractive index and potential for ferroelectricity.
        The mathematical description of the dielectric effect is a tensor constant of
        proportionality that relates an externally applied electric field to the field
        within the material. Along with the elastic and piezoelectric tensors, the
        dielectric tensor provides all the information necessary for the solution of
        the constitutive equations in applications where electric and mechanical
        stresses are coupled""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[DielectricSchema] = DielectricSchema


class MaterialsPiezoelectric(MPTool):
    name: str = "search_materials_piezoelectric__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need piezoelectric properties like piezoelectric
        tensor, piezoelectric modulus, piezoelectric strain constant, also useful
        when you need to perform filtering and sorting piezoelectric properties and
        retrieve the qualified materials. Piezoelectricity is a reversible physical
        process that occurs in some materials whereby an electric moment is generated
        upon the application of a stress. This is often referred to as the direct
        piezoelectric effect. Conversely, the indirect piezoelectric effect refers to
        the case when a strain is generated in a material upon the application of an
        electric field. The mathematical description of piezoelectricity relates the
        strain (or stress) to the electric field via a third order tensor. This tensor
        describes the response of any piezoelectric bulk material, when subjected to an
        electric field or a mechanical load""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[PiezoSchema] = PiezoSchema


class MaterialsRobocrystallographer(MPTool):
    name: str = "search_materials_robocrys__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """Robocrystallographer is a tool to generate text descriptions of crystal
            structures. User this more than 'search_materials_summary__get' if the
            question is more qualitative and descriptive.
            Similar to how a real-life crystallographer would analyse a
            structure, robocrystallographer looks at the symmetry, local coordination
            and polyhedral type, polyhedral connectivity, octahedral tilt angles,
            component-dimensionality, and molecule-within-crystal and fuzzy prototype
            identification when generating a description""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[RobocrysSchema] = RobocrysSchema


class MaterialsOxidation(MPTool):
    name: str = "search_materials_oxidation_states__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need possible oxidation states of elements in materials,
            also useful to sort the results by related materials properties like
            possible_species, possible_valences, average_oxidation_states, and method
            for oxidation state prediction""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[OxidationSchema] = OxidationSchema


class MaterialsBonds(MPTool):
    name: str = "search_materials_bonds__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need bond properties like bond lengths, coordination,
            also useful when you need to perform filtering and sorting bond properties
            and retrieve the qualified materials""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[BondsSchema] = BondsSchema


class MaterialsTasks(MPTool):
    name: str = "search_materials_tasks__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need task_ids of materials and detailed DFT calculation
            results""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[TasksSchema] = TasksSchema


class MaterialsSimilarity(MPTool):
    name: str = "get_by_key_materials_similarity__material_id___get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need to find similar materials based on a given material
            id. If you don't have relevant material_id, you should use
            `MaterialsSummary` tool first""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[SimilaritySchema] = SimilaritySchema


class MaterialsElectronic(MPTool):
    name: str = "search_materials_electronic_structure__get"
    description: str = (
        re.sub(
            r"\s+",
            " ",
            """useful when you need electronic structure properties like band gap,
            fermi energy, band structure, density of states, also useful when
            you need to perform filtering and sorting electronic structure properties
            and retrieve the qualified materials""",
        )
        .strip()
        .replace("\n", " ")
    )
    args_schema: type[ElectronicSchema] = ElectronicSchema
