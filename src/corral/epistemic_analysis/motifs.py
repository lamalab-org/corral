"""Direct temporal classification of the graph pipeline's final motifs.

Names, descriptions, and families come from the extracted analyze.py registry.
The extra criteria express relationships in observable conversation terms; they
do not reconstruct nodes/edges or claim to reproduce deterministic graph counts.
"""

from __future__ import annotations

from dataclasses import dataclass

from corral.epistemic_analysis.patterns import (
    ANTIPATTERN_DESCRIPTIONS,
    ANTIPATTERN_FAMILIES,
    SUBGRAPH_DESCRIPTIONS,
    SUBGRAPH_FAMILIES,
)

STATUS_DESCRIPTIONS = {
    "absent": "No supported instance or meaningful partial sequence in this prefix.",
    "emerging": (
        "A partial sequence is supported, but a defining event or a genuine "
        "opportunity to respond is still pending."
    ),
    "active": (
        "The full motif is supported and currently relevant. For an antipattern, "
        "at least one instance remains unaddressed after an opportunity to respond."
    ),
    "resolved": (
        "The motif occurred, but subsequent behavior repaired all outstanding "
        "instances (antipattern) or completed the process (productive pattern)."
    ),
    "uncertain": "The available observations cannot distinguish the other statuses.",
}

RELATION_DESCRIPTIONS = {
    "tests": (
        "A hypothesis is evaluated by a test, or a judgment motivates the test. "
        "The test must address that particular claim or interpretation."
    ),
    "observes": "A test produces the linked evidence as its observed result.",
    "informs": (
        "Evidence supports or is interpreted by a hypothesis, judgment, or "
        "commitment; a judgment can inform a hypothesis, another judgment, or "
        "commitment. Mere proximity in the conversation is insufficient."
    ),
    "updates_to": (
        "An earlier hypothesis is materially revised into a later hypothesis "
        "on the basis of intervening information. Rewording is not revision."
    ),
    "competes_with": (
        "Two coexisting hypotheses are compared as alternative explanations. "
        "This differs from replacing an earlier hypothesis with a revision."
    ),
    "contradicts": (
        "Observed evidence or a judgment conflicts with a particular hypothesis. "
        "Missing support alone does not establish a contradiction."
    ),
}

# Spell out the complete structures, including shared node identities, missing
# links, and temporal order. Names and graph shorthand alone are not definitions.
_GRAPH_DESCRIPTIONS = {
    "refutation_driven_belief_revision": (
        "Start with an initial hypothesis and a test that evaluates it. The test "
        "produces evidence, and a judgment interprets that evidence. The initial "
        "hypothesis is then materially revised into a different hypothesis based "
        "on what was learned. These events must concern the same original claim; "
        "an unrelated new hypothesis does not complete the structure."
    ),
    "fixed_hypothesis_test_tuning": (
        "A hypothesis motivates a first test, that test produces evidence, and a "
        "judgment interprets the result. That judgment motivates another test or "
        "an adjustment to the test design. The original hypothesis stays fixed: "
        "there is no revision of it in this sequence. The change is to the test, "
        "and its connection to the observed result must be identifiable."
    ),
    "explore_then_test_transition": (
        "An exploratory test first produces an observation. A hypothesis is "
        "formed after that exploration, and a subsequent targeted test evaluates "
        "the hypothesis. The defining order is exploration, observation, "
        "hypothesis formation, then hypothesis-directed testing. A test of an "
        "already stated hypothesis is not the initial exploratory event."
    ),
    "hypothesis_reranking": (
        "There are two distinct, coexisting hypotheses offered as competing "
        "explanations. They are explicitly compared, and each is linked to a "
        "test that evaluates it. Both alternatives must be under consideration; "
        "a later hypothesis that simply replaces the earlier one belongs to "
        "belief revision instead."
    ),
    "evidence_led_hypothesis_generation": (
        "An observation comes first and informs an interpretive judgment. A "
        "hypothesis is subsequently formed from that evidence and interpretation, "
        "and a test is directed at the new hypothesis. The required connection "
        "is from observed evidence through interpretation to a testable "
        "explanation, not just an observation and a later unrelated guess."
    ),
    "convergent_multi_test_evidence": (
        "One and the same hypothesis is linked to at least three distinct tests. "
        "Each of those tests produces its own observed evidence. The branches "
        "therefore share a hypothesis but have distinct test events and results. "
        "Several output fields from one test do not count as several branches; "
        "one or two completed branches form only a partial motif."
    ),
    "evidence_guided_test_redesign": (
        "A judgment interpreting previous evidence motivates a new or redesigned "
        "test, and that test produces an observed result. The judgment-to-test "
        "link must explain why this test was selected or changed. A planned "
        "test without its result is a partial sequence."
    ),
    "untested_claim": (
        "A hypothesis is present but has no test linked to its claim anywhere "
        "in the observed prefix. Unrelated experiments do not fill that missing "
        "link. Before calling the omission active, the agent must have had an "
        "opportunity to test the claim; a just-proposed hypothesis can be emerging."
    ),
    "evidence_non_uptake": (
        "An observation is present, but no judgment interprets it and no "
        "hypothesis uses it. It may still be linked to the test that produced "
        "it. The missing relationship is uptake into interpretation or belief. "
        "A later agent response must provide an opportunity for uptake before "
        "the omission is active."
    ),
    "unsupported_judgment": (
        "An interpretive or evaluative judgment is stated, but no observed "
        "evidence supports that judgment. The missing link is from evidence to "
        "the particular interpretation. Having observations elsewhere in the "
        "trajectory does not support a judgment about different results."
    ),
    "stalled_revision": (
        "An earlier hypothesis has been revised into a later hypothesis, but "
        "the revised hypothesis has no linked test. Tests of the original claim "
        "alone do not evaluate the revised claim. The revision must be followed "
        "by an opportunity to test it before this missing step is active."
    ),
    "contradiction_without_repair": (
        "Evidence or an interpretive judgment contradicts a particular "
        "hypothesis. That hypothesis has neither a revision nor a competing "
        "alternative that addresses the conflict. The contradiction must remain "
        "unaddressed after an opportunity to respond; merely receiving a "
        "contradictory result is not yet failure to repair."
    ),
    "premature_commitment": (
        "The same judgment informs both a hypothesis and a commitment to that "
        "hypothesis, while the hypothesis has no linked test. Commitment means "
        "treating the answer as settled or resisting revision, not simply "
        "choosing a provisional working assumption. The judgment, commitment, "
        "and untested claim must refer to the same proposed answer."
    ),
    "uninformative_test": (
        "An information-seeking test is present but has no resulting observed "
        "evidence. The missing relationship is from that test to its result. "
        "An outcome that disproves the hypothesis is still evidence. A result "
        "that has not yet had time to arrive is pending, not an active omission."
    ),
    "fixed_belief_trace": (
        "Across the entire observed prefix there is no relationship in which "
        "one hypothesis is revised into another. This is a trace-wide absence, "
        "not just a property of the latest hypothesis. An observed revision "
        "ends that absence even if the revised hypothesis subsequently stays "
        "fixed. Lack of revision does not itself show that revision was needed."
    ),
    "disconnected_evidence": (
        "An observation is isolated: no test is identified as its source and "
        "no hypothesis, judgment, or commitment is related to it. All incident "
        "relationships are missing. This is stronger than evidence non-uptake: "
        "a result linked to its generating test is not isolated even if the "
        "agent never interprets it."
    ),
    "one_sided_confirmation": (
        "A judgment informs both commitment to an answer and the corresponding "
        "hypothesis. That hypothesis has supporting evidence but no contradicting "
        "evidence in the observed prefix. The combination of commitment, support, "
        "and absent contradiction defines the motif; supportive results without "
        "commitment do not. Do not invent contrary evidence to complete a comparison."
    ),
    "precommitted_test_plan": (
        "Commitment to a proposed answer occurs before evidence collection, "
        "and hypothesis-directed testing follows that commitment. The temporal "
        "condition matters: becoming committed after considering results is a "
        "different sequence. Ordinary planning before a test is insufficient "
        "unless it also establishes commitment to the answer."
    ),
}

# Positive/negative examples distinguish relational evidence from vocabulary.
_CRITERIA = {
    "refutation_driven_belief_revision": (
        "An agent's hypothesis motivates a test; its result is interpreted and "
        "drives a materially revised hypothesis. Example: a failed prediction "
        "leads to a different mechanism. Merely rephrasing a claim is not revision."
    ),
    "fixed_hypothesis_test_tuning": (
        "The same hypothesis drives a test, the result is interpreted, and that "
        "judgment motivates an adjusted test while the hypothesis stays fixed. "
        "Example: adjust measurement resolution after noisy results. Blindly "
        "repeating a tool call does not establish evidence-guided tuning."
    ),
    "explore_then_test_transition": (
        "Exploratory information seeking produces observations before an explicit "
        "hypothesis is formed; that hypothesis then drives a targeted test. "
        "Example: survey spectra, propose a structure, test a predicted peak. "
        "A hypothesis stated before the exploratory test does not fit this order."
    ),
    "hypothesis_reranking": (
        "At least two coexisting alternative hypotheses are explicitly compared "
        "and both tested as evidence arrives. Example: compare two mechanisms "
        "using their predictions. Replacing one hypothesis with a later revision "
        "alone is not comparison of competing hypotheses."
    ),
    "evidence_led_hypothesis_generation": (
        "Observed evidence is interpreted, a hypothesis is subsequently formed "
        "from that interpretation, and it motivates a test. Example: interpret "
        "an unexpected peak, propose an impurity, test for it. An unsupported "
        "guess appearing later than unrelated evidence is insufficient."
    ),
    "convergent_multi_test_evidence": (
        "A single hypothesis is evaluated by at least three distinct tests, each "
        "with observed evidence. Example: spectroscopy, chromatography, and a "
        "reactivity measurement address the same structure. Several output lines from one "
        "test or repeated mentions of one result are not multiple tests."
    ),
    "evidence_guided_test_redesign": (
        "An interpretation of evidence motivates a new or redesigned test, "
        "which produces an observation. Example: a saturation judgment motivates "
        "a lower-concentration measurement. An unrelated next action is not "
        "evidence-guided redesign."
    ),
    "untested_claim": (
        "A hypothesis has no linked test in the observed prefix. Example: rely "
        "on an asserted mechanism across later actions without evaluating it. "
        "A newly proposed hypothesis awaiting its planned test is emerging, "
        "not an established failure. Task-provided facts are not hypotheses."
    ),
    "evidence_non_uptake": (
        "An observation never informs a hypothesis or an interpretation despite "
        "an opportunity to use it. It need not contradict the hypothesis. "
        "Example: collect a relevant measurement and then ignore it in subsequent "
        "judgments. A result that just arrived has not yet been ignored."
    ),
    "unsupported_judgment": (
        "An interpretation or evaluative claim about results has no supporting "
        "observation. Example: claim a test improved accuracy without any result. "
        "A tentative hypothesis labeled as such is not automatically a judgment."
    ),
    "stalled_revision": (
        "A hypothesis is revised but the revised claim is never tested despite "
        "subsequent opportunity. Example: adopt a new mechanism and proceed "
        "without evaluating it. A revision followed immediately by its first "
        "planned test is still emerging, not stalled."
    ),
    "contradiction_without_repair": (
        "Evidence or its interpretation contradicts a hypothesis without a "
        "revision or competing alternative addressing the conflict. Example: "
        "retain an incompatible mechanism after discussing a failed prediction. "
        "A just-arrived contradiction or an explicit alternative under "
        "investigation is not an established unresolved failure."
    ),
    "premature_commitment": (
        "An interpretation supports both a hypothesis and an explicit or "
        "behaviorally supported commitment to it, before that hypothesis is "
        "tested. Example: insist on an untested answer and refuse revision. "
        "A provisional working hypothesis or a plan alone is not commitment."
    ),
    "uninformative_test": (
        "An information-seeking test produces no observed evidence. Example: "
        "abandon a requested measurement without obtaining any observation. "
        "A pending result is emerging. A negative or unexpected result is still "
        "evidence; lack of confirmation does not make a test uninformative."
    ),
    "fixed_belief_trace": (
        "There is no hypothesis revision anywhere in the observed prefix. "
        "This is descriptive, not proof that revision was necessary. Example: "
        "the same hypothesis persists across several tests without revision. "
        "Before any hypothesis or opportunity for revision, use emerging or "
        "absent. Once any revision occurs this trace-wide absence is resolved."
    ),
    "disconnected_evidence": (
        "An observation has no identifiable relationship to a test, hypothesis, "
        "judgment, or commitment. Example: an isolated measurement is neither "
        "attributed to an action nor used afterward. An unused result linked to "
        "the test that produced it is not disconnected evidence."
    ),
    "one_sided_confirmation": (
        "A judgment supports commitment to a hypothesis with supporting evidence "
        "but no contradicting evidence in the prefix. Example: treat favorable "
        "results as conclusive and refuse revision. Mere absence of contrary "
        "results without commitment is insufficient; do not invent contradictions."
    ),
    "precommitted_test_plan": (
        "Commitment to an answer precedes evidence collection and is followed "
        "by hypothesis-linked testing. Example: insist on the answer before "
        "seeing results, then test that claim. An ordinary provisional test plan "
        "before collecting data is not commitment."
    ),
}


@dataclass(frozen=True)
class Motif:
    name: str
    kind: str
    family: str
    description: str
    graph_description: str
    criteria: str


MOTIFS = tuple(
    Motif(
        name=name,
        kind=kind,
        family=next(family for family, members in families.items() if name in members),
        description=description,
        graph_description=_GRAPH_DESCRIPTIONS[name],
        criteria=_CRITERIA[name],
    )
    for kind, descriptions, families in (
        ("productive", SUBGRAPH_DESCRIPTIONS, SUBGRAPH_FAMILIES),
        ("antipattern", ANTIPATTERN_DESCRIPTIONS, ANTIPATTERN_FAMILIES),
    )
    for name, description in descriptions.items()
)


def motif_questions() -> list[dict]:
    """Return all 18 named Decisions choice questions in registry order."""
    return [
        {
            "type": "choice",
            "name": motif.name,
            "instructions": (
                f"Classify {motif.name} ({motif.kind}; {motif.family}).\n"
                f"Graph description: {motif.graph_description}\n"
                f"Original graph notation: {motif.description}\n"
                f"Temporal criteria: {motif.criteria}\n"
                "Use this full description and the supplied node/relationship "
                "glossaries; the motif name is only an identifier. Require the "
                "specified links to concern the same claims, tests, and results, "
                "rather than unrelated events scattered through the trajectory. "
                "Apply the annotation rules supplied with the trajectory. "
                "Classify at the end of this prefix. If instances coexist, an "
                "unresolved active instance takes precedence over a resolved one."
            ),
            "choices": [
                {"value": status, "description": description}
                for status, description in STATUS_DESCRIPTIONS.items()
            ],
        }
        for motif in MOTIFS
    ]
