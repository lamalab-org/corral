"""Public, versioned AFM execution contract shared by every agent harness."""

from pathlib import Path

CONTRACT_VERSION = "managed-afm-v1"

INSTRUCTIONS = """AFM execution contract: managed-afm-v1
These tools control the real, named instrument; shell/workspace isolation does not
simulate hardware. Corral checks machine/process/head identity, stops inherited
manual scans, applies and reads back initial settings before the trial, and stops
remaining activity at its end. The initial snapshot below is observed state, not
the task's target settings. Do not operate the GUI or use other control processes.

Use Code_Executor with nanosurf.SPM() for the configured connection. StartFrameUp(),
StartFrameDown(), and acquire_frame() block until a fresh, stable, parseable NID is
saved and the instrument is idle; each returns its absolute path. Do not probe
other application constructors, import raw COM, change the output directory, or
start continuous scans. On InstrumentUnavailable, stop issuing hardware commands;
operator recovery is required. Script errors with successful cleanup may be corrected.

Use the exact gains and settings in the task. Tapping tasks require Nanosurf
PhaseContrast (4); contact tasks require LateralForce (9). Keep the initial center
and rotation unless the task requests movement. API distances are metres; parameter
snapshots use nm, times use seconds, and angles use degrees. Contact setpoints use V
(SetPointForceUnitMode=0), tapping amplitude setpoints use percent (e.g. 70, not 0.7).

Submit the required number of distinct raw NID acquisitions in the requested order.
Additional acquisitions are allowed while the instrument remains healthy; only the
submitted acquisitions are scored. Minimize unnecessary scans. Use the absolute
paths returned by the AFM tools, preserving Windows or POSIX spelling. Do not build
paths from a harness shell's working directory. Finish with submit_answer.

Metrics use raw, unlevelled data: h is Forward/Z-Axis in nm; Ra=mean(abs(h-mean(h)))
and Rq=sqrt(mean((h-mean(h))**2)). For friction, f=(Forward-Backward)/2 in V using
NSFopen's stored array orientation without flipping. Average friction is mean(f);
average friction magnitude is mean(abs(f)); RMS friction is sqrt(mean(f**2)),
without subtracting its mean. Image_Analyzer implements these definitions. A raw
channel selection or dynamic_code changes only returned image_data, never metrics.
Numeric settings and measurements have 1% relative tolerance; zero targets, counts,
modes and units must match exactly. Avoid premature rounding. Percentage changes
use the reported values; return JSON null when the reference value is zero.
Document_Retrieval returns the bundled API reference without a network or API key.
"""


def reference(query: str) -> str:
    """Return the complete small reference, without pretending to search a manual."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Provide a nonempty AFM API question")
    return (Path(__file__).with_name("afm_api_reference.md")).read_text(
        encoding="utf-8"
    )
