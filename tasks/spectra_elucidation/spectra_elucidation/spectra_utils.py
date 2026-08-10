import asyncio
import json
import os
import re
import shutil
import subprocess
import textwrap
import threading
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import ClassVar

import aiohttp
import backoff
import requests
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

_ELEMENT_PAT = re.compile(r"([A-Z][a-z]?)(\d*)")
_PAREN_PAT = re.compile(r"\(([^()]*)\)(\d*)")
_DOT_PAT = re.compile(r"·|\.")


def _parse_simple(formula: str) -> dict[str, int]:
    """
    Parse an already-expanded, dot-free formula into {element: count}.
    """
    counts = Counter()
    for el, cnt in _ELEMENT_PAT.findall(formula):
        counts[el] += int(cnt or 1)
    return counts


def _expand_parentheses(formula: str) -> str:
    """
    Recursively expand parentheses so that C6H5(CH3) becomes C6H5C1H3 etc.
    """
    while True:
        m = _PAREN_PAT.search(formula)
        if not m:
            return formula
        inner, mult = m.groups()
        mult = int(mult or 1)
        expanded = "".join(
            f"{el}{int(cnt or 1) * mult}" for el, cnt in _ELEMENT_PAT.findall(inner)
        )
        formula = formula[: m.start()] + expanded + formula[m.end() :]


def parse_molecular_formula(formula: str) -> dict[str, int]:
    """
    Parse molecular formula into an element-count mapping, handling
    parentheses and dot adducts.
    """
    parts = _DOT_PAT.split(formula.replace(" ", ""))
    total = Counter()
    for part in parts:
        expanded = _expand_parentheses(part)
        total += _parse_simple(expanded)
    return dict(total)


def enumerate_fragments_from_smiles(
    smi: str,
    max_cuts: int = 2,
    skip_ring_bonds: bool = True,
    min_heavy_atoms: int = 2,
    max_combos: int = 100000,
):
    """
    Generate fragment SMILES by breaking up to `max_cuts` bonds.
    Returns a sorted list of unique canonical SMILES of fragments.
    Preserves ions by setting formal charges instead of adding hydrogens.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # Candidate bonds: heavy-atom bonds only; optionally skip ring bonds
    cand_bond_idxs = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        if skip_ring_bonds and b.IsInRing():
            continue
        cand_bond_idxs.append(b.GetIdx())

    # Early exit: nothing to cut
    if not cand_bond_idxs or max_cuts < 1:
        return []

    uniq = set()
    tried = 0

    for ncuts in range(1, min(max_cuts, len(cand_bond_idxs)) + 1):
        for cutset in combinations(cand_bond_idxs, ncuts):
            tried += 1
            if tried > max_combos:
                # Safety valve against combinatorial explosion
                break

            rw = Chem.RWMol(mol)
            cut_atom_info = {}

            # Count how many bonds each atom will lose
            for bidx in cutset:
                b = rw.GetBondWithIdx(bidx)
                begin_idx = b.GetBeginAtomIdx()
                end_idx = b.GetEndAtomIdx()

                cut_atom_info[begin_idx] = cut_atom_info.get(begin_idx, 0) + 1
                cut_atom_info[end_idx] = cut_atom_info.get(end_idx, 0) + 1

            # Before removing bonds, adjust formal charges on cut atoms
            for atom_idx, bonds_lost in cut_atom_info.items():
                atom = rw.GetAtomWithIdx(atom_idx)
                current_charge = atom.GetFormalCharge()

                # Adjust charge based on bonds lost
                new_charge = current_charge - bonds_lost
                atom.SetFormalCharge(new_charge)

            # Now remove the selected bonds
            for bidx in sorted(
                cutset, reverse=True
            ):  # Remove in reverse order to maintain indices
                b = rw.GetBondWithIdx(bidx)
                rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

            # Get connected components as fragments
            try:
                frags = Chem.rdmolops.GetMolFrags(
                    rw.GetMol(), asMols=True, sanitizeFrags=False
                )

                for frag in frags:
                    # Filter tiny pieces
                    if (
                        sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() > 1)
                        < min_heavy_atoms
                    ):
                        continue

                    try:
                        # Try to sanitize the fragment with ionic charges
                        Chem.SanitizeMol(frag, catchErrors=False)
                        smi_frag = Chem.MolToSmiles(
                            frag, isomericSmiles=True, canonical=True
                        )
                        uniq.add(smi_frag)
                    except Exception:
                        # If sanitization fails with ionic charges, try without sanitization
                        try:
                            smi_frag = Chem.MolToSmiles(
                                frag, isomericSmiles=True, canonical=True
                            )
                            uniq.add(smi_frag)
                        except Exception:
                            continue  # Skip this fragment if it can't be processed

            except Exception:
                # If fragmentation fails completely, skip this cut combination
                continue

        else:
            continue
        break  # broke due to max_combos

    return sorted(uniq)


def format_hsqc_spectrum(zones_dict: dict) -> str:
    """
    Parse HSQC spectra data from dictionary format to standard NMR notation.

    Args:
        zones_dict (dict): Dictionary containing zones data with signals
        frequency (str): NMR frequency (default: "600 MHz")
        solvent (str): NMR solvent (default: "DMSO-d6")

    Returns:
        str: Formatted HSQC notation string
    """
    if "zones" not in zones_dict or "values" not in zones_dict["zones"]:
        return "Invalid input format"

    signals_data = []

    # Extract signals from each zone
    for zone in zones_dict["zones"]["values"]:
        if "signals" in zone:
            for signal in zone["signals"]:
                # Extract chemical shifts
                h_delta = signal["x"]["delta"]  # 1H chemical shift
                c_delta = signal["y"]["delta"]  # 13C chemical shift

                # Count hydrogen atoms from x.atoms (1H nuclei)
                h_count = len(signal["x"]["atoms"]) if "atoms" in signal["x"] else 1

                # Store the data for sorting
                signals_data.append(
                    {"h_delta": h_delta, "c_delta": c_delta, "h_count": h_count}
                )

    # Sort signals by 1H chemical shift (ascending order)
    signals_data.sort(key=lambda x: x["h_delta"])

    # Format each signal
    formatted_signals = []
    for signal in signals_data:
        h_delta = signal["h_delta"]
        c_delta = signal["c_delta"]
        h_count = signal["h_count"]

        # Format chemical shifts (remove trailing zeros)
        h_str = f"{h_delta:.2f}".rstrip("0").rstrip(".")
        c_str = f"{c_delta:.1f}".rstrip("0").rstrip(".")

        # Create the signal string
        signal_str = f"{h_str}/{c_str} ({h_count}H)"
        formatted_signals.append(signal_str)

    # Combine all signals
    signals_part = ", ".join(formatted_signals)

    # Create final HSQC string
    return f"HSQC: delta H/delta C {signals_part}."


def convert_ms_spectrum_to_string(spectrum_data):
    """
    Convert MS spectrum data from JSON format to string format.

    Args:
        spectrum_data (list): List of dictionaries with 'x' (m/z) and 'y' (intensity) keys

    Returns:
        str: Formatted string in the format "m/z 100.1 (intensity 500), 101.2 (intensity 450), ..."
    """
    if not spectrum_data:
        return ""

    # Convert each data point to the desired format
    formatted_peaks = []
    for peak in spectrum_data:
        mz = peak["x"]
        intensity = peak["y"]
        formatted_peaks.append(f"{mz} (intensity {intensity})")

    # Join all peaks with ", " and prepend "m/z "
    return "m/z " + ", ".join(formatted_peaks)

def predict_isotopic_distribution(smiles: str, ionization: str | None = None) -> list[dict]:
    """
    Predict the isotopic distribution for a molecule using the same implementation
    as the Modal deployment.

    Args:
        smiles: SMILES string of the molecule.
        ionization: Reserved for API compatibility with the deployed endpoint.

    Returns:
        List of isotopic distribution peaks returned by the isotopic-distribution
        JavaScript package.
    """
    del ionization

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        msg = "Invalid SMILES"
        raise ValueError(msg)

    formula = rdMolDescriptors.CalcMolFormula(mol)
    logger.debug(f"Calculating isotopic distribution for formula: {formula}")
    js = textwrap.dedent(
        f"""
        import {{ IsotopicDistribution }} from "isotopic-distribution";

        const isotopicDistribution = new IsotopicDistribution("{formula}");
        console.log(JSON.stringify(isotopicDistribution.getPeaks()));
        """
    )
    node_project_dir = os.environ.get("CORRAL_SPECTRA_JS_DIR", "/srv/js")
    cwd = node_project_dir if Path(node_project_dir).is_dir() else None
    node_executable = shutil.which("node") or "node"
    out = subprocess.check_output(
        [node_executable, "--input-type=module", "-e", js],
        cwd=cwd,
        text=True,
        stderr=subprocess.STDOUT,
    )
    return json.loads(out)


def make_api_call(url: str, payload: dict) -> dict:
    """
    Make a POST request to the specified URL with the given payload.

    Args:
        url (str): The URL to which the request is sent.
        payload (dict): The data to be sent in the request body.

    Returns:
        dict: The JSON response from the server.
    """
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()

class SpectraAPI:
    """Class for handling spectral predictions from the NMR and IR APIs

    Attributes:
        IR_BASE_URL: str - Base URL for the IR prediction API
        NMR_BASE_URL: str - Base URL for the NMR prediction API
    """

    IR_BASE_URL: str = "https://ir.cheminfo.org/v1/ir"
    NMR_BASE_URL: str = "https://nmr-prediction.service.zakodium.com/v1/predict"
    VALID_SPECTRUM_TYPES: ClassVar = {"carbon", "proton"}

    _request_lock = threading.Lock()
    _last_request_time: float = 0
    _min_request_interval = 1
    _timeout = 60

    @staticmethod
    def format_c13_nmr(json_response: dict) -> str:
        """Format C13 NMR response to literature format

        Args:
            json_response: C13 NMR prediction response

        Returns:
            str: Formatted C13 NMR prediction
        """
        shifts = sorted(
            [signal["delta"] for signal in json_response["data"]["signals"]],
            reverse=True,
        )
        return f"Deltas: {', '.join(f'{shift:.2f}' for shift in shifts)}"

    @staticmethod
    def format_h_nmr(json_response: dict) -> str:
        """Format 1H NMR response to literature format

        Args:
            json_response: 1H NMR prediction response

        Returns:
            str: Formatted 1H NMR prediction
        """
        formatted_signals = []

        for range_data in json_response["data"]["ranges"]:
            signal = range_data["signals"][0]
            delta = signal["delta"]
            integration = range_data["integration"]
            multiplicity = signal.get("multiplicity", "m")

            j_values = [
                round(j["coupling"], 1) for j in signal.get("js", []) if "coupling" in j
            ]

            signal_str = f"{delta:.2f}"
            if multiplicity != "m" and j_values:
                j_str = ", ".join(f"{j:.1f}" for j in j_values)
                signal_str += f" ({multiplicity}, J = {j_str} Hz, {integration}H)"
            else:
                signal_str += f" ({multiplicity}, {integration}H)"

            formatted_signals.append(signal_str)

        return f"Deltas {', '.join(formatted_signals)}."

    @staticmethod
    def format_ir(json_response: dict) -> str:
        """Format IR response to literature format

        Args:
            json_response: IR prediction response

        Returns:
            str: Formatted IR prediction

        Raises:
            ValueError: If invalid method provided
        """
        wavenumbers = sorted(
            [
                round(mode["wavenumber"])
                for mode in json_response["modes"]
                if not mode["imaginary"]
            ],
            reverse=True,
        )
        wavenumbers = sorted(
            [
                round(mode["wavenumber"])
                for mode in json_response["modes"]
                if not mode["imaginary"] and mode["wavenumber"] >= 1500
            ],
            reverse=True,
        )
        return f"Wavenumbers (cm-1): {', '.join(map(str, wavenumbers))}"

    @staticmethod
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=1,
        max_time=60,
        giveup=lambda e: isinstance(e, aiohttp.ClientResponseError)
        and e.status in {400, 401, 403, 404},
        jitter=backoff.full_jitter,
        base=2,
    )
    async def get_prediction_async(
        session: aiohttp.ClientSession,
        smiles: str,
        prediction_type: str,
        spectrum_type: str = "carbon",
        method: str = "GFN2xTB",
    ) -> dict:
        """
        Get spectral prediction for a given SMILES string

        Args:
            session: aiohttp.ClientSession - Aiohttp client session
            smiles: str - SMILES string of the molecule
            prediction_type: str - Type of prediction ("nmr" or "ir")
            spectrum_type: str - Type of NMR spectrum (carbon or proton), only for NMR
            method: str - IR prediction method, only for IR

        Returns:
            Dict: Prediction response

        Raises:
            ValueError: If invalid prediction_type or spectrum_type provided
            aiohttp.ClientError: If the request fails after all retries
        """
        # Tool calls run in worker threads, each with its own event loop. Reserve
        # request slots under a thread lock so rate limiting remains safe across
        # those loops without binding an asyncio.Lock to one of them.
        with SpectraAPI._request_lock:
            current_time = time.monotonic()
            request_time = max(
                current_time,
                SpectraAPI._last_request_time + SpectraAPI._min_request_interval,
            )
            SpectraAPI._last_request_time = request_time

        delay = request_time - current_time
        if delay > 0:
            await asyncio.sleep(delay)

        if prediction_type == "nmr":
            if spectrum_type not in SpectraAPI.VALID_SPECTRUM_TYPES:
                raise ValueError(
                    f"spectrum_type must be one of {SpectraAPI.VALID_SPECTRUM_TYPES}"
                )
            url = f"{SpectraAPI.NMR_BASE_URL}/{spectrum_type}"
            payload = {"smiles": smiles}
        elif prediction_type == "ir":
            url = SpectraAPI.IR_BASE_URL
            payload = {"smiles": smiles, "method": method}
        else:
            raise ValueError("prediction_type must be 'nmr' or 'ir'")

        async with session.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=SpectraAPI._timeout),
        ) as response:
            try:
                response.raise_for_status()
                return await response.json()
            except Exception as e:
                logger.error(
                    f"{prediction_type.upper()} prediction failed - Status: {response.status}"
                )
                logger.error(f"Headers: {response.headers}")
                logger.error(f"Response body: {await response.text()}")
                raise e

    @classmethod
    async def get_all_predictions(cls, smiles: str) -> dict[str, str]:
        """
        Get all spectral predictions for a molecule.
        If some predictions fail, still returns the successful ones.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dict containing formatted spectral predictions
        """
        async with aiohttp.ClientSession() as session:
            logger.debug(f"Getting predictions for {smiles}")
            tasks = [
                cls.get_prediction_async(session, smiles, "nmr", "carbon"),
                cls.get_prediction_async(session, smiles, "nmr", "proton"),
                cls.get_prediction_async(session, smiles, "ir"),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            predictions = {}

            # Process carbon NMR result
            if not isinstance(results[0], BaseException):
                predictions["c13_nmr"] = SpectraAPI.format_c13_nmr(dict(results[0]))
            else:
                logger.error(f"Failed to retrieve C13 NMR: {results[0]!s}")
                predictions["c13_nmr"] = "C13 NMR prediction failed"

            # Process proton NMR result
            if not isinstance(results[1], BaseException):
                predictions["h_nmr"] = SpectraAPI.format_h_nmr(dict(results[1]))
            else:
                logger.error(f"Failed to retrieve H NMR: {results[1]!s}")
                predictions["h_nmr"] = "H NMR prediction failed"

            # Process IR result
            if not isinstance(results[2], BaseException):
                predictions["ir"] = SpectraAPI.format_ir(dict(results[2]))
            else:
                logger.error(f"Failed to retrieve IR: {results[2]!s}")
                predictions["ir"] = "IR prediction failed"

            return predictions

    @classmethod
    async def get_c13_nmr_prediction(cls, smiles: str) -> str:
        """
        Get only the C13 NMR spectral prediction for a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Formatted C13 NMR prediction or error message
        """
        async with aiohttp.ClientSession() as session:
            try:
                result = await cls.get_prediction_async(
                    session, smiles, "nmr", "carbon"
                )
                return SpectraAPI.format_c13_nmr(result)
            except Exception as e:
                logger.error(f"Failed to retrieve C13 NMR: {e!s}")
                return "C13 NMR prediction failed"

    @classmethod
    async def get_h_nmr_prediction(cls, smiles: str) -> str:
        """
        Get only the H NMR spectral prediction for a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Formatted H NMR prediction or error message
        """
        async with aiohttp.ClientSession() as session:
            try:
                result = await cls.get_prediction_async(
                    session, smiles, "nmr", "proton"
                )
                return SpectraAPI.format_h_nmr(result)
            except Exception as e:
                logger.error(f"Failed to retrieve H NMR: {e!s}")
                return "H NMR prediction failed"

    @classmethod
    async def get_ir_prediction(cls, smiles: str) -> str:
        """
        Get only the IR spectral prediction for a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Formatted IR prediction or error message
        """
        async with aiohttp.ClientSession() as session:
            try:
                result = await cls.get_prediction_async(session, smiles, "ir")
                return SpectraAPI.format_ir(result)
            except Exception as e:
                logger.error(f"Failed to retrieve IR: {e!s}")
                return "IR prediction failed"

    @classmethod
    async def get_raw_h_nmr_prediction(cls, smiles: str) -> dict | str:
        """
        Get the raw H NMR spectral prediction for a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Raw H NMR prediction or error message
        """
        async with aiohttp.ClientSession() as session:
            try:
                return await cls.get_prediction_async(
                    session, smiles, "nmr", "proton"
                )
            except Exception as e:
                logger.error(f"Failed to retrieve raw H NMR: {e!s}")
                return "Raw H NMR prediction failed"

    @classmethod
    async def get_raw_c_nmr_prediction(cls, smiles: str) -> dict | str:
        """
        Get the raw C NMR spectral prediction for a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Raw H NMR prediction or error message
        """
        async with aiohttp.ClientSession() as session:
            try:
                return await cls.get_prediction_async(
                    session, smiles, "nmr", "carbon"
                )
            except Exception as e:
                logger.error(f"Failed to retrieve raw H NMR: {e!s}")
                return "Raw H NMR prediction failed"
