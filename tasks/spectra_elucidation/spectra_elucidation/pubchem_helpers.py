import asyncio
import random
from typing import Any
from urllib.parse import quote

import aiohttp
import backoff
from loguru import logger


class PubChem:
    """
    PubChem handler to retrieve compound data from the PubChem PUG REST API.
    Automatically converts the input to a CID and retrieves the data for that compound from PubChem.

    Example:
        >>> pubchem = await PubChem.create("2244")
        >>> isomers = await pubchem._get_number_atoms()
            21
    """

    def __init__(self):
        self.cid = None
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.long_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?response_type=display&heading="
        self.complete_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
        )
        logger.info(f"Initialized PubChem handler for CID {self.cid}")

    @classmethod
    async def create(cls, compound: str):
        """
        Factory method to create a PubChem handler with a compound identifier.
        Automatically converts the input to a CID and retrieves the data for that compound from PubChem.

        Args:
            compound (str): Any type of compound identifier (CID, SMILES, InChI, etc.)

        Returns:
            PubChem: Instance of PubChem handler with the compound data
        """
        self = cls()
        self.cid = await self._normalize_to_cid(compound)
        logger.info(f"Initialized PubChem handler for CID {self.cid}")
        return self

    @backoff.on_exception(
        backoff.expo, (aiohttp.ClientError, ValueError), max_tries=10, max_time=30
    )
    async def _normalize_to_cid(self, compound: str) -> str | None:
        """
        Convert any compound identifier to PubChem CID asynchronously.
        Tries different approaches to identify the compound format and get its CID.

        Args:
            compound (str): Any type of compound identifier (CID, SMILES, InChI, etc.)

        Returns:
            str: PubChem CID if found, None if not found
        """
        namespaces = ["cid"] if compound.isdigit() else ["name", "smiles", "inchi"]
        for namespace in namespaces:
            encoded_compound = quote(compound, safe="")
            url = f"{self.base_url}/compound/{namespace}/{encoded_compound}/cids/JSON"
            try:
                data = await self.get_data_from_url(url)
                cids = data.get("IdentifierList", {}).get("CID", [])
                if cids:
                    return str(cids[0])
            except Exception:
                continue

        logger.error(f"Invalid compound identifier: {compound}")
        raise ValueError(
            "Invalid compound identifier. Only CID, name, SMILES or InChI are supported."
        )

    @classmethod
    async def get_compound_isomers_by_formula(
        cls, formula: str, limit: int = 5
    ) -> list[str]:
        """Retrieve isomeric SMILES for compounds matching a molecular formula.

        Args:
            formula: Molecular formula accepted by PubChem's fast-formula search.
            limit: Maximum number of isomers to return. ``0`` returns all matches.

        Returns:
            A shuffled list of isomeric SMILES strings.
        """
        if not formula.strip():
            raise ValueError("Molecular formula cannot be empty")
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        self = cls()
        encoded_formula = quote(formula.strip(), safe="")
        url = f"{self.base_url}/compound/fastformula/{encoded_formula}/cids/JSON"
        try:
            data = await self.get_data_from_url(url)
            isomer_cids = data.get("IdentifierList", {}).get("CID", [])
        except Exception as e:
            raise ValueError(f"No compound isomers found for {formula}: {e}") from e

        random.shuffle(isomer_cids)
        if limit:
            isomer_cids = isomer_cids[:limit]

        isomers = []
        for batch_start in range(0, len(isomer_cids), 100):
            cid_batch = isomer_cids[batch_start : batch_start + 100]
            cid_path = ",".join(map(str, cid_batch))
            property_url = (
                f"{self.base_url}/compound/cid/{cid_path}/property/IsomericSMILES/JSON"
            )
            try:
                property_data = await self.get_data_from_url(property_url)
                properties = property_data.get("PropertyTable", {}).get(
                    "Properties", []
                )
                isomers.extend(
                    property_["SMILES"]
                    for property_ in properties
                    if property_.get("SMILES")
                )
            except Exception as e:
                logger.warning(f"Could not get SMILES for CIDs {cid_path}: {e}")

        logger.info(f"Found {len(isomers)} compound isomers for formula {formula}")
        return isomers

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=10,
        max_time=60,
    )
    async def get_data_from_url(
        self, record_url: str | None
    ) -> dict[Any, Any]:
        """
        Fetch compound data from PubChem REST API using a specific URL

        Args:
            record_url (str): URL to fetch compound data from

        Returns:
            dict: Compound data in JSON format
        """
        if not record_url:
            raise ValueError("No record URL provided.")
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(record_url) as record_response:
                    record_response.raise_for_status()
                    return await record_response.json()
            except Exception as e:
                raise e

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=10,
        max_time=120,
    )
    async def get_compound_data(
        self, record_url: str | None = str
    ) -> dict[Any, Any]:
        """
        Fetch compound data from PubChem REST API combining basic record and detailed data

        Args:
            record_url (str): URL to fetch compound data from

        Returns:
            dict: Combined compound data in JSON format containing both record and detailed information
        """
        if not record_url:
            raise ValueError("No record URL provided.")
        record_url = (
            self.base_url + record_url[0] + str(self.cid) + record_url[1]
            if record_url
            else None
        )
        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(record_url) as record_response:
                    record_response.raise_for_status()
                    return await record_response.json()

            except Exception as e:
                raise e

    async def _get_isomeric_smiles(self) -> str | None:
        """
        Get the isomeric SMILES for a compound from PubChem.

        Returns:
            str: Isomeric SMILES of the compound.

        Raises:
            ValueError: If the isomeric SMILES could not be retrieved

        Example:
            >>> await self._get_isomeric_smiles()
                'CCO'
        """
        url = [
            "/compound/cid/",
            "/property/IsomericSMILES/JSON",
        ]
        logger.info(f"Getting isomeric SMILES for CID {self.cid}")
        try:
            data = (await self.get_compound_data(url))["PropertyTable"]["Properties"][
                0
            ]["SMILES"]
        except Exception as e:
            logger.error(f"Failed to extract Isomeric SMILES: {e!s}")
            raise ValueError(f"Failed to extract Isomeric SMILES: {e!s}") from e
        logger.info(f"Isomeric SMILES for CID {self.cid}: {data}")
        return data

    async def _get_compound_formula(self) -> str | None:
        """
        Get the formula of a compound from PubChem.

        Returns:
            str: Formula of the compound.

        Raises:
            ValueError: If the formula could not be retrieved

        Example:
            >>> await self._get_compound_formula()
                'C2H6O'
        """
        url = [
            "/compound/cid/",
            "/property/MolecularFormula/JSON",
        ]
        logger.info(f"Getting compound formula for CID {self.cid}")
        try:
            data = (await self.get_compound_data(url))["PropertyTable"]["Properties"][
                0
            ]["MolecularFormula"]
        except Exception as e:
            logger.error(f"No compound formula found. {e}")
            raise ValueError(f"No compound formula found. {e}") from e
        logger.info(f"Compound formula for CID {self.cid}: {data}")
        return data

    async def _get_compound_isomers(self, limit: int = 5) -> list[str]:
        """
        Get the compound isomers for a compound from PubChem.
        This function can take some time depending on the number of isomers.
        Returns a maximum of 5 isomers by default.
        Shuffles the results each time to provide different isomers.

        Args:
            limit (int, optional): Maximum number of isomers to return. Defaults to 5.

        Returns:
            list: List of compound isomers (limited to the specified number).

        Raises:
            ValueError: If the isomers could not be retrieved

        Example:
            >>> await self._get_compound_isomers()
                ['CCO', 'COC']
            >>> await self._get_compound_isomers(limit=5)
                ['COC', 'CCO', 'CC[O-]', 'C[CH-]O', 'C[O-]C']
        """

        url = [
            "/compound/fastformula/",
            "/cids/JSON",
        ]
        logger.info(f"Getting compound isomers for CID {self.cid} (limit: {limit})")
        formula = await self._get_compound_formula()
        url = self.base_url + url[0] + quote(str(formula)) + url[1]
        try:
            isomers_cids = (await self.get_data_from_url(url))["IdentifierList"]["CID"]
            # Shuffle the isomers to get different ones each time
            random.shuffle(isomers_cids)
            # Limit the number of isomers to process
            isomers_cids = isomers_cids[:limit]
            data = []
            for i in isomers_cids:
                await asyncio.sleep(0.5)
                self.cid = i
                try:
                    # Isomeric SMILES to capture enantiomers
                    smiles = await self._get_isomeric_smiles()
                    if smiles:
                        data.append(smiles)
                except ValueError as ve:
                    logger.warning(f"Could not get SMILES for CID {i}: {ve}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error getting SMILES for CID {i}: {e}")
                    continue
        except Exception as e:
            logger.error(f"No compound isomers found. {e}")
            raise ValueError(f"No compound isomers found. {e}") from e
        logger.info(f"Found {len(data)} compound isomers for the given formula")
        return data
