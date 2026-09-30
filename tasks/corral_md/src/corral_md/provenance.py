"""Read controlled-execution identifiers from public submission layouts."""

from collections.abc import Mapping


def provenance_identifiers(
    manifest: Mapping, *, required: bool = False
) -> dict[str, str]:
    """Accept top-level or nested IDs without changing submitted metadata.

    These are identifiers only. The trusted backend must still attest the run
    and check its artifacts; a submitted receipt cannot do that.
    """
    nested = manifest.get("provenance", {})
    if not isinstance(nested, Mapping):
        raise ValueError("manifest.provenance must be a JSON object")
    result = {}
    for key in ("run_id", "action_id", "release_id"):
        for location, fields in (
            ("manifest", manifest),
            ("manifest.provenance", nested),
        ):
            if key not in fields:
                continue
            value = fields[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{location}.{key} must be a nonempty string")
            if key in result and result[key] != value:
                raise ValueError(
                    f"Conflicting manifest.{key} and manifest.provenance.{key}"
                )
            result[key] = value
    if required:
        missing = [key for key in ("run_id", "action_id") if key not in result]
        if missing:
            raise ValueError(
                "Missing controlled-execution identifiers: "
                + ", ".join(missing)
                + "; copy them from run_verified_md into manifest.provenance "
                "or the manifest's top level"
            )
    return result
