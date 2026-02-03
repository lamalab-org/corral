"""
Configuration module for retrosynthesis database connections.

This module manages database credentials and connection settings, reading from
environment variables for security. Never commit actual credentials to version control.

Environment Variables:
    RETRO_DB_HOST: Database host (default: localhost)
    RETRO_DB_PORT: Database port (default: 5432)
    RETRO_DB_NAME: Database name (default: reactions_production_db)
    RETRO_DB_USER: Database user (default: postgres)
    RETRO_DB_PASSWORD: Database password (default: postgres)

    STAGING_DB_HOST: Staging database host (default: localhost)
    STAGING_DB_PORT: Staging database port (default: 5432)
    STAGING_DB_NAME: Staging database name (default: reactions_raw_db)
    STAGING_DB_USER: Staging database user (default: postgres)
    STAGING_DB_PASSWORD: Staging database password (default: postgres)
"""

import os
from typing import Any


def get_db_config() -> dict[str, Any]:
    """
    Get the production database configuration from environment variables.

    Returns:
        Dict[str, Any]: Database configuration dictionary with keys:
            - host: Database host
            - port: Database port
            - database: Database name
            - user: Database user
            - password: Database password

    Note:
        For local development, these default to standard PostgreSQL localhost settings.
        For production/staging environments, set the appropriate environment variables.
    """
    return {
        "host": os.environ.get("RETRO_DB_HOST", "localhost"),
        "port": int(os.environ.get("RETRO_DB_PORT", "5432")),
        "database": os.environ.get("RETRO_DB_NAME", "reactions_production_db"),
        "user": os.environ.get("RETRO_DB_USER", "postgres"),
        "password": os.environ.get("RETRO_DB_PASSWORD", "postgres"),
    }


def get_staging_db_config() -> dict[str, Any]:
    """
    Get the staging database configuration from environment variables.

    Returns:
        Dict[str, Any]: Database configuration dictionary with keys:
            - host: Database host
            - port: Database port
            - database: Database name
            - user: Database user
            - password: Database password

    Note:
        This is used by Phase B production database builder to read from Phase A staging data.
    """
    return {
        "host": os.environ.get("STAGING_DB_HOST", "localhost"),
        "port": int(os.environ.get("STAGING_DB_PORT", "5432")),
        "database": os.environ.get("STAGING_DB_NAME", "reactions_raw_db"),
        "user": os.environ.get("STAGING_DB_USER", "postgres"),
        "password": os.environ.get("STAGING_DB_PASSWORD", "postgres"),
    }


# For backward compatibility and convenience
DB_CONFIG = get_db_config()
STAGING_DB_CONFIG = get_staging_db_config()
