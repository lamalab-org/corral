from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import MutableSequence, Sequence


class HealthCheck(ABC):
    """Abstract base class for tool health checks."""

    @abstractmethod
    def is_healthy(self) -> tuple[bool, str | None]:
        """Check if the tool is healthy and ready to use.

        Returns:
            tuple[bool, str | None]: A tuple containing:
                - bool: True if healthy, False otherwise
                - str | None: Error message if unhealthy, None if healthy
        """


class HealthCheckMixin:
    """Mixin to add health check capabilities to tools."""

    def __init__(self) -> None:
        self._health_checks: MutableSequence[HealthCheck] = []

    def add_health_check(self, check: HealthCheck) -> None:
        """Add a health check to the tool.

        Args:
            check: HealthCheck instance to add
        """
        if not isinstance(check, HealthCheck):
            raise TypeError("Check must be an instance of HealthCheck")
        self._health_checks.append(check)

    def remove_health_check(self, check: HealthCheck) -> bool:
        """Remove a health check from the tool.

        Args:
            check: HealthCheck instance to remove

        Returns:
            bool: True if check was removed, False if not found
        """
        try:
            self._health_checks.remove(check)
            return True
        except ValueError:
            return False

    def is_healthy(self) -> tuple[bool, Sequence[str]]:
        """Run all health checks.

        Returns:
            tuple[bool, Sequence[str]]: A tuple containing:
                - bool: True if all checks pass, False otherwise
                - Sequence[str]: List of error messages from failed checks
        """
        errors: list[str] = []
        for check in self._health_checks:
            try:
                is_healthy, error = check.is_healthy()
                if not is_healthy and error is not None:
                    errors.append(error)
            except Exception as e:
                errors.append(f"Health check failed with error: {e!s}")

        return len(errors) == 0, errors
