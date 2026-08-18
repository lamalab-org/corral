"""Errors raised while projecting and querying benchmark reports."""


class BenchmarkReportError(Exception):
    """Base class for reporting failures."""


class TaskNotFoundError(BenchmarkReportError):
    """Raised when a requested task is absent from benchmark results."""


class InsufficientTrialsError(BenchmarkReportError):
    """Raised when a calculation requires more trials than are available."""


class NoResultsError(BenchmarkReportError):
    """Raised when a metric cannot be calculated without results."""
