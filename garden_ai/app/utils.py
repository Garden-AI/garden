"""Common utility functions for CLI commands."""


def parse_list(value: str | None) -> list[str]:
    """Parse a comma-separated string into a list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str | None) -> list[int]:
    """Parse a comma-separated string of integers."""
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]
