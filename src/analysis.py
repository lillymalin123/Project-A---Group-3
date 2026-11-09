from __future__ import annotations


def count_values(values: list[float]) -> int:
    return len(values)


def sum_values(values: list[float]) -> float:
    return float(sum(values))


def average_values(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum_values(values) / count_values(values)
