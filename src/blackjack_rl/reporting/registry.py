"""Small registry: new chart modules register generators without changing the CLI."""

from dataclasses import dataclass


@dataclass
class Chart:
    name: str
    figure: object
    data: object
    note: str = ""


BUILDERS = []


def register(function):
    BUILDERS.append(function)
    return function
