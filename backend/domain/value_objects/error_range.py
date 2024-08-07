from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorRange:
    lower_bound: float
    upper_bound: float

    def __contains__(self, value: float) -> bool:
        return self.lower_bound <= value < self.upper_bound
