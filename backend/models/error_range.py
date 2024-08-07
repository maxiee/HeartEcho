from typing import List, Optional
from mongoengine import Document, FloatField


class ErrorRange(Document):
    lower_bound = FloatField(required=True)
    upper_bound = FloatField(required=True)

    meta = {"collection": "error_ranges"}

    # Class variable to store all ErrorRange instances
    _instances: List["ErrorRange"] = []

    @classmethod
    def initialize(cls):
        """Initialize ErrorRanges: load from database, ensure all exist, sort for efficiency."""
        cls.load_ranges()
        cls.ensure_ranges_exist()
        cls._instances.sort(key=lambda x: x.lower_bound)
        print("ErrorRange initialization and sorting complete.")

    @classmethod
    def load_ranges(cls):
        """Load all ErrorRange instances into memory."""
        cls._instances = list(cls.objects.all())
        print(f"Loaded {len(cls._instances)} ErrorRange instances into memory.")

    @classmethod
    def initialize_ranges(cls):
        for i in range(20):
            lower = i * 0.5
            upper = (i + 1) * 0.5
            cls(lower_bound=lower, upper_bound=upper).save()

    @classmethod
    def get_range_for_error(cls, error: float) -> Optional["ErrorRange"]:
        """Get the ErrorRange for a given error value from memory."""
        for range_instance in cls._instances:
            if range_instance.lower_bound <= error < range_instance.upper_bound:
                return range_instance
        return None

    @classmethod
    def ensure_ranges_exist(cls):
        """Ensure all necessary ErrorRange instances exist in the database and memory."""
        needed_ranges = [
            (i * 0.5, (i + 1) * 0.5) for i in range(20)
        ]  # 0 to 10 in 0.5 increments
        existing_ranges = cls._instances

        for lower, upper in needed_ranges:
            if not any(
                r.lower_bound == lower and r.upper_bound == upper
                for r in existing_ranges
            ):
                new_range = cls(lower_bound=lower, upper_bound=upper)
                new_range.save()
                cls._instances.append(new_range)
                print(f"Created new ErrorRange: {lower} - {upper}")

        print("ErrorRange initialization complete.")
