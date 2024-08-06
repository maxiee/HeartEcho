from mongoengine import Document, FloatField


class ErrorRange(Document):
    lower_bound = FloatField(required=True)
    upper_bound = FloatField(required=True)

    meta = {"collection": "error_ranges"}

    @classmethod
    def initialize_ranges(cls):
        for i in range(20):
            lower = i * 0.5
            upper = (i + 1) * 0.5
            cls(lower_bound=lower, upper_bound=upper).save()

    @classmethod
    def get_range_for_error(cls, error):
        return cls.objects(lower_bound__lte=error, upper_bound__gt=error).first()
