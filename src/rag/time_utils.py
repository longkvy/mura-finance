from datetime import datetime

def is_same_day(external_time: datetime, reference_time: datetime) -> bool:
    return external_time.date() == reference_time.date()
