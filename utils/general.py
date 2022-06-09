import datetime


def get_time_stamp():
    time = datetime.datetime.now()
    return time.strftime(r"%Y%m%d_%H%M%S")