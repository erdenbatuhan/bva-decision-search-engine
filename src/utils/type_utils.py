"""
File:   type_utils.py
Author: Batuhan Erden
"""

import datetime
from dateutil.parser import parse

YEAR_BOUNDARIES = 1900, int(datetime.date.today().strftime("%Y"))


def is_number(text):
    """
    Return whether or not the string can be interpreted as a number

    :param text: Text to be checked if it is a number
    :return: Whether or not the string can be interpreted as a number
    """
    try:
        int(text)
        return True
    except ValueError:
        return False


def is_date(text, fuzzy=True):
    """
    Return whether or not the string can be interpreted as a date

    Referenced from: https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format

    :param text: Text to be checked if it is a date
    :param fuzzy: When set to True, unknown tokens are ignored
    :return: Whether or not the string can be interpreted as a date
    """
    try:
        parse(text, fuzzy=fuzzy)

        # Check if it is a number and between the year boundaries defined
        if is_number(text):
            return YEAR_BOUNDARIES[0] <= int(text) <= YEAR_BOUNDARIES[1]

        return True
    except Exception:
        return False
