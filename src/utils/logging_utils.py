"""
 File:   logging_utils.py
 Author: Batuhan Erden
"""

import datetime


def log(message):
    print("Log (%s): %s" % (str(datetime.datetime.now()), message))

