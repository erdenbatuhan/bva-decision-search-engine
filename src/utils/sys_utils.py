import os

from src.utils.logging_utils import log


def create_dir(dir_name):
    """
    Creates the directory if it does not exist

    :param dir_name: The name of the directory
    """

    if not os.path.exists(dir_name):
        log("Creating a directory named %s" % dir_name)
        os.mkdir(dir_name)

