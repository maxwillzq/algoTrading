"""Utility to load code from an external user-supplied directory."""
from __future__ import absolute_import, division, print_function

import importlib
import logging
import os
import sys


def import_usr_dir(usr_dir):
    """Import module at usr_dir, if provided."""
    if not usr_dir:
        return

    dir_path = os.path.abspath(os.path.expanduser(usr_dir).rstrip("/"))
    containing_dir, module_name = os.path.split(dir_path)
    logging.info("Importing user module %s from path %s", module_name,
                 containing_dir)
    sys.path.insert(0, containing_dir)
    importlib.import_module(module_name)
    sys.path.pop(0)
