from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import yaml
logger = logging.getLogger(__name__)

def get_data_dict(file_name):
    TOP_DIR = os.path.dirname(__file__)
    full_file_path = os.path.join(TOP_DIR, file_name)
    with open(full_file_path, 'r') as f:
        result_dict = yaml.load(f)
    return result_dict