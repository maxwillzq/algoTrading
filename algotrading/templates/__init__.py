"""
Add all those helper functions we need
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import jinja2
from jinja2 import DebugUndefined, Environment, FileSystemLoader
logger = logging.getLogger(__name__)

def render_template_with_dict(template_string: str, render_dict: dict, result_file_path=None) -> str:
    """
    return the render string with template and python dictionary
    """
    template_env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=DebugUndefined,
    )
    template = template_env.from_string(template_string)
    result = template.render(render_dict)
    if result_file_path:
        with open(result_file_path, 'w') as f:
            f.write(result)
    return result

def get_template_file_from_name(file_name):
    TOP_DIR = os.path.dirname(__file__)
    full_file_path = os.path.join(TOP_DIR, file_name)
    return full_file_path
