import jinja2
import logging
import shlex
import pypandoc
import os
import json
import yaml
from subprocess import call, check_output
from typing import Dict

logger = logging.getLogger(__name__)

def render_template_with_dict(
    template_string: str, render_dict: dict, result_file_path=None
) -> str:
    """
    return the render string with template and python dictionary
    """
    template_env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.DebugUndefined,
    )
    template = template_env.from_string(template_string)
    result = template.render(render_dict)
    if result_file_path:
        with open(result_file_path, "w") as f:
            f.write(result)
    return result

def command_executor(cmd, stdout=None):
    """
    Executes the command.
    Example:
      - command_executor('ls -lrt')
      - command_executor(['ls', '-lrt'])
    """
    if type(cmd) == type([]):  # if its a list, convert to string
        cmd = " ".join(cmd)
    logger.info("cmd = {}".format(cmd))
    if call(shlex.split(cmd), stdout=stdout, stderr=stdout) != 0:
        raise Exception("Error running command: " + cmd)
    return True

def generate_report_from_markdown(md_file_path, result_dir, output_file_path, report_format):
    current_dir = os.getcwd()
    os.chdir(result_dir)
    extra_args = []
    if report_format == "pdf":
        extra_args = [
            "-V",
            "geometry:margin=1.5cm",
            "--pdf-engine=/Library/TeX/texbin/pdflatex",
        ]
    try:
        output = pypandoc.convert_file(
            md_file_path,
            report_format,
            outputfile=output_file_path,
            extra_args=extra_args,
        )
    except Exception as e:
        raise ValueError("Conversion failed:") from e
    finally:
      os.chdir(current_dir)

def read_dict_from_file(file_path) -> Dict:
    """read config from json or yaml file

    Args:
        file_path (str): the file path of config file
    """
    assert isinstance(file_path, str)
    assert os.path.isfile(file_path), "{} file does not exist".format(file_path)
    with open(file_path, "r") as f:
        if file_path.endswith("json"):
            result_dict = json.load(f)
        elif file_path.endswith("yaml") or file_path.endswith("yml"):
            result_dict = yaml.safe_load(f)
        else:
            raise RuntimeError(
                "not support type file. only supprot yaml, yml or json. file path is "
                + file_path
            )
        return result_dict
