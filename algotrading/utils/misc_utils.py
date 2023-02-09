import jinja2
import logging
import shlex
from subprocess import call, check_output

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
