from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from subprocess import call, check_output
from . import plotting
import shlex
import logging
logger = logging.getLogger(__name__)

def command_executor(cmd, stdout=None):
    '''
    Executes the command.
    Example:
      - command_executor('ls -lrt')
      - command_executor(['ls', '-lrt'])
    '''
    if type(cmd) == type([]):  # if its a list, convert to string
        cmd = ' '.join(cmd)
    logger.info("cmd = {}".format(cmd))
    if (call(shlex.split(cmd), stdout=stdout, stderr=stdout) != 0):
        raise Exception("Error running command: " + cmd)
    return True