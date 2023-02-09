"""Create the at_run CLI as the only entry."""
import argparse
import copy
from collections import defaultdict
import json
import logging
import sys

import pandas as pd
import yaml

import algotrading
from algotrading.utils.config_base import ConfigBase

logger = logging.getLogger(__name__)


class Manager:
    
    def __init__(self, input_config={}):
        self.input_config = input_config
        self.output_config = {}
        pass
    
    def prepare(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass

    def done(self, *args, **kwargs):
        pass


def run_main_flow(args):
    # load config from config file
    # command line argument has higher priority than config file
    main_cf = ConfigBase()
    if args.config:
        main_cf.read_from_file(args.config)
    if args.result_dir:
        main_cf.set_config_data("result_dir", args.result_dir,**dict(args.extra or {}))

    ## run tasks
    man = Manager(input_config=main_cf)
    man.prepare()
    man.run()
    man.done()

def parse_args():
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run the main flow of the manager.')
    parser.add_argument('--config', default=None, required=True,
                        help="set main flow config file")
    parser.add_argument('--result_dir', required=False,
                        help="result_dir")
    parser.add_argument("--extra", action='append',
                        type=lambda kv: kv.split("="), dest='extra',
                        help="key value pairs for all new created options.",
                        required=False)


    return parser.parse_args()

def main():  # type: () -> None
    args = parse_args()
    return run_main_flow(args)

if __name__ == '__main__':
    main()
