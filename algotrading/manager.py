from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import algotrading
import logging
import pandas as pd
import yaml
import json
import copy
logger = logging.getLogger(__name__)

class ConfigBase:
    """ base class for configuration

    Args:
        abc ([type]): [description]
    """

    def __init__(self):
        self._data = {}

    def read_from_dict(self, cf):  # Type: dict -> None
        """assign python dictionary to config
        """
        assert isinstance(cf, dict)
        self._data = copy.deepcopy(cf)

    def save(self, file_path, format='json'):
        """save config file as json and yaml file

        Args:
            file_path (str): the saved file path
            format (str, optional): the saved format. Defaults to 'json'.
            Only support 2 formats: 'json' or 'yaml'
        """
        assert format == 'json' or format == 'yaml'

    def set_config_data(self, key, value):  # Type: (dict) -> None
        assert isinstance(key, str)
        key_array = key.split('/')
        data = self._data
        for key in key_array[:-1]:
            if key not in data:
                data[key] = {}
            data = data[key]
        if key_array[-1] in data:
            logger.info('overwrite the key {}  by value {}'.format(key_array[-1], value))
        data[key_array[-1]] = value

    # Type: (Union[str, None]) -> Any
    def get_config_data(self, input_key=None, default_value=None):
        if input_key is None:
            return default_value
        data = self._data
        key_array = input_key.split('/')
        for key in key_array:
            if key not in data:
                logger.debug("empty key name \"{}\", will use default value".format(input_key))
                return default_value
            data = data[key]
        return data

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return self._data

    def checker(self):  # Type: () -> Bool
        logger.info("Should never reach me")
        return False

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
    main_cf.read_from_file(args.config)
    if args.result_dir:
        main_cf.set_config_data("result_dir", args.result_dir)
    if args.extra is not None:
        extra_dict = dict(args.extra)
        for key in extra_dict:
            value = extra_dict[key]
            if value == "False":
                value = False
            elif value == "True":
                value = True
            main_cf.set_config_data(key, value)

    ## run tasks
    man = Manager()
    man.prepare(main_cf)
    man.run()
    man.done()

def main():  # type: () -> None
    parser = argparse.ArgumentParser(
        description='run whole flow\n',
    )
    parser.add_argument('--config', default=None, required=True,
                        help="set main flow config file")
    parser.add_argument('--result_dir', required=False,
                        help="result_dir")
    parser.add_argument("--extra", action='append',
                        type=lambda kv: kv.split("="), dest='extra',
                        help="key value pairs for all new created options.",
                        required=False)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    return run_main_flow(args)

if __name__ == '__main__':
    main()
