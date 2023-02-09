import copy
import json
import logging
from collections import defaultdict

import yaml

logger = logging.getLogger(__name__)

class ConfigBase:
    """ base class for configuration

    Args:
        abc ([type]): [description]
    """

    def __init__(self):
        self._data = defaultdict(dict)

    def read_from_dict(self, cf):  # Type: dict -> None
        """assign python dictionary to config
        """
        assert isinstance(cf, dict)
        self._data = defaultdict(dict, copy.deepcopy(cf))

    def read_from_file(self, file_path, format='json'):
        if format == 'json':
            with open(file_path, 'r') as f:
                self._data = defaultdict(dict, json.load(f))
        elif format == 'yaml':
            with open(file_path, 'r') as f:
                self._data = defaultdict(dict, yaml.load(f, Loader=yaml.FullLoader))
        else:
            raise ValueError("Invalid format: {}. Supported formats: json, yaml".format(format))
 

    def save(self, file_path, format='json'):
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(dict(self._data), f)
        elif format == 'yaml':
            with open(file_path, 'w') as f:
                yaml.dump(dict(self._data), f)
        else:
            raise ValueError("Invalid format: {}. Supported formats: json, yaml".format(format))
        

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
        return str(dict(self._data))

    def checker(self):  # Type: () -> Bool
        logger.info("Should never reach me")
        return False
