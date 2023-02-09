import unittest
import tempfile
import json
import yaml
from collections import defaultdict
from config_base import ConfigBase

class TestConfigBase(unittest.TestCase):

    def test_read_from_dict(self):
        cfg = ConfigBase()
        cf = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
        cfg.read_from_dict(cf)
        self.assertDictEqual(dict(cfg._data), cf)

    def test_read_from_file_json(self):
        cfg = ConfigBase()
        cf = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            json.dump(cf, f)
            f.seek(0)
            cfg.read_from_file(f.name, 'json')
        self.assertDictEqual(dict(cfg._data), cf)

    def test_read_from_file_yaml(self):
        cfg = ConfigBase()
        cf = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            yaml.dump(cf, f)
            f.seek(0)
            cfg.read_from_file(f.name, 'yaml')
        self.assertDictEqual(dict(cfg._data), cf)

    def test_save_json(self):
        cfg = ConfigBase()
        cf = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
        cfg.read_from_dict(cf)
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            cfg.save(f.name, 'json')
            f.seek(0)
            self.assertDictEqual(json.load(f), cf)

    def test_save_yaml(self):
        cfg = ConfigBase()
        cf = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
        cfg.read_from_dict(cf)
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            cfg.save(f.name, 'yaml')
            f.seek(0)
            self.assertDictEqual(yaml.load(f, Loader=yaml.FullLoader), cf)

    def test_set_config_data(self):
        cfg = ConfigBase()
        cfg.set_config_data("a/b/c", 10)
        self.assertEqual(cfg.get_config_data("a/b/c"), 10)

if __name__ == '__main__':
    unittest.main()