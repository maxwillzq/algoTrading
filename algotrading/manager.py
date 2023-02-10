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
    
    def __init__(self):
        self.input_config = {}
        self.output_config = {}
        pass
    
    def prepare(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass

    def done(self, *args, **kwargs):
        pass
