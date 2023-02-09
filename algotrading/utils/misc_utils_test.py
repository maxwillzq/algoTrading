import unittest
from algotrading.utils.misc_utils import render_template_with_dict
from algotrading.utils.misc_utils import command_executor
import tempfile
import os
import subprocess
import shlex

class TestCommandExecutor(unittest.TestCase):
    def test_command_executor(self):
        # Test successful command execution
        cmd = 'ls -lrt'
        self.assertTrue(command_executor(cmd))
        
        cmd = ['ls', '-lrt']
        self.assertTrue(command_executor(cmd))
        
    def test_command_executor_fail(self):
        # Test failing command execution
        cmd = 'non_existent_command'
        with self.assertRaises(Exception):
            command_executor(cmd)      


class TestRenderTemplateWithDict(unittest.TestCase):
    def test_render_template_with_dict(self):
        template_string = 'Hello {{ name }}!'
        render_dict = {'name': 'John'}
        result = render_template_with_dict(template_string, render_dict)
        self.assertEqual(result, 'Hello John!')
        
    def test_render_template_with_dict_result_file_path(self):
        template_string = 'Hello {{ name }}!'
        render_dict = {'name': 'John'}
        # Test the result_file_path argument
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            result_file_path = f.name
            result = render_template_with_dict(template_string, render_dict, result_file_path)
            f.seek(0)
            contents = f.read()
            self.assertEqual(contents, 'Hello John!')
            os.remove(result_file_path)

        
if __name__ == '__main__':
    unittest.main()
