import unittest
from algotrading.utils.misc_utils import render_template_with_dict
from algotrading.utils.misc_utils import command_executor
from algotrading.utils.misc_utils import generate_report_from_markdown
import tempfile
from parameterized import parameterized
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

class TestGenerateReportFromMarkdown(unittest.TestCase):
    @parameterized.expand([("pdf", "test.pdf"), ("html", "test.html")])
    def test_conversion(self, report_format, output_file):
        result_dir = tempfile.mkdtemp()
        md_file = os.path.join(result_dir, "test.md")
        output_file = os.path.join(result_dir, output_file)

        # Create a test Markdown file
        with open(md_file, "w") as f:
            f.write("# Test Report\n")

        generate_report_from_markdown(md_file, result_dir, output_file, report_format)
        self.assertTrue(os.path.exists(output_file))

        # Clean up
        os.remove(md_file)
        os.remove(output_file)
        os.rmdir(result_dir)

        
if __name__ == '__main__':
    unittest.main()
