import setuptools
import os
import subprocess
from collections import namedtuple


TOP_DIR = os.path.realpath(os.path.dirname(__file__))
DEBUG = bool(os.getenv('DEBUG'))
COVERAGE = bool(os.getenv('COVERAGE'))

################################################################################
# Version
################################################################################

try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(),
        git_version=git_version
    )

################################################################################
# Package description
################################################################################
with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = []
setup_requires = []
tests_require = []
install_requires.extend([
    'pandas',
    'pandas_datareader',
    'matplotlib',
    'seaborn',
    'datetime',
    'numpy',
    'mplfinance'
])

################################################################################
# Test
################################################################################

setup_requires.append('pytest-runner')
tests_require.append('pytest')
tests_require.append('nbval')
tests_require.append('tabulate')

setuptools.setup(
    name="algotrading", # Replace with your own username
    version=VersionInfo.version,
    author="John",
    author_email="qzhang03022@gmail.com",
    description="Algo trading package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxwillzq/algoTrading",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3 License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'algotrading_daily_plot =  algotrading.scripts.draw_single_plot:main'
        ]
    }
)