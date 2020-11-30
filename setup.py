import codecs
import os
import re
from setuptools import find_packages, setup


def find_version(*file_paths):
    """Read the version number from a source file.
    Why read it, and not import?
    see https://groups.google.com/d/topic/pypa-dev/0PkjVpcxTzQ/discussion
    """
    # Open in Latin-1 so that we avoid encoding errors.
    # Use codecs.open for Python 2 compatibility
    with codecs.open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), *file_paths),
        "r",
        "latin1",
    ) as f:
        version_file = f.read()

    # The version line must have the form
    # __version__ = 'ver'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="onequietnight",
    version=find_version("onequietnight", "__init__.py"),
    url="https://github.com/One-Quiet-Night/COVID-19-forecast",
    packages=find_packages(),
    install_requires=[
        "pandas==1.1.4",
        "numpy==1.18.4",
        "numpyro==0.4.1",
        "jax==0.2.3",
        "requests==2.23.0",
        "beautifulsoup4==4.9.3",
        "sklearn-pandas==2.0.3",
    ],
)
