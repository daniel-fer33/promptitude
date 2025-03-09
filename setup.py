import os
import re
import codecs
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="promptitude",
    version=find_version("promptitude", "__init__.py"),
    url="https://github.com/daniel-33f/promptitude",
    author="Daniel Fernandez",
    author_email="daniel-fernandez@cookpad.com",
    description="A flexible prompting language framework for large language models based on Guidance",
    long_description="Promptitude is a powerful and flexible prompting language framework designed to simplify and "
                     "enhance interactions with large language models (LLMs). It allows you to create dynamic, "
                     "template-driven prompts that combine text, logical controls, and variable substitutions, "
                     "providing fine-grained control over LLM outputs. Promptitude offers an intuitive approach to "
                     "prompt engineering, particularly suited for rapid prototyping and iterative development of LLM "
                     "applications. With Promptitude, you can achieve more efficient and effective prompting compared "
                     "to traditional methods, making it an ideal tool for developers working with LLMs.",
    packages=find_packages(exclude=["user_studies", "notebooks", "client"]),
    package_data={"promptitude": ["resources/*"]},
    install_requires=[
        "diskcache",
        "gptcache",
        "openai>=0.1.0",
        "anthropic>=0.47.0",
        "pyparsing>=3.0.0",
        "pygtrie",
        "platformdirs",
        "tiktoken>=0.7",
        "nest_asyncio",
        "msal",
        "requests",
        "numpy",
        "aiohttp",
    ],
    extras_require={
        'docs': [
            'ipython',
            'numpydoc',
            'sphinx_rtd_theme',
            'sphinx',
            'nbsphinx'
        ],
        'test': [
            'pytest',
            'transformers',
            'torch',
            'pytest-cov',
            'python-dotenv'
        ]
    }
)
