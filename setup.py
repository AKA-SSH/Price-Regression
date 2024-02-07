# libraries required
from typing import List
from setuptools import setup, find_packages

# helper function: read_file -> reads content of file at given file_path
def read_file(file_path: str)->List[str]:
    with open(file_path, 'r') as file_object:
        contents= [line.rstrip('\n') for line in file_object]
    if '-e .' in contents:
        contents.remove('-e .')
    return contents

with open('README.md', 'r', encoding='utf-8') as content:
    long_description_content = content.read()

# setup configuration
setup (name='regression',
      version='0.0.1',
      author='AKA-SSH',
      author_email='aka.ssh.datascience@gmail.com',
      description='This project deals with the classification problem on an asteriod dataset.',
      long_description=long_description_content,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=read_file(r'Environment\requirements.txt'),
      python_requires='>=3.8')