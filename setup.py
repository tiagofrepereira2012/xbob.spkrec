from setuptools import setup, find_packages

setup(
    name='spkrectool',
    version='0.1',
    description='Speaker recognition and speaker recognition toolchain',

    #url='http://pypi.python.org/pypi/TowelStuff/',
    #license='LICENSE.txt',

    author='Elie Khoury',
    author_email='Elie.Khoury@idiap.ch',

    packages=find_packages(),

    entry_points={
      'console_scripts': [
        'spkverif_zt.py = spkrectool.script.spkverif_zt:main',
        ],
      },

    #long_description=open('doc/install.rst').read(),

    install_requires=[
        "setuptools", # for whatever
        "gridtk",   # SGE job submission at Idiap
        "bob >= 1.1.4",      # base signal proc./machine learning library
        # databases
        "xbob.db.faceverif_fl",
        "xbob.db.mobio",
        "pysox",
    ],
)
