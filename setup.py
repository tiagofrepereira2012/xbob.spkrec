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
        'spkverif_isv.py = spkrectool.script.spkverif_isv:main',
        'spkverif_ivector.py = spkrectool.script.spkverif_ivector:main',
        'para_ubm_spkverif_isv.py = spkrectool.script.para_ubm_spkverif_isv:main',
        'para_ubm_spkverif_ivector.py = spkrectool.script.para_ubm_spkverif_ivector:main',
        'manual_vad_conversion.py = spkrectool.script.manual_vad_conversion:main',
        ],
      },

    #long_description=open('doc/install.rst').read(),

    install_requires=[
        "setuptools", # for whatever
        "gridtk",   # SGE job submission at Idiap
        "bob >= 1.2.0",      # base signal proc./machine learning library
        # databases
        "xbob.db.verification.filelist",
        "pysox",
    ],
)
