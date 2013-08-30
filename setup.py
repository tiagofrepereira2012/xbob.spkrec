#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Elie Khoury <Elie.Khoury@idiap.ch>
# @date: Tue Oct 30 09:53:56 CET 2012
#
# Copyright (C) 2012-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from setuptools import setup, find_packages

setup(
    name='xbob.spkrec',
    version='0.0.1a0',
    description='Speaker recognition toolchain',
    url='https://pypi.python.org/pypi/xbob.spkrec',
    license='GPLv3',
    author='Elie Khoury',
    author_email='Elie.Khoury@idiap.ch',
    long_description=open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    entry_points={
      'console_scripts': [
        'spkverif_isv.py = spkrec.script.spkverif_isv:main',
        'spkverif_gmm.py = spkrec.script.spkverif_isv:main',
        'spkverif_ivector.py = spkrec.script.spkverif_ivector:main',
        'para_ubm_spkverif_isv.py = spkrec.script.para_ubm_spkverif_isv:main',
        'para_ubm_spkverif_gmm.py = spkrec.script.para_ubm_spkverif_isv:main',
        'para_ubm_spkverif_ivector.py = spkrec.script.para_ubm_spkverif_ivector:main',
#        'manual_vad_conversion.py = spkrec.script.manual_vad_conversion:main',
        ],
      },

    install_requires=[
        "setuptools", # for whatever
        "gridtk",   # SGE job submission at Idiap
        "bob >= 1.2.0",      # base signal proc./machine learning library
        "facereclib",
        "pysox",
        # databases
        "xbob.db.verification.filelist",
        "xbob.db.voxforge",      
    ],

    classifiers = [
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
)
