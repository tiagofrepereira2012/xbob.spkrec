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
    name='speaker_recognition',
    version='0.1',
    description='Speaker recognition and speaker recognition toolchain',

    #url='http://pypi.python.org/pypi/TowelStuff/',
    #license='LICENSE.txt',

    author='Elie Khoury',
    author_email='Elie.Khoury@idiap.ch',

    packages=find_packages(),

    entry_points={
      'console_scripts': [
        'spkverif_isv.py = speaker_recognition.script.spkverif_isv:main',
        'spkverif_ivector.py = speaker_recognition.script.spkverif_ivector:main',
        'para_ubm_spkverif_isv.py = speaker_recognition.script.para_ubm_spkverif_isv:main',
        'para_ubm_spkverif_ivector.py = speaker_recognition.script.para_ubm_spkverif_ivector:main',
        'manual_vad_conversion.py = speaker_recognition.script.manual_vad_conversion:main',
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
