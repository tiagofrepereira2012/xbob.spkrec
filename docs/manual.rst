.. vim: set fileencoding=utf-8 :
.. Elie Khoury <Elie.Khoury@idiap.ch>
.. This work is based on SpkRecTool (developed by Manuel Gunther and Laurent El Shafey)
.. Fri 16 11 2012

============
 SpkRecTool
============

The SpkRecTool is originally based on SpkRecTool that is developed at Idiap by Manuel Gunther and Laurent El Shafey.

The SpkRecTool is designed to run speaker verification/recognition experiments with the SGE grid infrastructure at Idiap.
It is designed in a way that it should be easily possible to execute experiments combining different mixtures of:

* Speaker Recognition databases and their according protocols
* Speech preprocessing
* Feature extraction
* Recognition/Verification tools

In any case, results of these experiments will directly be comparable when the same database is employed.

Installation instructions
-------------------------

To install the SpkRecTool, please check the latest version of it via:

.. code-block:: sh

  $ git clone /idiap/group/torch5spro/sandboxes/xxxxxx.git
  $ cd xxxxxx

For the spkRecTool to work, it requires `Bob`_ to be installed.
At Idiap, you can either have your local Bob installation or use the global one located at:

::

  > /idiap/group/torch5spro/nightlies/last/install/<VERSION>-release

where <VERSION> is your operating system version.

The SpkRecTool project is based on the `BuildOut`_ python linking system.
If you want to use another version of Bob than the nightlies, you have to modify the delivered *buildout.cfg* by specifying the path to your Bob installation.

Afterwards, execute the buildout script by typing:

.. code-block:: sh

  $ /remote/filer.gx/group.torch5spro/nightlies/externals/v3/ubuntu-10.04-x86_64/bin/python bootstrap.py
  $ bin/buildout



Running experiments
-------------------

These two commands will automatically download all desired packages (`local.bob.recipe`_ and `gridtk`_) from GitHub and generate some scripts in the bin directory, including the script *bin/faceverify_zt.py*.
This script can be used to employ face verification experiments.
To use it you have to specify at least three command line parameters (see also the ``--help`` option):

* ``--database``: The configuration file for the database
* ``--preprocessing``: The configuration file for speaker preprocessing
* ``--tool-chain``: The configuration file for the face verification tool chain

If you want to run the experiments in the Idiap GRID, you simply can specify:

* ``--grid``: The configuration file for the grid setup.

If no grid configuration file is specified, the experiment is run sequentially on the local machine.
For several databases, feature types, recognition algorithms, and grid requirements the SpkRecTool provides these configuration files.
They are located in the *config/...* directories.
It is also save to design one experiment and re-use one configuration file for all options as long as the configuration file includes all desired information:

* The database: ``name, db, protocol; img_input_dir, img_input_ext``; optional: ``pos_input_dir, pos_input_ext, first_annot; all_files_option, world_extractor_options, world_projector_options, world_enroler_options, features_by_clients_options``
* The preprocessing: ``preprocessor = spkrectool.preprocessing.<PREPROCESSOR>``; optional: ``color_channel``; plus configurations of the preprocessor itself
* The tool: ``tool = spkrectool.tools.<TOOL>``; plus configurations of the tool itself
* Grid parameters: ``training_queue; number_of_images_per_job, preprocessing_queue; number_of_features_per_job, extraction_queue, number_of_projections_per_job, projection_queue; number_of_models_per_enrol_job, enrol_queue; number_of_models_per_score_job, score_queue``

None of the parameters in the configurations are fixed, so please feel free to test different settings.
Please note that not all combinations of features and tools make sense since the tools expect different kinds of features (e.g. UBM/GMM needs 2D features, whereas PCA expects 1D features).


By default, the verification result will be written to directory */idiap/user/$USER/<DATABASE>/<EXPERIMENT>/<SUBDIR>/<PROTOCOL>*, where

* DATABASE: the name of the database. It is read from the database configuration file
* EXPERIMENT: a user-specified experiment name (``--sub-dir`` option), by default it is ``default``
* SUBDIR: another user-specified name (``--score-sub-dir`` option), e.g. to specify different options of the experiment
* PROTOCOL: the protocol which is read from the database configuration file

After running a  ZT-Norm based experiment, the output directory contains two sub-directories *nonorm*, *ztnorm*, each of which contain the files *scores-dev* and *scores-eval*.
One way to compute the final result is to use the *bob_compute_perf.py* script from your Bob installation, e.g., by calling:

.. code-block:: sh

  $ cd /idiap/user/$USER/<DATABASE>/<EXPERIMENT>/<SUBDIR>/<PROTOCOL>
  $ bob_compute_perf.py -d nonorm/scores-dev -t nonorm/scores-eval


Temporary files will by default be put to */scratch/$USER/<DATABASE>/<EXPERIMENT>* or */idiap/temp/$USER/<DATABASE>/<EXPERIMENT>* when run locally or in the grid, respectively.


Experiment design
-----------------

To be very flexible, the tool chain in the SpkRecTool is designed in several stages:

1. Feature Preprocessing and Extraction
3. Feature Projection
4. Model Enrollment
5. Scoring

Note that not all tools implement all of the stages.


Feature Preprocessing and Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This step aims to extract features. Depending on the configuration file, several routines can be enabled or disabled.

* LFCC/MFCC feature extraction
* Filtering speech part using existing VAD segmentation
* Energy-based VAD
* Use existing 4Hz Modulation energy segmentation (**TODO:** we are planning to implement soon this method in the tool)
* Feature normalization


Feature Projection
~~~~~~~~~~~~~~~~~~
Some provided tools need to process the features before they can be used for verification.
In the SpkRecTool, this step is referenced as the **projection** step.
Again, the projection might require training, which is executed using the extracted features from the training set.
Afterward, all features are projected (using the the previously trained Projector).


Model Enrollment
~~~~~~~~~~~~~~~~
Model enrollment defines the stage, where several (projected or unprojected) features of one identity are used to enroll the model for that identity.
In the easiest case, the features are simply averaged, and the average feature is used as a model.
More complex procedures, which again might require a model enrollment training stage, create models in a different way.


Scoring
~~~~~~~
In the final scoring stage, the models are compared to probe features and a similarity score is computed for each pair of model and probe.
Some of the models (the so-called T-Norm-Model) and some of the probe features (so-called Z-Norm-probe-features) are split up, so they can be used to normalize the scores later on.



Command line options
--------------------
Additionally to the required command line options discussed above, there are several options to modify the behavior of the SpkRecTool experiments.
One set of command line options change the directory structure of the output:

* ``--temp-directory``: Base directory where to write temporary files into (the default is */idiap/temp/$USER/<DATABASE>* when using the grid or */scratch/$USER/<DATABASE>* when executing jobs locally)
* ``--user-directory``: Base directory where to write the results, default is */idiap/user/$USER/<DATABASE>*
* ``--sub-directory``: sub-directory into *<TEMP_DIR>* and *<USER_DIR>* where the files generated by the experiment will be put
* ``--score-sub-directory``: name of the sub-directory in *<USER_DIR>/<PROTOCOL>* where the scores are put into

If you want to re-use parts previous experiments, you can specify the directories (which are relative to the *<TEMP_DIR>*, but you can also specify absolute paths):

* ``--preprocessed-image-directory``
* ``--features-directory``
* ``--projected-directory``
* ``--models-directories`` (one for each the Models and the T-Norm-Models)

or even trained Extractor, Projector, or Enroler (i.e., the results of the extraction, projection, or enrollment training):

* ``--extractor-file``
* ``--projector-file``
* ``--enroler-file``

For that purpose, it is also useful to skip parts of the tool chain.
To do that you can use:

* ``--skip-preprocessing``
* ``--skip-feature-extraction-training``
* ``--skip-feature-extraction``
* ``--skip-projection-training``
* ``--skip-projection``
* ``--skip-enroler-training``
* ``--skip-model-enrolment``
* ``--skip-score-computation``
* ``--skip-concatenation``

although by default files that already exist are not re-created.
To enforce the re-creation of the files, you can use the ``--force`` option, which of course can be combined with the ``--skip...``-options (in which case the skip is preferred).

There are some more command line options that can be specified:

* ``--no-zt-norm``: Disables the computation of the ZT-Norm scores.
* ``--groups``: Enabled to limit the computation to the development ('dev') or test ('eval') group. By default, both groups are evaluated.
* ``--preload-probes``: Speeds up the score computation by loading all probe features (by default, they are loaded each time they are needed). Use this option only, when you are sure that all probe features fit into memory.
* ``--dry-run``: When the grid is enabled, only print the tasks that would have been sent to the grid without actually send them. **WARNING** This command line option is ignored when no ``--grid`` option was specified!


Databases
---------

For the moment, there are 3 databases that are tested in SpkRecTool. Their protocols are also shipped with the tool. You can use the script ``bob_compute_perf.py`` to compute EER and HTER on DEV and EVAL as follows:

.. code-block:: sh

  $ bin/bob_compute_perf.py -d scores-dev -t scores-eval -x


BANCA database
~~~~~~~~~~~~~~
This is a clean database. The results are already very good with a simple baseline system. In the following example, we apply the UBM-GMM system.

.. code-block:: sh

  $ bin/spkverif_zt.py -d config/database/banca_audio_G.py -t config/tools/ubm_gmm_regular_scoring.py  -p config/preprocessing/mfcc_60.py -z
  

* ``DEV: EER = 1.282%``
* ``EVAL: EER = 0.908%``


MOBIO database
~~~~~~~~~~~~~~
This is a more challenging database. The noise and the short duration of the segments make the task of speaker recognition very difficult. The following experiment on male group uses the ISV modelling technique.

.. code-block:: sh

  $ ./bin/spkverif_zt.py -d config/database/mobio_male_twothirds_wav.py -t config/tools/isv.py -p config/preprocessing/mfcc_60.py 
  
  
* ``DEV: EER = 19.881%``
* ``EVAL: EER = 15.508%``

NIST-SRE2012 database
~~~~~~~~~~~~~~~~~~~~~
We first invite you to read the paper describing our system submitted to the NIST-SRE2012 Evaluation, and the paper describing I4U system (joint submission with I2R, RUN, UEF, VLD, LIA, UTD, UWS). The protocols on the development set are the results of a joint work by the I4U group (check if we can make them publicly available).



.. _Bob: http://idiap.github.com/bob/
.. _local.bob.recipe: https://github.com/idiap/local.bob.recipe
.. _gridtk: https://github.com/idiap/gridtk
.. _BuildOut: http://www.buildout.org/
.. _NIST: http://www.nist.gov/itl/iad/ig/focs.cfm

