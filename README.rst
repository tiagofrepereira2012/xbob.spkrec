Speaker Recognition Toolkit
===========================

This is the speaker recognition toolkit, designed to run speaker verification/recognition
experiments . It's originally based on facereclib tool:
https://pypi.python.org/pypi/facereclib

`xbob.spkrec`_ is designed in a way that it should be easily possible to execute experiments combining different mixtures of:

* Speaker Recognition databases and their according protocols
* Voice activity detection
* Feature extraction
* Recognition/Verification tools

In any case, results of these experiments will directly be comparable when the same dataset is employed.

`xbob.spkrec`_ is adapted to run speaker verification/recognition experiments with the SGE grid infrastructure at Idiap.


If you use this package and/or its results, please cite the following
publications:

1. The original paper presented at the NIST SRE 2012 workshop::

     @inproceedings{Khoury_NISTSRE_2012,
       author = {Khoury, Elie and El Shafey, Laurent and Marcel, S{\'{e}}bastien},
       month = {dec},
       title = {The Idiap Speaker Recognition Evaluation System at NIST SRE 2012},
       booktitle = {NIST Speaker Recognition Conference},
       year = {2012},
       location = {Orlando, USA},
       organization = {NIST},
       pdf = {http://publications.idiap.ch/downloads/papers/2012/Khoury_NISTSRE_2012.pdf}
    }


2. Bob as the core framework used to run the experiments::

    @inproceedings{Anjos_ACMMM_2012,
      author = {A. Anjos and L. El Shafey and R. Wallace and M. G\"unther and C. McCool and S. Marcel},
      title = {Bob: a free signal processing and machine learning toolbox for researchers},
      year = {2012},
      month = oct,
      booktitle = {20th ACM Conference on Multimedia Systems (ACMMM), Nara, Japan},
      publisher = {ACM Press},
      url = {http://publications.idiap.ch/downloads/papers/2012/Anjos_Bob_ACMMM12.pdf},
    }


I- Installation
----------------

Just download this package and decompress it locally::

  $ wget http://pypi.python.org/packages/source/x/xbob.spkrec/xbob.spkrec-0.0.1a8.zip
  $ unzip xbob.spkrec-0.0.1a8.zip
  $ cd xbob.spkrec

`xbob.spkrec`_ is based on the `BuildOut`_ python linking system. You only need to use buildout to bootstrap and have a working environment ready for
experiments::

  $ python bootstrap
  $ ./bin/buildout

This also requires that bob (>= 1.2.0) is installed.


II- Running experiments
------------------------

The above two commands will automatically download all desired packages (`gridtk`_, `xbob.sox`_ and `xbob.db.verification.filelist`_ ) from `pypi`_ and generate some scripts in the bin directory, including the following scripts::
  
   $ bin/spkverif_isv.py
   $ bin/spkverif_ivector.py
   $ bin/spkverif_gmm.py
   $ bin/spkverif_jfa.py
   $ bin/para_ubm_spkverif_isv.py
   $ bin/para_ubm_spkverif_ivector.py
   $ bin/para_ubm_spkverif_gmm.py
   $ bin/fusion.py
   $ bin/evaluate.py
   

  
These scripts can be used to employ different 
To use them you have to specify at least four command line parameters (see also the ``--help`` option):

* ``--database``: The configuration file for the database
* ``--preprocessing``: The configuration file for Voice Activity Detection
* ``--feature-extraction``: The configuration file for feature extraction
* ``--tool-chain``: The configuration file for the face verification tool chain

If you are not at Idiap, please precise the TEMP and USER directories:

* ``--temp-directory``: This typically contains the features, the UBM model, the client models, etc.
* ``--user-directory``: This will contain the output scores (in text format)

If you want to run the experiments in the GRID at Idiap or any equivalent SGE, you can simply specify:

* ``--grid``: The configuration file for the grid setup.

If no grid configuration file is specified, the experiment is run sequentially on the local machine.
For several datasets, feature types, recognition algorithms, and grid requirements the `xbob.spkrec`_ provides these configuration files.
They are located in the *config/...* directories.
It is also safe to design one experiment and re-use one configuration file for all options as long as the configuration file includes all desired information:

* The database: ``name, db, protocol; wav_input_dir, wav_input_ext``;
* The preprocessing: ``preprocessor = spkrec.preprocessing.<PREPROCESSOR>``;
* The feature extraction: ``extractor = spkrec.feature_extraction.<EXTRACTOR>``;
* The tool: ``tool = spkrec.tools.<TOOL>``; plus configurations of the tool itself
* Grid parameters: They help to fix which queues are used for each of the steps, how much files per job, etc. 


By default, the ZT score normalization is activated. To deactivate it, please add the ``-z`` to the command line.


III- Experiment design
-----------------------

To be very flexible, the tool chain in the `xbob.spkrec`_ is designed in several stages::

  1. Signal Preprocessing (Voice Activity Detection)
  2  Feature Extraction
  3. Feature Projection
  4. Model Enrollment
  5. Scoring
  6. Fusion
  7. Evaluation

Note that not all tools implement all of the stages.


1. Voice Activity Detection 
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This step aims to filter out the non speech part. Depending on the configuration file, several routines can be enabled or disabled.

* Energy-based VAD
* 4Hz Modulation energy based VAD

2. Feature Extraction
~~~~~~~~~~~~~~~~~~~~~
This step aims to extract features. Depending on the configuration file, several routines can be enabled or disabled.

* LFCC/MFCC feature extraction
* Spectrogram extraction
* Feature normalization


3. Feature Projection
~~~~~~~~~~~~~~~~~~~~~
Some provided tools need to process the features before they can be used for verification.
In the `xbob.spkrec`_, this step is referenced as the **projection** step.
Again, the projection might require training, which is executed using the extracted features from the training set.
Afterward, all features are projected (using the previously trained Projector).


4. Model Enrollment
~~~~~~~~~~~~~~~~~~~
Model enrollment defines the stage, where several (projected or unprojected) features of one identity are used to enroll the model for that identity.
In the easiest case, the features are simply averaged, and the average feature is used as a model.
More complex procedures, which again might require a model enrollment training stage, create models in a different way.


5. Scoring
~~~~~~~~~~
In the final scoring stage, the models are compared to probe features and a similarity score is computed for each pair of model and probe.
Some of the models (the so-called T-Norm-Model) and some of the probe features (so-called Z-Norm-probe-features) are split up, so they can be used to normalize the scores later on.

6. Fusion
~~~~~~~~~
The score fusion of different score outputs uses `logistic regression`_.


7. Evaluation
~~~~~~~~~~~~~
One way to compute the final result is to use the *bin/evaluate.py* e.g., by calling::

  $ bin/evaluate.py -d PATH/TO/USER/DIRECTORY/scores-dev -e PATH/TO/USER/DIRECTORY/scores-eval -c EER -D DET.pdf -x 
  
This will compute the EER, the minCLLR, CLLR, and draw the DET curve.


IV- Command line options
------------------------

Additionally to some of the required command line options discussed above, there are several options to modify the behavior of the `xbob.spkrec`_ experiments.
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


V- Datasets
------------

For the moment, there are 4 databases that are tested in `xbob.spkrec`_. Their protocols are also shipped with the tool. You can use the script ``bob_compute_perf.py`` to compute EER and HTER on DEV and EVAL as follows::


  $ bin/bob_compute_perf.py -d scores-dev -t scores-eval 

By default, this script will also generate the DET curve in a PDF file. 

In this README, we give examples of different toolchains applied on different databases: Voxforge, BANCA, TIMIT, MOBIO, and NIST SRE 2012.

1. Voxforge dataset
~~~~~~~~~~~~~~~~~~~
`Voxforge`_ is a free database used in free speech recognition engines. We randomly selected a small part of the english corpus (< 1GB).  It is used as a toy example for our speaker recognition tool since experiment can be easily run on a local machine, and the results can be obtained in a reasonnable amount of time (< 2h).

Unlike TIMIT and BANCA, this dataset is completely free of charge.

More details about how to download the audio files used in our experiments, and how the data is split into Training, Development and Evaluation set can be found here::
  
  https://pypi.python.org/pypi/xbob.db.voxforge
  
One example of command line is::

  $ ./bin/spkverif_gmm.py -d config/database/voxforge.py -p config/preprocessing/energy.py \
   -f config/features/mfcc_60.py -t config/tools/ubm_gmm/ubm_gmm_256G.py -b ubm_gmm -z \ 
   --user-directory PATH/TO/USER/DIR --temp-directory PATH/TO/TEMP/DIR 
  
In this example, we used the following configuration:

* Energy-based VAD,  
* (19 MFCC features + Energy) + First and second derivatives,
* **UBM-GMM** Modelling (with 256 Gaussians), the scoring is done using the linear approximation of the LLR.

The performance of the system on DEV and EVAL are:

* ``DEV: EER = 2.00%``
* ``EVAL: HTER = 1.65%``
 
Another example is to use **ISV** toolchain instead of UBM-GMM::

  $ ./bin/spkverif_isv.py -d config/database/voxforge.py -p config/preprocessing/energy.py \ 
   -f config/features/mfcc_60.py -t config/tools/isv/isv_256g_u50.py  -z -b isv \ 
   --user-directory PATH/TO/USER/DIR --temp-directory PATH/TO/TEMP/DIR  

* ``DEV: EER = 1.41%``
* ``EVAL: HTER = 1.56%``

One can also try **JFA** toolchain::

  $ ./bin/spkverif_jfa.py -d config/database/voxforge.py -p config/preprocessing/energy.py \ 
   -f config/features/mfcc_60.py -t config/tools/jfa/jfa_256_v5_u10.py  -z -b jfa \ 
   --user-directory PATH/TO/USER/DIR --temp-directory PATH/TO/TEMP/DIR
   
* ``DEV: EER = 5.65%``
* ``EVAL: HTER = 4.82%``   
  
or also **IVector** toolchain where **Whitening, L-Norm, LDA, WCCN** are used like in this example where the score computation is done using **Cosine distance**::

  $ ./bin/spkverif_ivector.py -d config/database/voxforge.py -p config/preprocessing/energy.py \
   -f config/features/mfcc_60.py -t config/tools/ivec/ivec_256g_t100_cosine.py -z -b ivector_cosine \ 
   --user-directory PATH/TO/USER/DIR --temp-directory PATH/TO/TEMP/DIR 
  
* ``DEV: EER = 15.33%``
* ``EVAL: HTER = 15.78%``
  
The scoring computation can also be done using **PLDA**::

  $ ./bin/spkverif_ivector.py -d config/database/voxforge.py -p config/preprocessing/energy.py \ 
   -f config/features/mfcc_60.py -t config/tools/ivec/ivec_256g_t100_plda.py -z -b ivector_plda \
   --user-directory PATH/TO/USER/DIR --temp-directory PATH/TO/TEMP/DIR 

* ``DEV: EER = 15.33%``
* ``EVAL: HTER = 16.93%``


Note that in the previous examples, our goal is not to optimize the parameters on the DEV set but to provide examples of use.
  

2. BANCA dataset
~~~~~~~~~~~~~~~~
`BANCA`_ is a simple bimodal database with relatively clean data. The results are already very good with a simple baseline UBM-GMM system. An example of use can be::

  $ bin/spkverif_gmm.py -d config/database/banca_audio_G.py -p config/preprocessing/energy.py \
    -f config/features/mfcc_60.py -t config/tools/ubm_gmm/ubm_gmm_256G_regular_scoring.py \
    --user-directory PATH/TO/USER/DIR --temp-directory PATH/TO/TEMP/DIR -z

The configuration in this example is similar to the previous one with the only difference of using the regular LLR instead of its linear approximation.

Here is the performance of this system:

* ``DEV: EER = 1.66%``
* ``EVAL: EER = 0.69%``


3. TIMIT dataset
~~~~~~~~~~~~~~~~
`TIMIT`_ is one of the oldest databases (year 1993) used to evaluate speaker recognition systems. In the following example, the processing is done on the development set, and LFCC features are used::

  $ ./bin/spkverif_gmm.py -d config/database/timit.py -p config/preprocessing/energy.py \ 
    -f config/features/lfcc_60.py -t config/tools/ubm_gmm/ubm_gmm_256G.py \ 
    --user-directory PATH/TO/USER/DIR --temp-directory PATH/TO/TEMP/DIR -b lfcc -z --groups dev
  
Here is the performance of the system on the Development set:

* ``DEV: EER = 2.68%``


4. MOBIO dataset
~~~~~~~~~~~~~~~~
This is a more challenging database. The noise and the short duration of the segments make the task of speaker recognition relatively difficult. The following experiment on male group uses the 4Hz modulation energy based VAD, and the ISV (with dimU=50) modelling technique::

  $ ./bin/spkverif_isv.py -d config/database/mobio_male_twothirds_wav.py -p config/preprocessing/mod_4hz.py \ 
   -f config/features/mfcc_60.py -t config/tools/isv/isv_u50.py \ 
   --user-directory PATH/TO/USER/DIR --temp-directory PATH/TO/TEMP/DIR -z
  
Here is the performance of this system:
  
* ``DEV: EER = 10.40%``
* ``EVAL: EER = 10.36%``


5. NIST SRE 2012
~~~~~~~~~~~~~~~~
We first invite you to read the paper describing our system submitted to the NIST SRE 2012 Evaluation. The protocols on the development set are the results of a joint work by the I4U group. To reproduce the results, please check this dedicated package::

  https://pypi.python.org/pypi/xbob.spkrec.nist_sre12


.. _Bob: http://www.idiap.ch/software/bob
.. _local.bob.recipe: https://github.com/idiap/local.bob.recipe
.. _gridtk: https://pypi.python.org/pypi/gridtk
.. _BuildOut: http://www.buildout.org/
.. _NIST: http://www.nist.gov/itl/iad/ig/focs.cfm
.. _xbob.db.verification.filelist: https://pypi.python.org/pypi/xbob.db.verification.filelist
.. _xbob.sox: https://pypi.python.org/pypi/xbob.sox
.. _xbob.spkrec: https://pypi.python.org/pypi/xbob.spkrec
.. _pypi: https://pypi.python.org/pypi
.. _Voxforge: http://www.voxforge.org/
.. _BANCA: http://www.ee.surrey.ac.uk/CVSSP/banca/
.. _TIMIT: http://www.ldc.upenn.edu/Catalog/catalogEntry.jsp?catalogId=LDC93S1
.. _logistic regression: http://en.wikipedia.org/wiki/Logistic_regression
