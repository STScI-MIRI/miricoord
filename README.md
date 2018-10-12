# miricoord
Code for MIRI Imager (LRS) and MRS distortion and coordinate transforms.

Includes scripts for generating pipeline CRDS distortion reference files from FITS, computing boresight offset tables, dither tables, etc.

## Contents:
- IDL implementations of some code for backward compatibility live in the idl/ directory
- Python code lives in the miricoord/ directory
  - /general: General utility functions (timers, logging tools, etc)
  - /imager: Tools for the MIRI imager (including coronagraphs and LRS)
  - /mrs: Tools for the MIRI medium resolution IFU spectrometer
  - /reffiles: FITS and CRDS format reference input files

## Installation:

Python code is developed for a python 3.5 environment.  If installing via a git clone command make sure it is in a location that is on your python path.  E.g.,

 export PYTHONPATH="/YourPathHere:$PYTHONPATH"

IDL code is developed for IDL 8.0 and requires that the global system variable 
$MIRICOORD_DIR
be defined and pointing to the root directory of this product, and that it be included on the search path.  This is necessary so that the code can find the relevant reference files.

E.g.,
 setenv MIRICOORD_DIR /YourPathHere/miricoord
 setenv IDL_PATH ${IDL_PATH}:+$MIRICOORD_DIR

Some routines are configured to write files to a specific set of subdirectories on disk (when, e.g., generating new reference files).  The base directory for these should be set as:

setenv MIRICOORD_DATA_DIR /YourDataPathHere/

If this is not set, these files will default to writing out in your current working directory.

## Dependencies:

 * Some python tools depend on the pysiaf package (https://github.com/spacetelescope/pysiaf) for interaction with the MIRI SIAF
 * Some python tools depend on the JWST calibration pipeline (https://github.com/spacetelescope/jwst)
