The R files are forked from https://rdrr.io/rforge/disttree/, as the package cannot be downloaded with more recent versions of R.

We only made minor changes to those files to simplify the usage of disttree with rpy2.

To install the dependencies requires for distree to work, run `uv run disttree.py`.
For this R must be installed on the system such that `uv sync` is able to install rpy2.
In some cases, the user is prompted to specify a default location for the packages.