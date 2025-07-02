## Build

Bit weird at the moment but I wanted to use `uv` becasue it was easy to setup with `scikit-build`/`pybind11` if we need C/C++/Fortran extentions....and then I relized MDSPlus can only use `conda`....

```
conda create -p $PWD/.venv -c conda-forge MDSplus
conda activate $PWD/.venv
uv pip install -e .
```