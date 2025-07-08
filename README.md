
```bash
mamba create -p $SCRATCH/bes-velocimetry -c ga-fdp -c conda-forge toksearch_d3d
mamba activate $SCRATCH/bes-velocimetry
pip install -e .
bes-velocimetry
```
