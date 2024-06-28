# A shape-based heuristic for the detection of urban block artifacts

This repository contains complete reproducible workflow for the research paper "A shape-based heuristic for the detection of urban block artifacts", published open-access in the [Journal of Spatial Information Science (JOSIS)](https://josis.org/index.php/josis/article/view/319). 

> Fleischmann M, Vybornova A (2024) A shape-based heuristic for the detection of urban block artifacts. doi: [10.5311/JOSIS.2024.28.31](https://josis.org/index.php/josis/article/view/319)

Martin Fleischmann<sup>1</sup>, Anastassia Vybornova<sup>2</sup>

1 Department of Social Geography and Regional Development, Charles University, Czechia, martin.fleischmann@natur.cuni.cz

2 NEtworks, Data and Society (NERDS), Computer Science Department, IT University of Copenhagen, anvy@itu.dk

## Repository structure

The folder `code` contains fully reproducible Jupyter notebooks (to be run in sequential order : `01`, then `02` etc.) and Python code used within the research.

The folder `data` contains:
* the file `sample.parquet`, generated within the notebook `01_download`, with metada on all 131 functional urban areas (FUAs) used in the analysis
* one subfolder `/data/<FUA_ID>/` for each FUA, with corresponding street network data and polygon shapes

The folder `plots` contains all figures produced in the analysis and used in the paper.

The folder `results` contains results on: shape metrics correlations; face artifact index thresholds for all 131 FUAs; and computational efficiency.

## Reproducibility

The research has been executed within a Docker container `darribas/gds_py:9.0`.

To reproduce the analysis locally, download or clone the repository or its archive, navigate to the folder (`cd urban-block-artifacts`) and start `docker` using the following command:

```
docker run --rm -ti -p 8888:8888 -e USE_PYGEOS=1 -v ${PWD}:/home/jovyan/work darribas/gds_py:9.0
```

That will start Jupyter Lab session on `localhost:8888` and mount the current working directory to `work` folder within the container.

Docker container is based on `jupyter/minimal-notebook`. Please see its [documentation](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-minimal-notebook) for details.
