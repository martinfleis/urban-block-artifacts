# bananas

This repository contains complete reproducible workflow for a research paper "*Bananas*".

> Fleischmann M, Vybornova A (2022) Bananas. The Journal. DOI: xxx.

Martin Fleischmann<sup>1</sup>, Anastassia Vybornova<sup>2</sup>

1 Geographic Data Science Lab, Department of Geography and Planning, University of Liverpool, m.fleischmann@liverpool.ac.uk.

2 NEtworks, Data and Society (NERDS), Computer Science Department, IT University of Copenhagen, anvy@itu.dk

## Repository structure

The repository contains `code` and `paper` folders, where the former one includes fully reproducible Jupyter notebooks and Python code used within the research and the latter contains LaTeX source file for the manuscript.

## Code and data

The research has been executed within a Docker container `darribas/gds_py:8.0`.

To reproduce the analysis in the MyBinder cloud environment: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/) - this does not work yet but will.

To reproduce the analysis locally, download or clone the repository or its archive, navigate to the folder (`cd code`) and start `docker` using the following command:

```
docker run --rm -ti -p 8888:8888 - -v ${PWD}:/home/jovyan/work darribas/gds_py:8.0
```

That will start Jupyter Lab session on `localhost:8888` and mount the current working directory to `work` folder within the container.

Docker container is based on `jupyter/minimal-notebook`. Please see its [documentation](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-minimal-notebook) for details.
