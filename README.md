# symODE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13523852.svg)](https://doi.org/10.5281/zenodo.13523852)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy (inofficial badge https://github.com/python/mypy/issues/12796#issuecomment-2311686298)](https://img.shields.io/badge/type%20checked-mypy-039dfc)](https://mypy-lang.org/)

Python software for the algebraic (SymPy) and numerical (SciPy) treatment of ordinary differential equations (ODEs)

## Backlog

- CD with GitHub actions
  - deploy to conda and pypi package
- allow ODEs to be dependent on time explicitly
- refactor `get_limit_cycle`
- refactor `get_event_based_evolution`
- refactor `get_isochrones_isostables`
- refactor `get_isostables_around_focus`
- refactor `get_isostable_map_at_fixed_point`
  - review whether the new DynamicalSystem is really necessary
  - check for degenerate case of eigenvalues with multipl. > 1
- add feature `find_symmetries`
- add feature `get_inflection_set` (see Ref. ?)
- add feature to check given ansatz for adjoint equation
- add feature to compute Jacobian, Hessian and compiler only when needed
- add feature to define functions in dynamical systems definition
  - e.g., mean field in Kuramoto model
  - more efficient computation
  - visually present in equations
- add links to systems in catalogues
