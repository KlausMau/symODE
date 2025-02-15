# symODE
Python software for the algebraic (SymPy) and numerical (SciPy) treatment of ordinary differential equations (ODEs)

The used coding convention is PEP8.

## Backlog

### Features

- allow ODEs to be dependent on time explicitly

- get_limit_cycle

- find_symmetries

- inflection set (see Ref. ?)

- check given ansatz for adjoint equation

- get_isostable_map_at_fixed_points:
    - review whether the new DynamicalSystem is really necessary
    - check for degenerate case of eigenvalues with multipl. > 1

- new_coupled:
    - as default: if no specifications are given, impose mean field coupling in first variable with coupling parameter "epsilon"

- compute Jacobian, Hessian and compiler only when needed

- make possible to give functions in dynamical equations definition, e.g. mean field in Kuramoto model (more efficient computation; also visually present in equations)

- add links to systems in catalogues

### Deployment

- anaconda/pip package

- CI/CD with GitHub actions