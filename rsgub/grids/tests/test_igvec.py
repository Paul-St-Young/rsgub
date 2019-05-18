#!/usr/bin/env python
import numpy as np

def get_gvecs(nx):
  from rsgub.grids.forlib.fill_qk import fill_qk
  qvecs = fill_qk(nx, nx, nx, 0, 0, 0).T
  gcand = qvecs*nx
  gvecs = np.around(gcand).astype(int)
  return gvecs

def test_find_gvecs():
  from rsgub.grids.misc import get_regular_grid_dimensions
  nx = 3
  gvecs = get_gvecs(nx)
  gmin, gmax, gs = get_regular_grid_dimensions(gvecs)
  ngs = np.prod(gs)
  from rsgub.grids.forlib.find_gvecs import map_gvectors_to_grid
  ridx = map_gvectors_to_grid(gvecs, gmin, gs, ngs)
  from rsgub.grids.forlib.find_gvecs import igvec
  for ig, gvec in enumerate(gvecs):
    ig_expect = ig+1  # fortran uses 1-based indexing
    assert ig_expect == igvec(gvec, ridx, gmin, gmax, gs)

if __name__ == '__main__':
  test_find_gvecs()
# end __main__
