import numpy as np

def get_regular_grid_dimensions(gvecs):
  """Find the dimensions of a regular grid of integers

  Args:
    gvecs (np.array): array of grid points as integers
  Return:
    (np.array, np.array, np.array): (gmin, gmax, gs), the minima
     maxima and sizes of the grid
  """
  gmin = gvecs.min(axis=0)
  gmax = gvecs.max(axis=0)
  gs = np.around(gmax-gmin+1).astype(int)
  return gmin, gmax, gs
