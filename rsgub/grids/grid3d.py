import os
import numpy as np

class RegularGrid3D:
  def __init__(self, ng, ncol=1, dtype=float, gmin=None, dg=None):
    """Create a regular grid of scalar data.
    The grid is defined by its (origin, spacing, # of points).
    The scalar data are stored as columns.

    Args:
      ng (np.array): number of grid points (3,) vector of ints
      ncol (int, optional): number of data columns, default is 1
      dtype (type): type of data columns, default is float
      gmin (np.array, optional): origin (3,) vector of floats
      dg (np.array, optional): spacing (3,) vector of floats
    """
    # define grid
    self._gmin = gmin
    self._dg = dg
    self._ng = ng
    # add empty columns
    ntot = np.prod(ng)  # total number of grid points
    self._cols = np.zeros([ntot, ncol], dtype=dtype)
    self._cols[:] = np.nan
  def _initialized(self):
    """Check if the grid structure is defined

    Return:
      bool: True if initialized
    """
    initialized = np.all([
      self._gmin is not None,
      self._dg is not None
    ])
    return initialized
  def get_grid(self):
    """Get regular grid points.

    Return:
      np.array: rgvecs, regular grid points
    """
    ndim = 3
    nxyz = [np.arange(self._ng[idim], dtype=int) for idim in range(ndim)]
    upos = np.stack(
      np.meshgrid(*nxyz, indexing='ij'),
      axis=-1
    ).reshape(-1, 3)
    rgvecs = self._gmin + self._dg*upos
    return rgvecs
  def get_col(self, icol=0):
    """Get a column of data from the regular grid

    Return:
      np.array: vals, array of scalars
    """
    return self._cols[:, icol]
  def find(self, qvec, tol=1e-2):
    """Find the index for a grid point.

    Args:
      qvec (np.array): grid point to be found
      tol (float, optional): tolerance for integer match, default 1e-2
    Return:
      int: index
    """
    nvec = (qvec-self._gmin)/self._dg
    idx3d = np.around(nvec)
    if np.any(abs(idx3d-nvec) > tol):
      msg = '%s not on grid given tol=%e' % (str(qvec), tol)
      raise RuntimeError(msg)
    return np.ravel_multi_index(idx3d.astype(int), self._ng)
  def findall(self, qvecs):
    """Vectorized version of find. Fast and furious i.e. no checking.
    If this function fails, then fall back to find.

    Args:
      qvecs (np.array): an array of grid points to be found
    Return:
      np.array: array of indices
    """
    idx3d = np.around((qvecs-self._gmin)/self._dg).astype(int)
    ng = self._ng
    idx1d = idx3d[:, 0]*ng[0]*ng[1] + idx3d[:, 1]*ng[1] + idx3d[:, 2]
    return idx1d
  def add(self, qvecs, vals, icol=0, tol=1e-2):
    """Add data to the grid.

    Args:
      qvecs (np.array): array of 3D coordinates
      vals (np.array): values at given coordinates
      icol (int, optional): which column to add to, default is 0
      tol (float, optional): tolerance for grid points, default is 1e-2
    """
    if not self._initialized():
      raise RuntimeError('must initialize before adding data')
    for qvec, val in zip(qvecs, vals):
      # find closest grid point
      idx = self.find(qvec, tol=tol)
      self._cols[idx, icol] = val
  def scatter(self, icol=0):
    import matplotlib.pyplot as plt
    from qharv.inspect import volumetric
    if self._cols is None:
      raise RuntimeError('no data to plot')
    vals = self._cols[:, icol]
    sel = ~np.isnan(vals)
    rgvecs = self.get_grid()
    fig, ax = volumetric.figax3d()
    volumetric.color_scatter(ax, rgvecs[sel], vals[sel])
    plt.show()
  def init_from_cube(self, fcube, force=False):
    from qharv.inspect.volumetric import read_gaussian_cube
    if self._initialized and (not force):
      msg = 'refusing to initialize twice;'
      msg += ' use force if you have to.'
      raise RuntimeError(msg)
    # read cube file
    entry = read_gaussian_cube(fcube)
    # initialize internal variables
    rgrid = entry['data']
    raxes = entry['axes']
    origin = entry['origin']
    self._ng = rgrid.shape
    self._dg = np.diag(raxes)
    self._gmin = origin
    self._cols = rgrid.ravel()[:, np.newaxis]
  def write_cube(self, fcube, icol=0, force=False):
    if os.path.isfile(fcube):
      msg = 'refusing to overwrite %s;' % fcube
      msg += ' use force if you have to.'
      raise RuntimeError(msg)
    from qharv.inspect.volumetric import write_gaussian_cube
    vals = self._cols[:, icol]
    vol = vals.reshape(self._ng)
    axes = np.diag(self._dg)
    text = write_gaussian_cube(vol, axes, origin=self._gmin)
    with open(fcube, 'w') as f:
      f.write(text)
