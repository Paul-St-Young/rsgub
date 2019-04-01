import numpy as np

class RegularGrid3D:
  def __init__(self, gmin=None, dg=None, ng=None, cols=None):
    """Create a regular grid of scalar data.
    The grid is defined by its (origin, spacing, # of points).
    The scalar data are stored as columns.

    Args:
      gmin (np.array, optional): origin (3,) vector of floats
      dg (np.array, optional): spacing (3,) vector of floats
      ng (np.array, optional): number of grid points (3,) vector of ints
    """
    self._gmin = gmin
    self._dg = dg
    self._ng = ng
    self._cols = cols
  def _initialized(self):
    """Check if the grid structure is defined

    Return:
      bool: True if initialized
    """
    initialized = np.all([
      self._gmin is not None,
      self._dg is not None,
      self._ng is not None
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
  def find(self, qvec, tol=1e-2):
    """Find the index for a grid point

    Args:
      qvec (np.array): grid point to be found
      tol (float, optional): tolerance on integer match, default 1e-2
    Return:
      int: index
    """
    nvec = (qvec-self._gmin)/self._dg
    idx3d = np.around(nvec)
    if np.any(abs(idx3d-nvec) > tol):
      msg = '%s not on grid given tol=%e' % (str(qvec), tol)
      raise RuntimeError(msg)
    return np.ravel_multi_index(idx3d.astype(int), self._ng)
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
    if self._cols is None:  # add first column
      ntot = np.prod(self._ng)
      self._cols = np.empty([ntot, icol+1])
      self._cols[:] = np.nan
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
    from qharv.inspect import volumetric
    if self._initialized and (not force):
      msg = 'refusing to initialize twice;'
      msg += ' use force if you have to.'
      raise RuntimeError(msg)
    # read cube file
    entry = volumetric.read_gaussian_cube(fcube)
    # initialize internal variables
    rgrid = entry['data']
    raxes = entry['axes']
    origin = entry['origin']
    self._ng = rgrid.shape
    self._dg = np.diag(raxes)
    self._gmin = origin
    self._cols = rgrid.ravel()[:, np.newaxis]
