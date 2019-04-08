import numpy as np

def resample(rsg, ng1, icol=0, method='rgi', **kwargs):
  """Interpolate regular grid and resample with a difference spacing

  Args:
    rsg (grids.grid3d.RegularGrid3D): regular grid filled with data
    ng1 (np.array): 3 integer numbers of grid points along each dimension
    icol (int, optional): data column to interpolate, default is 0
    method (str, optional): 'rgi' (RegularGridInterpolator) or 'rbf' (Rbf),
      default is 'rgi'
    kwargs (dict, optional): keyword arguments for interpolation method
  Return:
    grids.grid3d.RegularGrid3D: resampled data on a grid
  """
  # choose interpolation method
  if method == 'rgi':
    interpolate = interpolate_rgi
  elif method == 'rbf':
    interpolate = interpolate_rbf
  else:
    raise RuntimeError('unknown interpolation method %s' % method)
  # resize grid
  from rsgub.grids.grid3d import RegularGrid3D
  gmin = rsg.get_gmin()
  ng = rsg.get_ng()
  dg = rsg.get_dg()
  dg1 = (ng-1)*dg/(ng1-1)  # new grid spacing
  ncol = rsg.get_ncol()
  rsg1 = RegularGrid3D(ng1, gmin=gmin, dg=dg1, ncol=ncol)
  # transfer data with interpolation
  qvecs1 = rsg1.get_grid()
  for icol in range(ncol):
    fval = interpolate(rsg, icol)
    rsg1.add(qvecs1, fval(qvecs1), icol=icol)
  return rsg1

def interpolate_rgi(rsg, icol=0, **kwargs):
  """Interpolate regular grid using RegularGridInterpolator.

  Args:
    rsg (grids.grid3d.RegularGrid3D): regular grid filled with data
    icol (int, optional): data column to interpolate, default is 0
    kwargs (dict, optional): keyword arguments for RegularGridInterpolator
  Return:
    callable: on np.array of size (npt, ndim=3)
  """
  from scipy.interpolate import RegularGridInterpolator
  xyz = rsg.get_xyz()
  ng = rsg.get_ng()
  rvol = rsg.get_col(icol).reshape(ng)
  return RegularGridInterpolator(xyz, rvol, **kwargs)

def interpolate_rbf(rsg, icol=0, **kwargs):
  """Interpolate regular grid using radial basis function (Rbf).

  Args:
    rsg (grids.grid3d.RegularGrid3D): regular grid filled with data
    icol (int, optional): data column to interpolate, default is 0
    kwargs (dict, optional): keyword arguments for Rbf
  Return:
    callable: on np.array of size (npt, ndim=3)
  """
  from scipy.interpolate import Rbf
  rvals = rsg.get_col(icol)
  data = np.zeros([len(rvals), 4])
  data[:, :3] = rsg.get_grid()
  data[:, 3] = rvals
  x, y, z, d = data.T
  rbfi = Rbf(x, y, z, d)

  def fval(qvecs):
    qx, qy, qz = qvecs.T
    return rbfi(qx, qy, qz)
  return fval
