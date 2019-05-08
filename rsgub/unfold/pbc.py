import numpy as np
def pbc_unit_box(rsg, icol=0, atol=1e-3):
  """Apply periodic boundary condition (PBC) to regular grid.

  Args:
    rsg (RegularGrid3D): grid inside the unit box
    icol (int, optional): column index, default is 0
    atol (float, optional): tolerance for closeness, default is 1e-3
  Return:
    RegularGrid3D: new grid with PBC ghost points
  """
  from rsgub.grids.grid3d import RegularGrid3D
  ndim = 3
  gmin = rsg.get_gmin()
  dg = rsg.get_dg()
  ng = rsg.get_ng()
  gmax = gmin+(ng-1)*dg
  kvecs = rsg.get_grid()
  vals = rsg.get_col(icol=icol)
  # initialize bigger grid (add 1 layer of ghost points in each direction)
  #  same spacing but 2 bigger along each dimension
  ng1 = np.array(rsg.get_ng())+2
  gmin1 = gmin-dg
  rsg1 = RegularGrid3D(ng1, gmin=gmin1, dg=dg)
  # add PBC data
  for gside, gshift in zip([gmin, gmax], [1, -1]):
    #  1. mirror planes
    for idim in range(ndim):
      xmin = gside[idim]
      zsel = abs(kvecs[:, idim]-xmin) < atol
      shift = np.zeros([ndim])
      shift[idim] = gshift
      rsg1.add(kvecs[zsel]+shift, vals[zsel], icol=icol)
    #  2. mirror edges
    for idim in range(ndim):
      xmin = gside[idim]
      zsel = abs(kvecs[:, idim]-xmin) < atol
      zsel = zsel & (abs(kvecs[:, (idim+1) % ndim]-xmin) < 1e-3)
      shift = np.zeros([ndim])
      shift[idim] = gshift
      shift[(idim+1) % ndim] = gshift
      rsg1.add(kvecs[zsel]+shift, vals[zsel], icol=icol)
    #  3. mirror cornor
    zsel = np.all(abs(kvecs-gside) < atol, axis=1)
    shift = gshift*np.ones([ndim])
    rsg1.add(kvecs[zsel]+shift, vals[zsel], icol=icol)
  # end for gside, gshift

  # add original data
  rsg1.add(kvecs, vals, icol=icol)
  ## check success
  #vals1 = rsg1.get_col()
  #success = ~np.isnan(vals1).any()
  #assert success
  return rsg1

def pbc_unit_box_one_side(rsg, atol=1e-3):
  """Apply periodic boundary condition (PBC) to regular grid.

  !!!! assume grid sits in unit box [-.5, .5) with one side sitting
  close to the box edge -.5 !!!!

  Args:
    rsg (RegularGrid3D): grid inside the unit box
    icol (int, optional): column index, default is 0
    atol (float, optional): tolerance for closeness, default is 1e-3
  Return:
    RegularGrid3D: new grid with PBC applied
  """
  from rsgub.grids.grid3d import RegularGrid3D
  ndim = 3
  # initialize bigger grid
  ncol = rsg.get_ncol()
  ng1 = np.array(rsg.get_ng())+1
  rsg1 = RegularGrid3D(ng1, gmin=rsg.get_gmin(), dg=rsg.get_dg(), ncol=ncol)
  kvecs = rsg.get_grid()
  for icol in range(ncol):
    # add original data
    vals = rsg.get_col(icol=icol)
    rsg1.add(kvecs, vals, icol=icol)
    # add PBC data
    #  1. mirror planes
    for idim in range(ndim):
      zsel = abs(kvecs[:, idim]+0.5) < atol
      shift = np.zeros([ndim])
      shift[idim] = 1.
      rsg1.add(kvecs[zsel]+shift, vals[zsel], icol=icol)
    #  2. mirror edges
    for idim in range(ndim):
      zsel = abs(kvecs[:, idim]+0.5) < atol
      zsel = zsel & (abs(kvecs[:, (idim+1) % ndim]+0.5) < 1e-3)
      shift = np.zeros([ndim])
      shift[idim] = 1.
      shift[(idim+1) % ndim] = 1.
      rsg1.add(kvecs[zsel]+shift, vals[zsel], icol=icol)
    #  2. mirror cornor
    zsel = np.all(abs(kvecs+0.5) < atol, axis=1)
    shift = np.ones([ndim])
    rsg1.add(kvecs[zsel]+shift, vals[zsel], icol=icol)
  return rsg1
