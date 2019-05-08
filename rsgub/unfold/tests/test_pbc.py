import numpy as np
from rsgub.grids.grid3d import RegularGrid3D

def test_pbc_unit_box():
  from rsgub.unfold.pbc import pbc_unit_box
  ng = np.array([2]*3)
  gmin = np.array([-0.25]*3)
  dg = np.array([0.5]*3)
  dtype = int
  rsg = RegularGrid3D(ng, gmin=gmin, dg=dg, dtype=dtype)
  vals = np.arange(np.prod(ng))
  rsg.set_col(vals)
  rsg1 = pbc_unit_box(rsg)
  vals = rsg1.get_col()
  vals0 = np.array([7, 6, 7, 6, 5, 4, 5, 4, 7, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 2, 1, 0, 1, 0, 3, 2, 3, 2, 1, 0, 1, 0, 7, 6, 7, 6, 5, 4, 5, 4, 7, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 2, 1, 0, 1, 0, 3, 2, 3, 2, 1, 0, 1, 0]
    , dtype=dtype)
  assert np.allclose(vals, vals0)

if __name__ == '__main__':
  test_pbc_unit_box()
# end __main__
