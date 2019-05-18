subroutine map_gvectors_to_grid(gvecs,&
  gmin, gs, ngs,&
  ng, ndim,&
  ridx)
  ! allocate regular grid in reciprocal space to contain gvecs
  implicit none
  integer, intent(in) :: gvecs(ng, ndim) ! gvectors to put on grid
  integer, intent(in) :: gmin(ndim), gs(ndim), ngs ! grid specifiers
  integer, intent(in) :: ng, ndim ! dimensions
  integer, intent(out) :: ridx(ngs) ! index map

  ! temporary variables
  integer :: idx3d(ndim), idx1d, ig

  ! check input
  if (ngs.ne.product(gs)) then
    print*, 'wrong size ridx'
    stop
  endif

  ! assign index to grid
  ridx = -1
  do ig=1,ng
    idx3d = gvecs(ig,:) - gmin
    idx1d = idx3d(3)+idx3d(2)*gs(3)+idx3d(1)*gs(3)*gs(2)+1
    ridx(idx1d) = ig
  enddo
end

integer function igvec(gvec, ridx, gmin, gmax, gs,&
  ndim, ngs)
  implicit none
  integer, intent(in) :: gvec(ndim) ! gvector to find
  integer, intent(in) :: ridx(ngs)  ! index map to regular grid
  integer, intent(in) :: gmin(ndim) ! regular grid minima
  integer, intent(in) :: gmax(ndim) ! regular grid maxima
  integer, intent(in) :: gs(ndim)   ! regular grid sizes
  integer, intent(in) :: ndim, ngs  ! input dimensions

  ! temporary variables
  integer :: idx3d(ndim), idx1d, i
  logical :: leftout, rightout

  ! check input
  if (ngs.ne.product(gs)) then
    print*, 'wrong size ridx'
    stop
  endif

  ! check bounds
  leftout = .false.
  rightout = .false.
  do i=1,ndim
    leftout = leftout.or.(gvec(i).lt.gmin(i))
    rightout = rightout.or.(gvec(i).gt.gmax(i))
    if (leftout.or.rightout) then
      igvec = -1
      return
    endif
  enddo

  ! find gvector
  idx3d = gvec-gmin
  idx1d = idx3d(3)+idx3d(2)*gs(3)+idx3d(1)*gs(3)*gs(2)+1
  igvec = ridx(idx1d)
end
