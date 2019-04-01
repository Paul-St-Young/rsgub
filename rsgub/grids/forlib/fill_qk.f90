subroutine fill_qk(nk1, nk2, nk3, k1, k2, k3, qk)
  ! --------------------------------------------------------------------------
  !   Fill ``quotient'' kgrid (qk). More specifically, uniform kgrid in
  ! reciprocal lattice units. Same order as:
  !  K_POINTS automatic
  !   nk1 nk2 nk3 k1 k2 k3
  implicit none
  integer, intent(in) :: nk1, nk2, nk3, k1, k2, k3
  double precision, intent(out) :: qk(3, nk1*nk2*nk3)
  double precision :: xkg(3, nk1*nk2*nk3)
  integer i, j, k, n, nk, nkr
  ! from PW/src/kpoint_grid.f90
  DO i=1,nk1
     DO j=1,nk2
        DO k=1,nk3
           !  this is nothing but consecutive ordering
           n = (k-1) + (j-1)*nk3 + (i-1)*nk2*nk3 + 1
           !  xkg are the components of the complete grid in crystal axis
           xkg(1,n) = dble(i-1)/nk1 + dble(k1)/2/nk1
           xkg(2,n) = dble(j-1)/nk2 + dble(k2)/2/nk2
           xkg(3,n) = dble(k-1)/nk3 + dble(k3)/2/nk3
        ENDDO
     ENDDO
  ENDDO
  nkr = nk1*nk2*nk3
  DO nk=1,nkr
    DO i=1,3
      qk(i,nk) = xkg(i,nk)-nint(xkg(i,nk))
    ENDDO
  ENDDO
end subroutine fill_qk
