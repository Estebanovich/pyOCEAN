!> Reads columns written by columns.py and remaps them with MOM6 remapping_core_h.
!! Usage: remap_driver in.txt out.txt
program remap_driver
  use MOM_remapping, only : remapping_CS, initialize_remapping, remapping_core_h
  implicit none
  type(remapping_CS) :: CS
  integer :: ncase, c, n0, n1
  character(len=16) :: scheme
  character(len=512) :: fin, fout
  real, allocatable :: h0(:), u0(:), h1(:), u1(:)
  call get_command_argument(1, fin) ; call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old') ; open(11, file=trim(fout), status='replace')
  read(10,*) ncase ; write(11,'(i6)') ncase
  do c = 1, ncase
    read(10,*) scheme ; read(10,*) n0, n1
    allocate(h0(n0), u0(n0), h1(n1), u1(n1))
    read(10,*) h0 ; read(10,*) u0 ; read(10,*) h1
    call initialize_remapping(CS, trim(scheme), boundary_extrapolation=.false., &
         om4_remap_via_sub_cells=.false., answer_date=99991231, nk=max(n0,n1), &
         h_neglect=1.0e-30, h_neglect_edge=1.0e-10)
    call remapping_core_h(CS, n0, h0, u0, n1, h1, u1)
    write(11,'(*(es25.17))') u1
    deallocate(h0, u0, h1, u1)
  enddo
  close(10) ; close(11)
end program remap_driver
