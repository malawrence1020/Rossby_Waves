program test_get_command_argument
 integer :: i
 character(len=32) :: arg

 i=0
 do
  call get_command_argument(i,arg)
  if (len_trim(arg) == 0) exit

  write (*,*) trim(arg)
  i = i+1
 end do
end program