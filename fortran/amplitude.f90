program amplitude
 implicit none
 real :: k, l, amp
 real, parameter :: alpha = 1E11
 real, parameter :: n = 2E-6
 character(len=20) :: k1, l1
 
 call get_command_argument(1, k1)
 !if (len_trim(k) == 0) exit
 read(k1,'(f9.0)')k
 print*,k

 call get_command_argument(2, l1)
 !if (len_trim(l) == 0) exit
 read(l1,'(f9.0)')l
 print*,l

 amp = alpha * exp(-k**2/n**2 - l**2/n**2) * (k**2 + l**2)
 print*,amp

end program amplitude