program circle
 implicit none
 real :: r, area
 real, parameter :: pi = 4.*atan(1.) 
 !This is apparently a common way of defining pi
 !parameter declares that the variable cannot be changed

 r = 2. !This is a real number
 area = pi*r**2 !Here, 2 is an integer
 print*,'pi is ',pi
 print*,'the area of a circle of radius ',r,' is ',area

end program circle