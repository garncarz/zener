!  ZenerNLElast.f90
!
!  FUNCTIONS:
!  ZenerNLElast - Entry point of console application.
!

!****************************************************************************
!
!  PROGRAM: ZenerNLElast
!
!  PURPOSE:  Entry point for the console application.
!
!****************************************************************************

    program ZenerNLElast

    ! use ifport


    implicit none

    ! Variables

integer, parameter :: Np = 21  ! Number of periods Tp
integer, parameter :: Nsamp = 1000 ! Number of points for period Tp

character(20)     :: TimeStampString ! Start Time Stamp for output files
double precision  :: RunTime, StartTime, EndTime

double precision :: Em, EmNL, Sig0, Tp, Tau_e, Tau_s, Alpha, R, t0
double precision :: Sigma, t, dt
integer          :: Rsig(Np)
double precision :: valSig, valEpsVE, valEpsNL
double precision :: FvalSig, FvalEps   ! Final value at the preceeding period
integer :: i,j ! Cycle variables

 ! Zener model parameters
Em=1          ! Modulus
Tau_e=50      ! Relaxation time for strain
Tau_s=25      ! Relaxation time for stress

! Nonlinear elastic serial element parameters
EmNL=0.2                ! Modulus
Alpha=3                 ! degree of nonli                                               nearity

Sig0=1                  ! initial maximum of Sigma at the end of the firs Tp
Tp=100                  ! minimal period alias period for ramping to Sig0


!DESCCRIPTION OF LOADING PATH in field (now only dimension Np)
! Signum of Rate at period Tp +1-loading, -1-unloading, 0 - delay at constant stress
Rsig=(/ +1,0,-1, +1,0,-1, +1,0,-1,  +1, +1, 0, 0,-1,-1,  +1, +1, 0, 0,-1,-1 /)

! Above Parametrs will be rewriten from INI file
open(1, FILE='ZenerNLElast.INI')
    read(1,*) Em
    read(1,*) Tau_e
    read(1,*) Tau_s
    read(1,*) EmNL
    read(1,*) Alpha
    read(1,*) Sig0
    read(1,*) Tp
    read(1,*)
    read(1,*) (Rsig(i), i=1, Np)

CLOSE(1)

! Check parameters at output
    write(*,*) 'Em   : ', Em
    write(*,*) 'Tau_e: ', Tau_e
    write(*,*) 'Tau_s: ', Tau_s
    write(*,*) 'EmNL : ', EmNL
    write(*,*) 'Alpha: ', Alpha
    write(*,*) 'Sig0 : ', Sig0
    write(*,*) 'Tp   : ', Tp
    write(*,*) 'Path :'
    write(*,*) Rsig
    pause


!Other parameters
t0=0                    ! Zero time (here just for conversion to double precision)
R= Sig0/Tp              ! The magnitude of constant rate  for Loading Unloading



dt=Tp/Nsamp
t=0  !
FvalSig=0
FvalEps=0

! Make output directory
! if (.not. MAKEDIRQQ("OUTPUT")) then
! print *, "OUTPUT subdirectory already exist"
! else
! print *, "OUTPUT subdirectory created"
! end if

!CALL DATE_AND_TIME(dateString, timeString)
TimeStampString=TimeStamp()
call cpu_time(StartTime)


open(1, file = 'OUTPUT\'//TimeStampString//'ZenerModel.dat', status = 'new')
write (1,*) "time          " , "Stress       ", "StrainNonLinEl", " StrainViscEl ", "TotStrain"
! "(5(16X))"
do i=1, Np

    do  j=1, Nsamp
t=t+dt
valSig=FvalSig+fSig(t,(i-1)*Tp,Rsig(i)*R)
valEpsNL=fEpsNL(valSig)
valEpsVE=fEpsVE (t,i)

write (1,'(5E14.7)') t, valSig, valEpsNL, valEpsVE, valEpsNL+valEpsVE
    end do
FvalSig=valSig

end do

close(1)

! Parameters - output
call cpu_time(EndTime)
RunTime=StartTime-EndTime

open(2, file = 'OUTPUT\'//TimeStampString//'ZenerModelParameters.txt', status = 'new')

write(2,*) 'Start: ', TimeStampString
write(2,*) 'Stop : ',TimeStamp ()
write(2,*) 'RunTime [sec]', RunTime
write(2,*)
write(2,*) 'Model parameters'
write(2,*) 'Em    =', Em
write(2,*) 'Tau_e =', Tau_e
write(2,*) 'Tau_s =', Tau_s
write(2,*) 'EmNL  =', EmNL
write(2,*) 'Alpha =', Alpha
write(2,*) 'Sig0  =', Sig0
write(2,*) 'Tp    =', Tp
write(2,*) 'Rate  =', R
write(2,*)
write(2,*) 'Stress path - Number of period Tp and Rsig'
write(2,*) '  Numb Rsig'
do i=1, Np
write(2,'(2I5)') i, Rsig(i)
enddo

close(2)

! Main code finish
write(*,*) 'Program END - Output writen to file'
pause

     !INTERNAL FUNCTIONS - DEFINITIONS
contains
    ! Response to ramping by stress rate R
double precision function fSig(x,x1,Rs)
    double precision, intent(in):: x, x1, Rs
    fSig=Rs * (x-x1)
    return
end function

double precision function fEps(x, x1, x2, Rs)
    double precision, intent(in):: x, x1, x2, Rs
    fEps= (Rs/Em)* &
    ((x2-x1)+(Tau_s-Tau_e)*(exp(x2/Tau_e)-exp(x1/Tau_e))*exp(-x/Tau_e))
    return
    end function

double precision function fEps0(x,Rs)
    double precision, intent(in):: x, Rs
    fEps0=fEps(x, t0, x, Rs)
    return
end function

double precision function fEpsNL(Sig)
    double precision, intent(in) :: Sig
    fEpsNL=Sig/(EmNL*(1+Alpha*Sig))
    return
end function


double precision function fEpsVE(x,i)
    double precision, intent(in) :: x
 integer, intent(in) :: i
    double precision k
        fEpsVE=fEps0(x-(i-1)*Tp,Rsig(i)*R)

        do k=1, i-1
             fEpsVE=fEpsVE+fEps(x,(k-1)*Tp, k*Tp, Rsig(k)*R)
        end do

    return
end function

! Time Stemp
function timestamp() result(str)
    implicit none
    character(len=20)                     :: str
    integer                               :: values(8)
    character(len=4)                      :: year
    character(len=2)                      :: month
    character(len=2)                      :: day, hour, minute, second
    character(len=5)                      :: zone

    ! Get current time
    call date_and_time(VALUES=values, ZONE=zone)
    write(year,'(i4.4)')    values(1)
    write(month,'(i2.2)')   values(2)
    write(day,'(i2.2)')     values(3)
    write(hour,'(i2.2)')    values(5)
    write(minute,'(i2.2)')  values(6)
    write(second,'(i2.2)')  values(7)

    str = year//'-'//month//'-'//day//'_'&
          //hour//'-'//minute//'-'//second
  end function timestamp

    end program ZenerNLElast