

module control_vmc
    !> Arguments: idump, irstar, isite, nconf, nblk, nblk_max, nblkeq, nconf_new, nstep,
    !> icharged_atom, nblk_ci for vmc module

    integer :: vmc_idump
    integer :: vmc_irstar
    integer :: vmc_isite
    integer :: vmc_nconf
    integer :: vmc_nblk
    integer :: vmc_nblk_max
    integer :: vmc_nblkeq
    integer :: vmc_nconf_new
    integer :: vmc_nstep
    integer :: vmc_icharged_atom
    integer :: vmc_nblk_ci

    private
    public :: vmc_idump, vmc_irstar, vmc_isite, vmc_nconf, vmc_nblk, vmc_nblk_max
    public :: vmc_nblkeq, vmc_nconf_new, vmc_nstep, vmc_icharged_atom, vmc_nblk_ci
    save
end module control_vmc

module control_dmc
    !> Arguments: idump, irstar, isite, nconf, nblk, nblk_max, nblkeq, nconf_new, nstep,
    !> icharged_atom, nblk_ci for vmc module

    integer :: dmc_idump
    integer :: dmc_irstar
    integer :: dmc_isite
    integer :: dmc_nconf
    integer :: dmc_nblk
    integer :: dmc_nblk_max
    integer :: dmc_nblkeq
    integer :: dmc_nconf_new
    integer :: dmc_nstep
    integer :: dmc_icharged_atom
    integer :: dmc_nblk_ci

    private
    public :: dmc_idump, dmc_irstar, dmc_isite, dmc_nconf, dmc_nblk, dmc_nblk_max
    public :: dmc_nblkeq, dmc_nconf_new, dmc_nstep, dmc_icharged_atom, dmc_nblk_ci
    save
end module control_dmc



module contrl_file

    implicit None
    character(20) :: log_filename
    character(20) :: proc_filename
    character(80) :: file_input, file_output, file_error
    integer       :: iunit, ounit, errunit

    private
    public :: log_filename, proc_filename
    public :: file_input, file_output, file_error
    public :: close_files
    public :: init_procfile, init_logfile, initialize
    public :: iunit, ounit, errunit
    save
contains

    subroutine initialize()
        ! Ravindra
        use mpiconf, only: wid      ! logical :: true only for mpirank=0
        use, intrinsic :: iso_fortran_env !, only: stdin=>input_unit, stdout=>output_unit, stderr=>error_unit


        implicit none
        character(len=40), parameter            :: VERSION = 'v2.0.0'
        character(len=80), allocatable          :: arg(:)
        integer                                 :: i, j, iostat, argcount
        character(len=10), dimension(12)        :: extensions
        character(len=100)                      :: string_format  = '(A, T40, A)'

        ! Get all the command line arguments

        argcount = command_argument_count()
        if ( .not. allocated(arg)) allocate(arg(12))
        do i = 1, argcount
            call get_command_argument(i, arg(i))
        end do

        do i = 1,argcount
            select case (arg(i))
                case ('-v', '-ver', '--version')
                    write(*, '(a)') 'version ', VERSION
                    stop

                case ('-V', '--verbose')
                    write(output_unit, '(a)') 'verbose mode '

                case ('-d', '-debug', '--debug')
                    write(output_unit, '(a)') 'debug mode '

                case ('-p', '-prefix', '--prefix')
                    write(output_unit, '(a)') 'prefix text for generated files '

                case ('-i', '-in', '-inp', '-input', '--input')
                    if ((index(arg(i+1), ".in") /= 0) .or. (index(arg(i+1), ".inp") /= 0) .or. &
                        (index(arg(i+1), ".dat") /= 0) ) then
                        file_input = arg(i+1)
                    else
                        write(error_unit,*) "input file should have an extention .in / .inp / .dat"
                        stop
                    endif
!                        write(output_unit, fmt=string_format) ' input file      :: ', file_input
                    if (wid) then
                        open (newunit=iunit,file=file_input, iostat=iostat, action='read' )
                        if (iostat /= 0) error stop "error in opening input file"
                    endif

                case ('-o', '-ou', '-out', '-output', '--output')
                    if ((index(arg(i+1), ".out") /= 0) .or. (index(arg(i+1), ".log") /= 0) .or. &
                        (index(arg(i+1), ".dat") /= 0) ) then
                        file_output = arg(i+1)
                    else
                        write(error_unit,*) "output file should have an extention .log / .out / .dat"
                        stop
                    endif
                    if (.not. wid ) then
                        file_output = '/dev/null'
                        close (6)
                        open (6, file='/dev/null')
                    endif
                    open (newunit=ounit,file=file_output, iostat=iostat, action='write', status='unknown' )
                    if (iostat /= 0) error stop "error in opening output unit"

                case ('-e', '-er', '-err', '-error', '--error')
                    if ((index(arg(i+1), "error") /= 0) .or. (index(arg(i+1), ".err") /= 0) .or. &
                        (index(arg(i+1), ".e") /= 0) ) then
                        file_error = arg(i+1)
                    else
                        write(error_unit,*) "error file should be named 'error' or should have &
                                            an extention .e / .err to the filename"
                        stop
                    endif
                    open (newunit=errunit,file=file_error, iostat=iostat, action='write' )

                case ('-h', '--help')
                    call print_help()
                    stop

                case default
                    extensions(1) = ".in"  ; extensions(2) = ".inp" ; extensions(3) = ".dat" ; extensions(4) = ".out" ;
                    extensions(5) = ".log" ; extensions(6) = ".err" ; extensions(7) = ".e"   ; extensions(8) = "error"

                    do j = 1, 8
                        if (index(arg(i+1), extensions(j)) /= 0) then
                            write(output_unit, '(2a)') 'unrecognised command-line option: ', arg(i)
                            call print_help()
                        endif
                    enddo

            end select
        enddo
        if ( allocated(arg)) deallocate(arg)


        contains
        subroutine print_help()
            write(output_unit, '(a, /)') "Command line options::"
            write(output_unit, '(a, /)') "  '-i', '-in', '-inp', '-input',  '--input'   Provide input  file i.e. '-i input.inp' "
            write(output_unit, '(a, /)') "  '-o', '-ou', '-out', '-output', '--output'  Provide output file i.e. '-o output.log' "
            write(output_unit, '(a, /)') "  '-e', '-er', '-err', '-error',  '--error'   Provide error  file i.e. '-e error' "

            write(output_unit, '(a, /)') "  '-v', '-ver', '--version'                   print version information"
            write(output_unit, '(a, /)') "  '-h', '--help'                              print usage information"
            write(output_unit, '(a, /)') "  '-V', '--verbose'                           verbose mode printing"
            write(output_unit, '(a, /)') "  '-d', '-debug', '--debug'                   run in debug mode"
            write(output_unit, '(a, /)') "  '-p', '-prefix', '--prefix'                 Attach prefix to generated filenames"
        end subroutine print_help

    end subroutine initialize



    subroutine init_procfile()
        use mpiconf, only: idtask
        use const, only: ipr

        implicit none

        integer  :: iunit

        if (ipr .gt. 1) then
            write(proc_filename, '(a,i5.5)') "check.", idtask
            open (newunit=iunit, form='formatted', file=proc_filename)
        endif
    end subroutine init_procfile

end module contrl_file