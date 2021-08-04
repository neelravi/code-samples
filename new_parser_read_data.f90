
subroutine header_printing()
    !> This subroutine prints the header in each output file. It contains some
    !! useful information about the compilers, version of the code, input and output file names.
    !! @author Ravindra Shinde (r.l.shinde@utwente.nl)

    use mpi
    use mpiconf, only: idtask, nproc
    use, intrinsic :: iso_fortran_env, only: iostat_end
    use contrl_file,    only: file_input, file_output, file_error
    use contrl_file,    only: ounit, errunit

    implicit none

    integer                             :: status, i
    character(len=8)                    :: date
    character(len=10)                   :: time
    character(len=40)                   :: env_variable
    character(len=100)                  :: input_filename, output



    write(ounit,*) "____________________________________________________________________"
    write(ounit,*)
    write(ounit,*)
    write(ounit,*) ' .d8888b.   888    888         d8888  888b     d888  8888888b. '
    write(ounit,*) 'd88P  Y88b  888    888        d88888  8888b   d8888  888   Y88b'
    write(ounit,*) '888    888  888    888       d88P888  88888b.d88888  888    888'
    write(ounit,*) '888         8888888888      d88P 888  888Y88888P888  888   d88P'
    write(ounit,*) '888         888    888     d88P  888  888 Y888P 888  8888888P" '
    write(ounit,*) '888    888  888    888    d88P   888  888  Y8P  888  888       '
    write(ounit,*) 'Y88b  d88P  888    888   d8888888888  888   "   888  888       '
    write(ounit,*) ' "Y8888P"   888    888  d88P     888  888       888  888       '
    write(ounit,*)
    write(ounit,*) "____________________________________________________________________"
    write(ounit,*)
    write(ounit,*) ' Cornell Holland Ab-initio Materials Package'
    write(ounit,*)
    write(ounit,*)

    write(ounit,*) " information about the contributors goes here"
    write(ounit,*)
    write(ounit,*)
    write(ounit,*) " https://github.com/filippi-claudia/champ"
    write(ounit,*)
    write(ounit,*)

    write(ounit,*) " paper to cite for this code goes here"
    write(ounit,*)
    write(ounit,*)
    write(ounit,*)
    write(ounit,*)

    write(ounit,*) " license information goes here"

    write(ounit,*) "____________________________________________________________________"
    write(ounit,*)
    write(ounit,*)
    write(ounit,*)
    write(ounit,*)

    call date_and_time(date=date,time=time)
    write(ounit, '(12a)') " Calculation started on     :: ",  &
                            date(1:4), "-", date(5:6), "-", date(7:8), " at ",  time(1:2), ":", time(3:4), ":", time(5:6)
    call get_command_argument(number=0, value=output)
    write(ounit, '(2a)') " Executable                 :: ",   output

#if defined(GIT_BRANCH)
    write(ounit,'(2a)')  " Git branch                 :: ", GIT_BRANCH
#endif

#if defined(GIT_HASH)
    write(ounit,'(2a)')  " Git commit hash            :: ", GIT_HASH
#endif

#if defined(COMPILER)
    write(ounit,'(2a)')  " Compiler                   :: ", COMPILER
#endif

#if defined(COMPILER_VERSION)
    write(ounit,'(2a)')  " Compiler version           :: ", COMPILER_VERSION
#endif

    call hostnm(output)
    write(ounit, '(2a)') " Hostname                   :: ",   output
    call get_environment_variable ("PWD", output)
    write(ounit, '(2a)') " Current directory          :: ",   output
    call get_environment_variable ("USER", output)
    write(ounit, '(2a)') " Username                   :: ",   output
    write(ounit, '(2a)') " Input file                 :: ",   file_input
    write(ounit, '(2a)') " Output file                :: ",   file_output
    write(ounit, '(2a)') " Error file                 :: ",   file_error
    write(ounit, '(4a)') " Code compiled on           :: ",__DATE__, " at ", __TIME__
    write(ounit, '(a,i0)') " Number of processors       :: ", nproc
    write(ounit,*)



end subroutine header_printing


subroutine read_molecule_file(file_molecule)
    !> This subroutine reads the .xyz molecule file. It then computes the
    !! number of types of atoms, nuclear charges (from the symbol), and
    !! number of valence electrons if pseudopotential is provided.
    !! @author Ravindra Shinde (r.l.shinde@utwente.nl)
    !! @date
    use custom_broadcast,   only: bcast
    use mpiconf,            only: wid
    use atom,               only: znuc, cent, pecent, iwctype, nctype, ncent, ncent_tot, nctype_tot, symbol, atomtyp
    use ghostatom, 		    only: newghostype, nghostcent
    use inputflags,         only: igeometry
    use periodic_table,     only: atom_t, element
    use contrl_file,        only: ounit, errunit
    use general,            only: pooldir

    implicit none

    !   local use
    character(len=72), intent(in)   :: file_molecule
    character(len=40)               :: temp1, temp2, temp3, temp4
    character(len=80)               :: comment, file_molecule_path
    integer                         :: iostat, i, j, k, iunit
    logical                         :: exist
    type(atom_t)                    :: atoms
    character(len=2), allocatable   :: unique(:)

    !   Formatting
    character(len=100)               :: int_format     = '(A, T60, I0)'
    character(len=100)               :: float_format   = '(A, T60, f12.8)'
    character(len=100)               :: string_format  = '(A, T60, A)'

    !   External file reading

    if((file_molecule(1:6) == '$pool/') .or. (file_molecule(1:6) == '$POOL/')) then
        file_molecule_path = pooldir // file_molecule(7:)
    else
        file_molecule_path = file_molecule
    endif

    write(ounit,*) '-----------------------------------------------------------------------'
    write(ounit,string_format)  " Reading molecular coordinates from the file :: ",  file_molecule_path
    write(ounit,*) '-----------------------------------------------------------------------'

    if (wid) then
        inquire(file=file_molecule_path, exist=exist)
        if (exist) then
            open (newunit=iunit,file=file_molecule_path, iostat=iostat, action='read' )
            if (iostat .ne. 0) stop "Problem in opening the molecule file"
        else
            call fatal_error (" molecule file "// pooldir // trim(file_molecule) // " does not exist.")
        endif

        read(iunit,*) ncent
    endif
    call bcast(ncent)

    write(ounit,fmt=int_format) " Number of atoms ::  ", ncent
    write(ounit,*)

    if (.not. allocated(cent)) allocate(cent(3,ncent))
    if (.not. allocated(symbol)) allocate(symbol(ncent))
    if (.not. allocated(iwctype)) allocate(iwctype(ncent))
    if (.not. allocated(unique)) allocate(unique(ncent))

    if (wid) read(iunit,'(A)')  comment
    call bcast(comment)

    write(ounit,*) "Comment from the molecule file :: ", trim(comment)
    write(ounit,*)

    if (wid) then
        do i = 1, ncent
            read(iunit,*) symbol(i), cent(1,i), cent(2,i), cent(3,i)
        enddo
    endif
    call bcast(symbol)
    call bcast(cent)

    if (wid) close(iunit)


    ! Count unique type of elements
    nctype = 1
    unique(1) = symbol(1)
    do j= 2, ncent
        if (any(unique == symbol(j) ))  cycle
        nctype = nctype + 1
        unique(nctype) = symbol(j)
    enddo

    write(ounit,fmt=int_format) " Number of distinct types of elements (nctype) :: ", nctype
    write(ounit,*)

    if (.not. allocated(atomtyp)) allocate(atomtyp(nctype))
    if (.not. allocated(znuc)) allocate(znuc(nctype))

    ! get the correspondence for each atom according to the rule defined for atomtypes
    do j = 1, ncent
        do k = 1, nctype
            if (symbol(j) == unique(k))   iwctype(j) = k
        enddo
    enddo

    ! Get the correspondence rule
    do k = 1, nctype
        atomtyp(k) = unique(k)
    enddo

    if (allocated(unique)) deallocate(unique)

    ! Get the znuc for each unique atom
    do j = 1, nctype
        atoms = element(atomtyp(j))
        znuc(j) = atoms%nvalence
    enddo

    ncent_tot = ncent + nghostcent
    nctype_tot = nctype + newghostype

    write(ounit,*) '-----------------------------------------------------------------------'
    write(ounit,'(a, t15, a, t27, a, t39, a, t45, a)') 'Symbol', 'x', 'y', 'z', 'Type'
    write(ounit,'(t14, a, t26, a, t38, a )') '(A)', '(A)', '(A)'
    write(ounit,*) '-----------------------------------------------------------------------'

    do j= 1, ncent
        write(ounit,'(A4, 2x, 3F12.6, 2x, i3)') symbol(j), (cent(i,j),i=1,3), iwctype(j)
    enddo

    write(ounit,*) '-----------------------------------------------------------------------'
    write(ounit,*) " Values of znuc (number of valence electrons) "
    write(ounit,'(10F12.6)') (znuc(j), j = 1, nctype)
    write(ounit,*) '-----------------------------------------------------------------------'
    write(ounit,*)
end subroutine read_molecule_file


subroutine read_determinants_file(file_determinants)
    !> This subroutine reads the single state determinant file.
    !! @author Ravindra Shinde

    use custom_broadcast,   only: bcast
    use mpiconf,            only: wid
    use, intrinsic :: iso_fortran_env, only: iostat_eor
    use contrl_file,    only: ounit, errunit
    use dets,           only: cdet, ndet
    use vmc_mod,        only: MDET
    use dorb_m,         only: iworbd
    use coefs,          only: norb
    use inputflags,     only: ideterminants
    use wfsec,          only: nwftype
    use csfs,           only: nstates
    use mstates_mod,    only: MSTATES
    use general,        only: pooldir
    use elec,           only: ndn, nup
    use const,          only: nelec

    implicit none

    !   local use
    character(len=72), intent(in)   :: file_determinants
    character(len=80)               :: temp1, temp2, temp3
    integer                         :: iostat, i, j, iunit, counter
    logical                         :: exist, skip = .true.

    !   Formatting
    character(len=100)               :: int_format     = '(A, T40, I8)'
    character(len=100)               :: string_format  = '(A, T40, A)'

    !   External file reading
    write(ounit,*) '------------------------------------------------------'
    write(ounit,string_format)  " Reading determinants from the file :: ",  trim(file_determinants)
    write(ounit,*) '------------------------------------------------------'

    if (wid) then
        inquire(file=file_determinants, exist=exist)
        if (exist) then
            open (newunit=iunit,file=file_determinants, iostat=iostat, action='read' )
            if (iostat .ne. 0) stop "Problem in opening the determinant file"
        else
            call fatal_error (" determinant file "// trim(file_determinants) // " does not exist.")
        endif
    endif

    ndn  = nelec - nup

    write(ounit,*)
    write(ounit,int_format) " Number of total electrons ", nelec
    write(ounit,int_format) " Number of alpha electrons ", nup
    write(ounit,int_format) " Number of beta  electrons ", ndn
    write(ounit,*)


    ! to escape the comments before the "determinants" line
    if (wid) then
        do while (skip)
            read(iunit,*, iostat=iostat) temp1
            temp1 = trim(temp1)
            if (temp1 == "determinants") then
                backspace(iunit)
                skip = .false.
            endif
        enddo
    endif

!   Read the first main line
    if (wid) then
        read(iunit, *, iostat=iostat)  temp2, ndet, nwftype
        if (iostat == 0) then
            if (trim(temp2) == "determinants") write(ounit,int_format) " Number of determinants ", ndet
        else
            call fatal_error ("Error in reading number of determinants / number of wavefunction types")
        endif
    endif
    call bcast(ndet)
    call bcast(nwftype)

    ! Note the hack here about capitalized variables. DEBUG
    MDET = ndet

    if (.not. allocated(cdet)) allocate(cdet(ndet,MSTATES,nwftype))

    if (wid) then
        read(iunit,*, iostat=iostat) (cdet(i,1,1), i=1,ndet)
        if (iostat /= 0) call fatal_error( "Error in determinant coefficients ")
    endif
    call bcast(cdet)

    write(ounit,*)
    write(ounit,*) " Determinant coefficients "
    write(ounit,'(10(1x, f11.8, 1x))') (cdet(i,1,1), i=1,ndet)

!       allocate the orbital mapping array
    if (.not. allocated(iworbd)) allocate(iworbd(nelec, ndet))

    if (wid) then
        do i = 1, ndet
            read(iunit,*, iostat=iostat) (iworbd(j,i), j=1,nelec)
            if (iostat /= 0) call fatal_error("Error in reading orbital -- determinants mapping ")
        enddo
    endif
    call bcast(iworbd)
    ! This part replaces a call to verify_orbitals
    !if(any(iworbd .gt. norb))  call fatal_error('INPUT: iworbd > norb')


    write(ounit,*)
    write(ounit,*) " Orbitals <--> Determinants mapping :: which orbitals enter in which dets"
    do i = 1, ndet
        write(ounit,'(<nelec>(i4, 1x))') (iworbd(j,i), j=1,nelec)
    enddo

    if (wid) then
        read(iunit,*) temp1
        if (temp1 == "end" ) write(ounit,*) " Single state determinant file read successfully "
        ideterminants = ideterminants + 1
    endif
    call bcast(ideterminants)

    if (wid) close(iunit)

end subroutine read_determinants_file
