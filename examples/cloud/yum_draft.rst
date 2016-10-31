This script for yum based linux is a draft and has not been thoroughly tested
=============================================================================

Build from scratch in yum based AMIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    #! /bin/bash
    # Install build environment
    # The sed instruction allows the Extended Packages For
    # Enterprise Linux (epel) repository to get added to the yum package
    # manager. sed -i option edits in place.
    echo "*****************Setting up the Build Environment******************"
    sudo sed -i 's/enabled=0/enabled=1/' /etc/yum.repos.d/epel.repo
    sudo yum -y update
    sudo yum -y upgrade
    # I need to check which of these are needed
    sudo yum -y install git gcc-gfortran clang m4 patch ncurses-devel python-devel

    #Set up Julia
    echo "*****************Cloning Julia*************************************"
    git clone https://github.com/JuliaLang/julia.git
    cd julia
    git checkout v0.4.7

    #Determine the number of CPUs to build on
    NUM_CPUS=$(lscpu | awk '/^CPU\(s\):/ {print $2}')

    echo "*****************Making Julia on $NUM_CPUS CPUs***************************"
    #Takes 30 minutes on a 4CPU p2.xlarge AWS instance
    time make -j $NUM_CPUS
