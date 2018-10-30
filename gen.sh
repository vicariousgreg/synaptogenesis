#!/bin/bash

build_flag=serial
openmp_flag=
mpi_flag=
gui_flag=
executable_flag=
debug_flag=
jobs=
jobs_flag=

while getopts "ponmdcjhe" OPTION
do
	case $OPTION in
		p)
			build_flag=parallel
			;;
		o)
			openmp_flag='OPENMP=true'
			;;
		n)
			gui_flag='NO_GUI=true'
			;;
		m)
			mpi_flag='MPI=true'
			;;
		e)
			executable_flag='MAIN=true'
			;;
		d)
			debug_flag='DEBUG=true'
			;;
		c)
			echo "Cleaning build..."
      echo
      make clean
			;;
		j)
      eval numjobs=\$$OPTIND

      if [ "$numjobs" == '' ]; then
        echo "error: build jobs must be a positive integer"
        exit 1
			fi

			re='^[0-9]+$'
			if ! [[ $numjobs =~ $re ]] ; then
        echo "error: build jobs must be a positive integer"
        exit 1
      fi

			if test $numjobs -gt 0; then
				jobs=$numjobs
				jobs_flag="JOBS=$numjobs"
			else
        echo "error: build jobs must be a positive integer"
				exit 1
			fi

      shift
			;;
		\?|h)
			echo Builds and installs synaptogenesis
			echo "  usage: gen.sh [-p] [-o] [-n] [-d] [-c]"
			echo "    -p builds parallel version (requires CUDA)"
			echo "    -o builds with openmp"
			echo "    -n builds without GUI (normally requires GTK)"
			echo "    -m builds C++ main executable"
			echo "    -d builds with debug flags"
			echo "    -c cleans the build first"
			echo "    -j sets the number of build jobs"
			exit
			;;
	esac
done

echo ===========================
echo Building synaptogenesis ...
if [ "$build_flag" == serial ]; then
	echo "  ... serial"
else
	echo "  ... parallel"
fi

if [ "$openmp_flag" == '' ]; then
	echo "  ... without OpenMP"
else
	echo "  ... with OpenMP"
fi

if [ "$mpi_flag" == '' ]; then
	echo "  ... without MPI"
else
	echo "  ... with MPI"
fi

if [ "$gui_flag" == '' ]; then
	echo "  ... with GUI"
else
	echo "  ... without GUI"
fi

if [ "$executable_flag" == '' ]; then
	echo "  ... without C++ main executable"
else
	echo "  ... with C++ main executable"
fi

if [ "$debug_flag" == '' ]; then
	echo "  ... without debugging"
else
	echo "  ... with debugging"
fi

if [ "$jobs" != '' ]; then
	echo "  ... with" $jobs "jobs"
fi
echo ===========================

echo
echo

if ! make $build_flag $gui_flag $openmp_flag $mpi_flag $executable_flag $debug_flag $jobs_flag ; then
  echo
  echo "Failed to build!"
  exit 1
fi

echo
echo

echo ===========================
echo Installing \(requires root\) ...
echo ===========================
make install
