cmake_minimum_required(VERSION 3.11)
project(pastis LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(
	${PROJECT_SOURCE_DIR}/CombBLAS/install/include
	${PROJECT_SOURCE_DIR}/seqan/include
	${PROJECT_SOURCE_DIR}/GPU-BSW)
	# possibly add path to boost library (not necessary on Cori, just type 'module load boost')
link_directories(
    ${PROJECT_SOURCE_DIR}/CombBLAS/install/lib
	${PROJECT_SOURCE_DIR}/GPU-BSW/build)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DSEQAN_ENABLE_DEBUG=0")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
#find_package(cuda REQUIRED)

add_executable(pastis
        src/Alphabet.cpp
        include/Alphabet.hpp
        src/Constants.cpp
        include/Constants.hpp
        include/cxxopts.hpp
        src/DistributedPairwiseRunner.cpp
        include/DistributedPairwiseRunner.hpp
        src/DistributedFastaData.cpp
        include/DistributedFastaData.hpp
        src/FastaData.cpp
        include/FastaData.hpp
        src/main.cpp
        src/ParallelFastaReader.cpp
        include/ParallelFastaReader.hpp
        src/ParallelOps.cpp
        include/ParallelOps.hpp
        src/TraceUtils.cpp
        include/TraceUtils.hpp
        include/Types.hpp
        include/Utils.hpp
        include/pw/PairwiseFunction.hpp
        src/pw/PairwiseFunction.cpp
        include/pw/OverlapFinder.hpp
        src/pw/OverlapFinder.cpp
        include/AlignmentInfo.hpp
        include/ScoreMat.hpp
        src/ScoreMat.cpp
        include/kmer/CommonKmers.hpp
        include/kmer/KmerIntersectSR.hpp
        include/kmer/MatrixEntry.hpp
        include/kmer/KmerOps.hpp
        src/kmer/KmerOps.cpp
        include/kmer/Kmer.hpp
        include/NearestKmers2.hpp
        src/NearestKmers2.cpp
        include/kmer/SubKmerIntersectSR.hpp
		include/pw/BswGPUAligner.hpp
		src/pw/BswGPUAligner.cpp
        )


target_link_libraries(pastis PUBLIC CombBLAS GraphGenlib Usortlib)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    target_link_libraries(pastis PUBLIC ${MPI_LIBRARIES})
endif()

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
if (CUDA_FOUND)
	message(STATUS "Cuda found...")
endif()
message(STATUS "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message(STATUS "${CUDART_LIBRARY}")
target_include_directories(pastis PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(pastis PUBLIC ${CUDART_LIBRARY})

# ADEPT
target_link_libraries(pastis PUBLIC bsw_gpu)

if(OPENMP_FOUND OR OpenMP_CXX_FOUND)
    # Set THREADED if OpenMP is found
    target_compile_definitions(pastis PUBLIC THREADED)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(pastis PUBLIC OpenMP::OpenMP_CXX)
    else()
        target_compile_options(pastis PUBLIC "${OpenMP_CXX_FLAGS}")
        target_link_libraries(pastis PUBLIC "${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Testing
ENABLE_TESTING()
INCLUDE(CTest)
ADD_SUBDIRECTORY(tests)
