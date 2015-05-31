cmake_minimum_required(VERSION 2.8)
project(waifu2x)

include_directories("../../include")
find_package(OpenMP)

add_library(
	waifu2x
	waifu2x.cpp
	avx2/avx2_impl.cpp)

if(MSVC)
	set_target_properties(waifu2x PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
	set_source_files_properties(avx2/impl.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
else()
	set_target_properties(waifu2x PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -std=c++11")
	set_source_files_properties(avx2/impl.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
endif()

