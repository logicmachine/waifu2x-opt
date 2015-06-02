cmake_minimum_required(VERSION 2.8)
project(waifu2x)

include_directories("../../include")
find_package(OpenMP)

add_library(
	waifu2x
	waifu2x.cpp
	avx2/avx2_impl.cpp
	avx2_fp16/avx2_fp16_impl.cpp)

if(MSVC)
	# Use static runtime
	foreach(flag_var
	        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
	        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
		if(${flag_var} MATCHES "/MD")
			string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
		endif()
	endforeach(flag_var)
	set_target_properties(waifu2x PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
	set_source_files_properties(avx2/avx2_impl.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
	set_source_files_properties(avx2_fp16/avx2_fp16_impl.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
else()
	set_target_properties(waifu2x PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -std=c++11")
	set_source_files_properties(avx2/avx2_impl.cpp PROPERTIES COMPILE_FLAGS "-mfma -mavx2")
	set_source_files_properties(avx2_fp16/avx2_fp16_impl.cpp PROPERTIES COMPILE_FLAGS "-mfma -mavx2")
endif()

