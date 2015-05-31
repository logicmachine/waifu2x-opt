cmake_minimum_required(VERSION 2.8)
project(waifu2x-opt)

if(WIN32)
	set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif()

add_subdirectory(waifu2x)
add_subdirectory(frontend)

