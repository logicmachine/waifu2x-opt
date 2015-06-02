cmake_minimum_required(VERSION 2.8)
project(waifu2x-frontend)

find_package(OpenCV REQUIRED)
find_package(OpenMP)

include_directories(../../include)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(waifu2x-frontend main.cpp)
target_link_libraries(waifu2x-frontend waifu2x)
target_link_libraries(waifu2x-frontend ${OpenCV_LIBS})

if(MSVC)
	# Use static runtime
	foreach(flag_var
	        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
	        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
		if(${flag_var} MATCHES "/MD")
			string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
		endif()
	endforeach(flag_var)
else()
	set_target_properties(waifu2x-frontend PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
	set_target_properties(waifu2x-frontend PROPERTIES COMPILE_FLAGS "-std=c++11")
endif()
