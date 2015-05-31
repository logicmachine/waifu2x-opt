cmake_minimum_required(VERSION 2.8)
project(waifu2x-frontend)
find_package(OpenMP)

include_directories(../../include)

add_executable(waifu2x-frontend main.cpp)
target_link_libraries(waifu2x-frontend waifu2x)

if(NOT MSVC)
	set_target_properties(waifu2x-frontend PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
	set_target_properties(waifu2x-frontend PROPERTIES COMPILE_FLAGS "-std=c++11")
endif()
