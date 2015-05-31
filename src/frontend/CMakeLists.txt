cmake_minimum_required(VERSION 2.8)
project(waifu2x-frontend)

include_directories(../../include)

add_executable(waifu2x-frontend main.cpp)
target_link_libraries(waifu2x-frontend waifu2x)

