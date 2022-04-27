cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(execute_poly)

find_package(PCL 1.2 REQUIRED)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
include_directories(${PCL_INCLUDE_DIRS})
#link_directories(LINK_WITH pcl_filters pcl_io pcl_kdtree pcl_features)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (execute_poly versao_poly.cpp)
target_link_libraries (execute_poly ${PCL_LIBRARIES})
