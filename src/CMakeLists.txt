cmake_minimum_required(VERSION 3.25)
project(yolo_cpp)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

FILE( GLOB sources ${CMAKE_SOURCE_DIR}/*.cpp )
FILE( GLOB headers ${CMAKE_SOURCE_DIR}/*.h )

message( STATUS "HEADERS: ${headers}" ) 
message( STATUS "SOURCES: ${sources}" )


#set(PROJECT_SOURCES
 #       testV8ObjDetection.cpp
  #      )

add_executable(yolo_cpp ${sources} ${headers})

target_link_libraries(yolo_cpp ${OpenCV_LIBS})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT yolo_cpp)

