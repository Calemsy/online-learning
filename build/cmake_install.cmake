# Install script for directory: /data0/users/shuaishuai3/wt/t/t1_8

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data0/users/shuaishuai3/wt/t/t1_8/src/libpytrain.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data0/users/shuaishuai3/wt/t/t1_8/src/libpytrain.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data0/users/shuaishuai3/wt/t/t1_8/src/libpytrain.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data0/users/shuaishuai3/wt/t/t1_8/src/libpytrain.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/data0/users/shuaishuai3/wt/t/t1_8/src" TYPE SHARED_LIBRARY FILES "/data0/users/shuaishuai3/wt/t/t1_8/build/libpytrain.so")
  if(EXISTS "$ENV{DESTDIR}/data0/users/shuaishuai3/wt/t/t1_8/src/libpytrain.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data0/users/shuaishuai3/wt/t/t1_8/src/libpytrain.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data0/users/shuaishuai3/wt/t/t1_8/src/libpytrain.so"
         OLD_RPATH "/data1/shiduo/anaconda2/envs/py36t/lib:/data0/users/shuaishuai3/wt/grpc/cmake/build:/data0/users/shuaishuai3/wt/grpc/cmake/build/third_party/cares/cares/lib:/data0/users/shuaishuai3/wt/grpc/cmake/build/third_party/zlib:/data0/users/shuaishuai3/wt/tensorflow-1.15.0/bazel-bin/tensorflow:/data0/users/shuaishuai3/wt/t/t1_8/./third_part/libs:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}/data0/users/shuaishuai3/wt/t/t1_8/src/libpytrain.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/data0/users/shuaishuai3/wt/t/t1_8/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
