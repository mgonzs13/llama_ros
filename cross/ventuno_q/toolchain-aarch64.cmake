set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

set(CMAKE_SYSROOT /sysroot)
set(CMAKE_FIND_ROOT_PATH /sysroot /sysroot/opt/ros/jazzy /workspace/install)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_PREFIX_PATH /sysroot/opt/ros/jazzy)
set(CMAKE_CROSSCOMPILING_EMULATOR "/usr/bin/qemu-aarch64-static;-L;/sysroot")

set(ENV{PKG_CONFIG_SYSROOT_DIR} /sysroot)
set(ENV{PKG_CONFIG_LIBDIR} "/sysroot/usr/lib/aarch64-linux-gnu/pkgconfig:/sysroot/usr/lib/pkgconfig:/sysroot/usr/share/pkgconfig")

set(PYTHON_SOABI cpython-312-aarch64-linux-gnu CACHE STRING "")
