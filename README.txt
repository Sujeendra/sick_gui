How to build?
mkdir build 
cd build
cmake -DCMAKE_PREFIX_PATH=/home/sujee/open3d_install/lib/cmake/Open3D ..
make

How to run?
./sick_app

Starting point of the code is main.cpp which calls main.py as well using pybind method.

