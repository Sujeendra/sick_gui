How to build?
mkdir build 
cd build
cmake -DCMAKE_PREFIX_PATH=/home/sujee/open3d_install/lib/cmake/Open3D ..
make

How to run?
source /opt/ros/humble/setup.bash -- ros dependent libraries
./sick_app

Starting point of the code is main.cpp which calls main.py as well using pybind method.

<p align="center">
  <img src="sick.png" width="600"/>
</p>

