cmake_minimum_required(VERSION 3.10)
project(ImGuiExample LANGUAGES C CXX)

# Find dependencies
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(Pangolin REQUIRED)
find_package(RTABMap REQUIRED)
find_package(nav_msgs REQUIRED) 

# Find OpenCV (cv_bridge depends on OpenCV)
find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)


# Find cv_bridge (ensure that cv_bridge is installed)
find_package(cv_bridge REQUIRED)
find_package(GLU REQUIRED)
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
if(STATIC_WINDOWS_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Find installed Open3D, which exports Open3D::Open3D
find_package(Open3D REQUIRED)

# Include Directories
include_directories(
    imgui
    implot/implot
    imgui/backends
    third_party/ImGuiFileDialog  # Include ImGuiFileDialog headers
    glfw/include
    ${rclcpp_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${nav_msgs_INCLUDE_DIRS}  # Correct the inclusion of nav_msgs
    ${rosbag2_cpp_INCLUDE_DIRS}
    ${cv_bridge_INCLUDE_DIRS}  # Include cv_bridge headers
    ${OpenCV_INCLUDE_DIRS}     # Include OpenCV headers (for cv_bridge)
    /opt/ros/humble/include/cv_bridge
    # /home/sujee/ORB_SLAM3/Thirdparty/Sophus
    # /home/sujee/ORB_SLAM3/include
    # /home/sujee/ORB_SLAM3/include/CameraModels
    ${Pangolin_INCLUDE_DIRS}

)
link_directories(/opt/ros/humble/lib)

# Add Executable
add_executable(sick_app 
    main.cpp
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp
    implot/implot/implot.cpp
    implot/implot/implot_items.cpp
    third_party/ImGuiFileDialog/ImGuiFileDialog.cpp  # Add ImGuiFileDialog source
)

# Link Libraries
target_link_libraries(sick_app PRIVATE
    glfw
    OpenGL::GL
    dl
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${nav_msgs_LIBRARIES}
    ${rosbag2_cpp_LIBRARIES}
    ${OpenCV_LIBS}  # Link OpenCV (for cv_bridge)
    cv_bridge
    OpenGL::GLU
    Open3D::Open3D
    # ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
    # ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    # ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib/libg2o.so
    rtabmap::gui
    ${GTSAM_LIBRARIES}
    pybind11::embed 
    Python3::Python
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
