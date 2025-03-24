#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <open3d/Open3D.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include "rclcpp/serialization.hpp" // Include the serialization header
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <GL/gl.h>
#include "ImGuiFileDialog.h"
#include <map>
#include <GL/glu.h>
#include <thread>
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"

#include <rtabmap/core/Odometry.h>
#include "rtabmap/core/Rtabmap.h"
#include "rtabmap/core/RtabmapThread.h"
#include "rtabmap/core/CameraRGBD.h"
#include "rtabmap/core/CameraStereo.h"
#include "rtabmap/core/CameraThread.h"
#include "rtabmap/core/OdometryThread.h"
#include "rtabmap/core/Graph.h"
#include "rtabmap/utilite/UEventsManager.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/filter.h>
#include <atomic>
#include <sensor_msgs/msg/imu.hpp>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/utilite/ULogger.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <memory>
#include <pybind11/embed.h>

using namespace rtabmap;
namespace py = pybind11;

// LiDAR data structure
struct Point3D
{
    float x, y, z, intensity;
    float r, g, b; // Color for each point
    bool clicked;  // Flag to check if the point has been clicked

    Point3D(float x, float y, float z, float intensity)
        : x(x), y(y), z(z), intensity(intensity), r(0.0f), g(0.0f), b(0.0f), clicked(false) {}
};
struct IMUData
{
    Eigen::Quaternionf orientation;
    Eigen::Vector3f angularVelocity;
    Eigen::Vector3f linearAcceleration;
};
class MapBuilder
{
private:
    rtabmap::Rtabmap rtabmap_;
    rtabmap::ParametersMap parameters_;
    std::vector<Eigen::Matrix4f> local_transforms_;
    int seq_;
    // Convert Point3D to LaserScan format
    rtabmap::LaserScan convertToLaserScan(const std::vector<Point3D> &points)
    {
        // First convert to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->reserve(points.size());

        for (const auto &point : points)
        {
            pcl::PointXYZRGB pcl_point;
            pcl_point.x = point.x;
            pcl_point.y = point.y;
            pcl_point.z = point.z;
            pcl_point.r = static_cast<uint8_t>(point.r * 255);
            pcl_point.g = static_cast<uint8_t>(point.g * 255);
            pcl_point.b = static_cast<uint8_t>(point.b * 255);
            cloud->push_back(pcl_point);
        }

        // Create scan data using RTAB-Map's utility function
        return rtabmap::util3d::laserScanFromPointCloud(*cloud);
    }

    rtabmap::Transform imuToTransform(const IMUData &imu_data)
    {
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

        // // Print IMU orientation (quaternion)
        // std::cout << "IMU Orientation: "
        //           << "x=" << imu_data.orientation.x() << ", "
        //           << "y=" << imu_data.orientation.y() << ", "
        //           << "z=" << imu_data.orientation.z() << ", "
        //           << "w=" << imu_data.orientation.w() << std::endl;

        // Convert orientation to rotation matrix
        Eigen::Matrix3f rotation = imu_data.orientation.toRotationMatrix();

        // // Print rotation matrix
        // std::cout << "Rotation Matrix: \n"
        //           << rotation << std::endl;

        // // Print IMU linear acceleration
        // std::cout << "IMU Linear Acceleration: "
        //           << "x=" << imu_data.linearAcceleration.x() << ", "
        //           << "y=" << imu_data.linearAcceleration.y() << ", "
        //           << "z=" << imu_data.linearAcceleration.z() << std::endl;

        // Here we assume linearAcceleration is used as translation (revisit logic)
        Eigen::Vector3f translation = imu_data.linearAcceleration;

        // Construct the transform matrix
        transform.block<3, 3>(0, 0) = rotation;
        transform.block<3, 1>(0, 3) = translation;

        // Print the full transform matrix
        // std::cout << "Transform Matrix: \n"
        //           << transform << std::endl;

        // Return the RTAB-Map Transform object
        return rtabmap::Transform::fromEigen4f(transform);
    }

public:
    MapBuilder() : seq_(0)
    {
        // Configure RTAB-Map parameters
        parameters_ = rtabmap::ParametersMap({
            {rtabmap::Parameters::kRegStrategy(), "1"},             // ICP registration
            {rtabmap::Parameters::kRegForce3DoF(), "false"},        // 6DoF SLAM
            {rtabmap::Parameters::kRGBDProximityBySpace(), "true"}, // Local loop closure detection
            {rtabmap::Parameters::kRGBDAngularUpdate(), "0.1"},     // Update map on 0.1 rad change
            {rtabmap::Parameters::kRGBDLinearUpdate(), "0.1"},      // Update map on 0.1 m change
        });

        rtabmap_.init(parameters_, "");
    }

    void processFrame(const std::vector<Point3D> &points, const IMUData &imu_data)
    {
        // Convert point cloud to LaserScan format
        rtabmap::LaserScan laser_scan = convertToLaserScan(points);

        // Get transform from IMU
        rtabmap::Transform current_transform = imuToTransform(imu_data);
        // std::cout << "Transform: " << current_transform.prettyPrint() << std::endl;

        // Create sensor data using LaserScan constructor
        rtabmap::SensorData sensor_data(
            laser_scan,             // LaserScan data
            cv::Mat(),              // RGB image (empty)
            cv::Mat(),              // Depth image (empty)
            rtabmap::CameraModel(), // Camera model
            seq_,                   // Sequence number
            0.0,                    // Timestamp (you might want to add actual timestamps)
            cv::Mat()               // User data (empty)
        );
        // Process data with RTAB-Map
        std::vector<float> dummy_confidence;
        std::map<std::string, float> dummy_stats;

        // rtabmap_.process(sensor_data, current_transform, cv::Mat::eye(6, 6, CV_64FC1), dummy_confidence, dummy_stats);

        // Store transform for trajectory
        // local_transforms_.push_back(current_transform.toEigen4f());

        // seq_++;
    }

    void saveMap(const std::string &filename)
    {
        // Close and save the database
        rtabmap_.close(true, "results.db");

        // Save the trajectory
        std::string trajectory_filename = filename + "_trajectory.txt";
        std::ofstream trajectory_file(trajectory_filename);
        if (trajectory_file.is_open())
        {
            for (const auto &transform : local_transforms_)
            {
                // Save in format: tx ty tz qx qy qz qw
                Eigen::Vector3f translation = transform.block<3, 1>(0, 3);
                Eigen::Quaternionf quaternion(transform.block<3, 3>(0, 0));
                trajectory_file << translation.x() << " "
                                << translation.y() << " "
                                << translation.z() << " "
                                << quaternion.x() << " "
                                << quaternion.y() << " "
                                << quaternion.z() << " "
                                << quaternion.w() << "\n";
            }
            trajectory_file.close();
        }
    }

    std::vector<Eigen::Matrix4f> getTrajectory() const
    {
        return local_transforms_;
    }

    void reset()
    {
        rtabmap_.resetMemory();
        local_transforms_.clear();
        // seq_ = 0;
    }
};
bool showprocessSLAM = false;
bool IsProcessed = false;
std::atomic<bool> isProcessing(false);
std::atomic<float> progress(0.0f);
std::thread backgroundThread;

// Global variables for camera
float cameraAngleX = 0.0f; // Horizontal rotation
float cameraAngleY = 0.0f; // Vertical rotation
float zoomLevel = 15.0f;   // Zoom distance
bool isDragging = false;   // Mouse drag state
double lastMouseX, lastMouseY;
GLuint imageTexture;
std::vector<cv::Mat> imageFrames; // Vector to hold image frames
std::vector<IMUData> imuFrames;
std::string bagFilePath = ""; // Initialize as empty
int max_label = 0;
bool showClusters = false;
std::atomic<bool> IsVisualizerOpen(false); // To track if the visualizer is open
bool showPlane = false;
bool applyMesh = false;
std::shared_ptr<open3d::geometry::TriangleMesh> mesh; // Your mesh

// Vector to hold point cloud data for each frame
std::vector<std::vector<Point3D>> pointCloudFrames;
std::vector<double> pointCloudTimestamps; // Store timestamps for each frame
int currentFrame = 0;
float pointSize = 5.0f;
float alpha = 0.2;

std::map<int, std::string> timestampComments;
int currentTimestamp = 1;
char commentBuffer[512] = "";
char searchBuffer[512] = ""; // Buffer for searching annotations
std::vector<std::pair<int, std::string>> filteredComments;
open3d::visualization::Visualizer visualizer;

std::vector<nav_msgs::msg::OccupancyGrid> mapFrames; // For storing map data
std::vector<std::vector<Point3D>> cloudMapFrames;    // For storing point clouds from /rtabmap/cloud_map
std::vector<nav_msgs::msg::Path> mapPaths;           // For storing paths

bool IsDataLoaded = false;

// Function to process frames in the background
void processInBackground(MapBuilder &mapper, const std::vector<std::vector<Point3D>> &pointCloudFrames, const std::vector<IMUData> &imuFrames)
{
    isProcessing = true;
    for (size_t i = 0; i < pointCloudFrames.size(); ++i)
    {
        mapper.processFrame(pointCloudFrames[i], imuFrames[i]);
        progress = static_cast<float>(i + 1) / pointCloudFrames.size();
    }

    // mapper.saveMap("current");
    isProcessing = false;
}

void startBackgroundProcessing(MapBuilder &mapper, const std::vector<std::vector<Point3D>> &pointCloudFrames, const std::vector<IMUData> &imuFrames)
{
    // Launch the background thread
    backgroundThread = std::thread(processInBackground, std::ref(mapper), std::cref(pointCloudFrames), std::cref(imuFrames));
    // for (size_t i = 0; i < pointCloudFrames.size(); ++i)
    // {
    //     mapper.processFrame(pointCloudFrames[i], imuFrames[i]);
    //     progress = static_cast<float>(i + 1) / pointCloudFrames.size();
    // }
    // mapper.saveMap("current");
    // isProcessing = false;
}

// Call this in your main loop to draw the ImGui interface
void drawImGuiInterface()
{
    if (isProcessing)
    {
        ImGui::Text("Processing...");
        ImGui::ProgressBar(progress, ImVec2(0.0f, 0.0f));
    }
    else
    {
        if (ImGui::Button("Start SLAM Processing"))
        {
            MapBuilder mapper;
            // Example usage of the start function
            startBackgroundProcessing(mapper, pointCloudFrames, imuFrames);
        }
    }
}

// Remember to join the thread at the end of the program
void cleanup()
{
    if (backgroundThread.joinable())
    {
        backgroundThread.join();
    }
}

void RunVisualizer()
{
    visualizer.CreateVisualizerWindow("Alpha Shape Mesh");
    visualizer.AddGeometry(mesh);
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    IsVisualizerOpen = false; // Set the flag when visualizer is closed
    applyMesh = false;
}
void ToggleVisualizer()
{
    if (applyMesh && !IsVisualizerOpen)
    {
        // If applyMesh is true and visualizer is not open, create and run it in a new thread
        IsVisualizerOpen = true;
        std::thread visualizerThread(RunVisualizer); // Launch visualizer in a separate thread
        visualizerThread.detach();                   // Detach so it runs independently
    }
    else if (!applyMesh && IsVisualizerOpen)
    {
        // If applyMesh is false and the visualizer is open, close it
        visualizer.DestroyVisualizerWindow();
        IsVisualizerOpen = false;
    }
}
// Function to convert Point3D to Open3D point cloud
std::shared_ptr<open3d::geometry::PointCloud> ConvertToOpen3DPointCloud(const std::vector<Point3D> &points)
{
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    for (const auto &point : points)
    {
        cloud->points_.emplace_back(Eigen::Vector3d(point.x, point.y, point.z));
    }
    return cloud;
}

void createTextureFromImage(const cv::Mat &image)
{
    if (image.empty())
    {
        std::cerr << "Error: Image is empty!" << std::endl;
        return;
    }

    if (imageTexture)
        glDeleteTextures(1, &imageTexture);

    glGenTextures(1, &imageTexture);
    glBindTexture(GL_TEXTURE_2D, imageTexture);

    // Convert image to RGB if it's in BGR format (OpenCV default)
    cv::Mat rgbImage;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
    }
    else
    {
        rgbImage = image; // If already in RGB format, no need for conversion
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgbImage.cols, rgbImage.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgbImage.data);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void renderImage()
{
    if (imageTexture == 0)
    {
        std::cerr << "Error: No texture available!" << std::endl;
        return;
    }

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, imageTexture);

    // Get the dimensions of the image
    int imgWidth = imageFrames[currentFrame].cols;
    int imgHeight = imageFrames[currentFrame].rows;

    // Calculate the aspect ratio of the image
    float aspectRatio = (float)imgWidth / imgHeight;

    // Ensure the aspect ratio is not too large; limit it to fit within the screen
    float maxAspect = 2.0f; // Limit aspect ratio to prevent the image from becoming too large
    aspectRatio = std::min(aspectRatio, maxAspect);

    // Set up the vertices for rendering the image in the right half
    // Adjust the image's size to fit within the right half of the screen
    float left = 0.0f;                // Start at the middle of the screen
    float right = left + aspectRatio; // End at right based on aspect ratio
    float top = 1.0f;                 // Top of the screen
    float bottom = -1.0f;             // Bottom of the screen

    // Draw the image as a textured quad
    glBegin(GL_QUADS);

    // Bottom-left corner (texture coordinates are flipped vertically)
    glTexCoord2f(0.0f, 1.0f); // Flip the Y coordinate
    glVertex2f(left, bottom);

    // Bottom-right corner
    glTexCoord2f(1.0f, 1.0f); // Flip the Y coordinate
    glVertex2f(right, bottom);

    // Top-right corner
    glTexCoord2f(1.0f, 0.0f); // Flip the Y coordinate
    glVertex2f(right, top);

    // Top-left corner
    glTexCoord2f(0.0f, 0.0f); // Flip the Y coordinate
    glVertex2f(left, top);

    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}
float getMaxIntensity(const std::vector<Point3D> &points)
{
    float maxIntensity = 0.0f;
    for (const auto &point : points)
    {
        maxIntensity = std::max(maxIntensity, point.intensity);
    }
    return maxIntensity;
}

void renderPointCloud(const std::vector<Point3D> &points, float pointSize)
{
    if (!showprocessSLAM)
    {
        if (applyMesh)
        {
            // Render mesh
            glBegin(GL_TRIANGLES);

            // Access triangles and vertices using getter functions
            const auto &triangles = mesh->triangles_;
            const auto &vertices = mesh->vertices_;
            const auto &vertex_colors = mesh->vertex_colors_;

            // Loop through all triangles and render them
            for (const auto &triangle : triangles)
            {
                // Loop over the vertices of each triangle
                for (int i = 0; i < 3; ++i)
                {
                    int vertexIndex = triangle[i];
                    const auto &vertex = vertices[vertexIndex];
                    const auto &color = vertex_colors.empty() ? Eigen::Vector3d(1.0, 1.0, 1.0) : vertex_colors[vertexIndex]; // Default to white if no color is assigned

                    glColor3f(color(0), color(1), color(2));     // Set vertex color
                    glVertex3f(vertex(0), vertex(1), vertex(2)); // Render vertex position
                }
            }

            glEnd();
        }
        else
        {
            // Render points
            glPointSize(pointSize);
            glBegin(GL_POINTS);
            float maxIntensity = getMaxIntensity(points);

            for (const auto &point : points)
            {
                if (point.clicked || showClusters || showPlane)
                {
                    glColor3f(point.r, point.g, point.b); // Color for clicked points
                }
                else
                {
                    float intensityNormalized = std::min(1.0f, std::max(0.0f, point.intensity / maxIntensity)); // Normalize intensity
                    float red = intensityNormalized;
                    float green = 0.0f;
                    float blue = 1.0f - intensityNormalized;
                    glColor3f(red, green, blue); // Color based on intensity
                }
                glVertex3f(point.x, point.y, point.z); // Render the point
            }
            glEnd();
        }
    }
    // Only render map point cloud and path if showProcessSLAM is true
    if (showprocessSLAM)
    {

        if (applyMesh)
        {
            // Render mesh
            glBegin(GL_TRIANGLES);

            // Access triangles and vertices using getter functions
            const auto &triangles = mesh->triangles_;
            const auto &vertices = mesh->vertices_;
            const auto &vertex_colors = mesh->vertex_colors_;

            // Loop through all triangles and render them
            for (const auto &triangle : triangles)
            {
                // Loop over the vertices of each triangle
                for (int i = 0; i < 3; ++i)
                {
                    int vertexIndex = triangle[i];
                    const auto &vertex = vertices[vertexIndex];
                    const auto &color = vertex_colors.empty() ? Eigen::Vector3d(1.0, 1.0, 1.0) : vertex_colors[vertexIndex]; // Default to white if no color is assigned

                    glColor3f(color(0), color(1), color(2));     // Set vertex color
                    glVertex3f(vertex(0), vertex(1), vertex(2)); // Render vertex position
                }
            }

            glEnd();
        }

        else
        {

            int step = pointCloudFrames.size() / cloudMapFrames.size();
            int newFrame = std::min((currentFrame / step), static_cast<int>(cloudMapFrames.size() - 1));

            // Render map point cloud

            glPointSize(3.0f); // Set point size for map
            glBegin(GL_POINTS);

            for (const auto &point : cloudMapFrames[newFrame])
            {
                // Color the map points based on some property (e.g., intensity or fixed color)
                glColor3f(0.0f, 1.0f, 0.0f);           // Green for map point cloud
                glVertex3f(point.x, point.y, point.z); // Render the map point
            }
            glEnd();

            // Render map path

            glLineWidth(2.0f); // Set line width for path
            glBegin(GL_LINE_STRIP);
            step = pointCloudFrames.size() / mapPaths.size();
            newFrame = std::min((currentFrame / step), static_cast<int>(mapPaths.size() - 1));
            for (const auto &pathPoint : mapPaths[newFrame].poses)
            {
                glColor3f(0.0f, 0.0f, 1.0f);                                                                 // Blue for map path
                glVertex3f(pathPoint.pose.position.x, pathPoint.pose.position.y, pathPoint.pose.position.z); // Render path point
            }

            glEnd();
        }
    }
}

void parsePointCloudDataFromBag(const std::string &bagFile, const std::string &pointCloudTopic, const std::string &imageTopic, const std::string &imuTopic, const std::string &mapTopic, const std::string &cloudMapTopic, const std::string &mapPathTopic)
{
    rosbag2_cpp::Reader reader;
    reader.open({bagFile, "sqlite3"});

    int pointCloudCount = 0;
    int imageCount = 0;
    int imuCount = 0;
    int mapCount = 0;
    int cloudMapCount = 0;
    int mapPathCount = 0;

    cv::Mat lastImage; // To store the last valid image for ZOH
    std::vector<IMUData> fullImuFrames;
    while (reader.has_next())
    {
        auto msg = reader.read_next();

        if (msg->topic_name == pointCloudTopic)
        {
            // Process PointCloud2 message (same as before)
            rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
            rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serializer;
            auto pointCloudMsg = std::make_shared<sensor_msgs::msg::PointCloud2>();
            serializer.deserialize_message(&serialized_msg, pointCloudMsg.get());

            // Process point cloud
            std::vector<Point3D> points;
            sensor_msgs::PointCloud2Iterator<float> iter_x(*pointCloudMsg, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(*pointCloudMsg, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(*pointCloudMsg, "z");
            bool hasIntensity = pointCloudMsg->fields.size() > 3;
            sensor_msgs::PointCloud2Iterator<float> iter_intensity(*pointCloudMsg, "i");

            for (size_t i = 0; i < pointCloudMsg->width; ++i)
            {
                Point3D point{*iter_x, *iter_y, *iter_z, hasIntensity ? *iter_intensity : 1.0f};
                points.push_back(point);
                ++iter_x;
                ++iter_y;
                ++iter_z;
                if (hasIntensity)
                    ++iter_intensity;
            }

            pointCloudFrames.push_back(points);
            pointCloudTimestamps.push_back(pointCloudCount);
            pointCloudCount++;

            // Add the last held image and IMU data to maintain alignment
            if (!lastImage.empty())
            {
                imageFrames.push_back(lastImage.clone());
            }
            else
            {
                imageFrames.push_back(cv::Mat()); // Push an empty image if none received yet
            }
        }
        else if (msg->topic_name == imageTopic)
        {
            // Process Image message (same as before)
            rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
            rclcpp::Serialization<sensor_msgs::msg::Image> serializer;
            auto imageMsg = std::make_shared<sensor_msgs::msg::Image>();
            serializer.deserialize_message(&serialized_msg, imageMsg.get());

            // Convert ROS 2 Image message to OpenCV Mat
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(imageMsg, "bgr8");
            lastImage = cv_ptr->image.clone(); // Update the last held image for ZOH
            imageCount++;
        }
        else if (msg->topic_name == imuTopic)
        {
            // Process IMU message (same as before)
            rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
            rclcpp::Serialization<sensor_msgs::msg::Imu> serializer;
            auto imuMsg = std::make_shared<sensor_msgs::msg::Imu>();
            serializer.deserialize_message(&serialized_msg, imuMsg.get());

            // Process IMU message into a custom IMUData structure
            IMUData imuData;
            imuData.orientation = {static_cast<float>(imuMsg->orientation.x),
                                   static_cast<float>(imuMsg->orientation.y),
                                   static_cast<float>(imuMsg->orientation.z),
                                   static_cast<float>(imuMsg->orientation.w)};
            imuData.angularVelocity = {static_cast<float>(imuMsg->angular_velocity.x),
                                       static_cast<float>(imuMsg->angular_velocity.y),
                                       static_cast<float>(imuMsg->angular_velocity.z)};
            imuData.linearAcceleration = {static_cast<float>(imuMsg->linear_acceleration.x),
                                          static_cast<float>(imuMsg->linear_acceleration.y),
                                          static_cast<float>(imuMsg->linear_acceleration.z)};
            fullImuFrames.push_back(imuData);
            imuCount++;
        }
        else if (msg->topic_name == mapTopic)
        {
            // Process OccupancyGrid message
            rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
            rclcpp::Serialization<nav_msgs::msg::OccupancyGrid> serializer;
            auto mapMsg = std::make_shared<nav_msgs::msg::OccupancyGrid>();
            serializer.deserialize_message(&serialized_msg, mapMsg.get());

            // Store the map data or process it as needed
            mapFrames.push_back(*mapMsg);
            mapCount++;
        }
        else if (msg->topic_name == cloudMapTopic)
        {
            // Process PointCloud2 message from the cloud map
            rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
            rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serializer;
            auto cloudMapMsg = std::make_shared<sensor_msgs::msg::PointCloud2>();
            serializer.deserialize_message(&serialized_msg, cloudMapMsg.get());

            // Process point cloud (same as before)
            std::vector<Point3D> points;
            sensor_msgs::PointCloud2Iterator<float> iter_x(*cloudMapMsg, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(*cloudMapMsg, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(*cloudMapMsg, "z");
            bool hasIntensity = cloudMapMsg->fields.size() > 3;
            // sensor_msgs::PointCloud2Iterator<float> iter_intensity(*cloudMapMsg, "i");

            for (size_t i = 0; i < cloudMapMsg->width; ++i)
            {
                Point3D point{*iter_x, *iter_y, *iter_z, hasIntensity ? 1 : 1.0f};
                points.push_back(point);
                ++iter_x;
                ++iter_y;
                ++iter_z;
            }

            cloudMapFrames.push_back(points);
            cloudMapCount++;
        }
        else if (msg->topic_name == mapPathTopic)
        {
            // Process Path message
            rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
            rclcpp::Serialization<nav_msgs::msg::Path> serializer;
            auto mapPathMsg = std::make_shared<nav_msgs::msg::Path>();
            serializer.deserialize_message(&serialized_msg, mapPathMsg.get());

            // Store or process the path data
            mapPaths.push_back(*mapPathMsg);
            mapPathCount++;
        }
    }

    // Downsampling IMU data to match the point cloud count
    // size_t step = imuCount / pointCloudCount;
    // for (size_t i = 0; i < imuCount; i += step)
    // {
    //     imuFrames.push_back(fullImuFrames[i]);
    // }
}

void renderAxes()
{
    glBegin(GL_LINES);
    // X-axis (red)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(10.0f, 0.0f, 0.0f);

    // Y-axis (green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 10.0f, 0.0f);

    // Z-axis (blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 10.0f);
    glEnd();
}

// Function to find the nearest point in the point cloud
int findClosestPoint(double mouseX, double mouseY, const std::vector<Point3D> &points,
                     int screenWidth, int screenHeight, const GLdouble *modelMatrix,
                     const GLdouble *projMatrix, const GLint *viewport)
{
    int closestIndex = -1;
    double minDistance = 0.1; // Adjust threshold based on your needs

    for (size_t i = 0; i < points.size(); ++i)
    {
        double screenX, screenY, screenZ;

        // Project 3D point to screen space using double precision matrices
        gluProject(points[i].x, points[i].y, points[i].z,
                   modelMatrix, projMatrix, viewport,
                   &screenX, &screenY, &screenZ);

        // Convert OpenGL screen coordinates to window coordinates
        screenY = screenHeight - screenY; // Invert the Y-axis to match window coordinates

        // Compute the distance between mouse and point in screen space
        double dist2D = std::sqrt((screenX - mouseX) * (screenX - mouseX) +
                                  (screenY - mouseY) * (screenY - mouseY));

        // Find the closest point within the threshold
        if (dist2D < minDistance)
        {
            minDistance = dist2D;
            closestIndex = i;
        }
    }

    return closestIndex;
}

// Mouse button callback for drag and click functionality
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            // Check if the click was a mouse click or if we are starting a drag
            isDragging = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);

            // Get the current mouse position
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);

            // Fetch OpenGL transformation matrices and viewport
            float modelMatrixFloat[16], projMatrixFloat[16];
            int viewport[4];
            glGetFloatv(GL_MODELVIEW_MATRIX, modelMatrixFloat);
            glGetFloatv(GL_PROJECTION_MATRIX, projMatrixFloat);
            glGetIntegerv(GL_VIEWPORT, viewport);

            // Convert matrices to double
            GLdouble modelMatrix[16], projMatrix[16];
            for (int i = 0; i < 16; ++i)
            {
                modelMatrix[i] = static_cast<GLdouble>(modelMatrixFloat[i]);
                projMatrix[i] = static_cast<GLdouble>(projMatrixFloat[i]);
            }

            // Ensure we have a valid point cloud
            if (pointCloudFrames.empty())
            {
                std::cerr << "Point cloud is empty. Click ignored." << std::endl;
                return;
            }

            // Find the closest point to the mouse click
            // int closestIndex = findClosestPoint(mouseX, mouseY, pointCloudFrames[currentFrame],
            //                                     viewport[2], viewport[3],
            //                                     modelMatrix, projMatrix, viewport);

            // If a valid point was found
            // if (closestIndex != -1)
            // {
            //     Point3D &clickedPoint = pointCloudFrames[currentFrame][closestIndex];

            //     // Toggle the clicked state
            //     clickedPoint.clicked = !clickedPoint.clicked;

            //     if (clickedPoint.clicked)
            //     {
            //         // Change the color to white
            //         clickedPoint.r = 1.0f;
            //         clickedPoint.g = 1.0f;
            //         clickedPoint.b = 1.0f;
            //     }
            //     else
            //     {
            //         // Reset color based on intensity
            //         float intensityNormalized = std::clamp(clickedPoint.intensity, 0.0f, 1.0f);
            //         clickedPoint.r = intensityNormalized;
            //         clickedPoint.g = 0.0f;
            //         clickedPoint.b = 1.0f - intensityNormalized;
            //     }
            // }
            // else
            // {
            //     std::cout << "No point found near the click location." << std::endl;
            // }
        }
        else if (action == GLFW_RELEASE)
        {
            isDragging = false;
        }
    }
}

// Cursor position callback for camera rotation
void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    if (isDragging)
    {
        double dx = xpos - lastMouseX;
        double dy = ypos - lastMouseY;
        cameraAngleX += static_cast<float>(dx) * 0.1f;
        cameraAngleY += static_cast<float>(dy) * 0.1f;
        lastMouseX = xpos;
        lastMouseY = ypos;
    }
}

// Scroll callback for zoom
void scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    zoomLevel -= static_cast<float>(yoffset);
    if (zoomLevel < 2.0f)
        zoomLevel = 2.0f; // Minimum zoom level
    if (zoomLevel > 50.0f)
        zoomLevel = 50.0f; // Maximum zoom level
}
// Helper function to convert a string to lowercase
std::string toLower(const std::string &str)
{
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    return lowerStr;
}

// Helper function to filter comments based on search (case insensitive)
void FilterComments()
{
    filteredComments.clear();
    std::string searchText = toLower(std::string(searchBuffer)); // Convert search query to lowercase
    for (const auto &entry : timestampComments)
    {
        std::string timestampStr = toLower(std::to_string(entry.first)); // Convert timestamp to lowercase
        std::string commentStr = toLower(entry.second);                  // Convert comment to lowercase

        // Perform case-insensitive search
        if (timestampStr.find(searchText) != std::string::npos || commentStr.find(searchText) != std::string::npos)
        {
            filteredComments.push_back(entry);
        }
    }
}
// Function to calculate the average slope between neighboring points
float calculateSlope(const std::vector<Point3D> &points)
{
    float totalSlope = 0.0f;
    int count = 0;

    for (size_t i = 0; i < points.size() - 1; ++i)
    {
        float dx = points[i + 1].x - points[i].x;
        float dy = points[i + 1].y - points[i].y;
        float dz = points[i + 1].z - points[i].z;

        float distance = std::sqrt(dx * dx + dy * dy);          // Horizontal distance
        float slope = std::atan(dz / distance) * 180.0f / M_PI; // Convert slope to degrees

        totalSlope += slope;
        count++;
    }

    return count > 0 ? totalSlope / count : 0.0f; // Return average slope
}
// Function to calculate intensity histogram
void calculateIntensityHistogram(const std::vector<Point3D> &points, std::vector<float> &histogram, int numBins = 10)
{
    // Initialize histogram bins
    histogram.resize(numBins, 0.0f);

    // Find min and max intensity values to normalize
    float minIntensity = std::numeric_limits<float>::max();
    float maxIntensity = std::numeric_limits<float>::lowest();

    for (const auto &point : points)
    {
        minIntensity = std::min(minIntensity, point.intensity);
        maxIntensity = std::max(maxIntensity, point.intensity);
    }

    // Normalize intensities and populate the histogram
    for (const auto &point : points)
    {
        float normalizedIntensity = (point.intensity - minIntensity) / (maxIntensity - minIntensity);
        int binIndex = static_cast<int>(normalizedIntensity * (numBins - 1)); // Bin index between 0 and numBins-1
        histogram[binIndex]++;
    }
}
// Function to calculate the maximum and minimum elevation
void calculateElevationMetrics(const std::vector<Point3D> &points, float &maxElevation, float &minElevation)
{
    maxElevation = std::numeric_limits<float>::lowest();
    minElevation = std::numeric_limits<float>::max();

    for (const auto &point : points)
    {
        if (point.z > maxElevation)
            maxElevation = point.z;
        if (point.z < minElevation)
            minElevation = point.z;
    }
}
// Function to calculate point density
float calculatePointDensity(const std::vector<Point3D> &points, float areaSize)
{
    return points.size() / areaSize; // Points per square meter
}

void showMetricsInUI()
{
    ImGui::Begin("LiDAR Metrics");
    // Get the current point cloud
    if (bagFilePath != "")
    {
        const std::vector<Point3D> &points = pointCloudFrames[currentFrame];

        // Calculate metrics
        float pointDensity = calculatePointDensity(points, 100.0f);
        float maxElevation, minElevation;
        calculateElevationMetrics(points, maxElevation, minElevation);

        std::vector<float> intensityHistogram;
        calculateIntensityHistogram(points, intensityHistogram);

        float slopeAngle = calculateSlope(points);

        // Display metrics in ImGui UI
        ImGui::Text("Point Density: %.2f points/m^2", pointDensity);
        ImGui::Text("Max Elevation: %.2f", maxElevation);
        ImGui::Text("Min Elevation: %.2f", minElevation);

        // Display intensity histogram
        ImGui::Text("Intensity Histogram:");
        ImGui::PlotHistogram("##intensityHistogram", intensityHistogram.data(), intensityHistogram.size(), 0, nullptr, 0.0f, 100.0f, ImVec2(200, 100));

        // Display slope
        ImGui::Text("Slope: %.2f degrees", slopeAngle);

        // open 3d metrics
        ImGui::Text("Number of Clusters: %.2d", max_label + 1);
        ImGui::SameLine();
        if (ImGui::Checkbox("Show Clusters", &showClusters))
            IsDataLoaded = false;

        if (ImGui::Checkbox("Show Plane", &showPlane))
            IsDataLoaded = false;

        if (ImGui::Checkbox("Apply mesh", &applyMesh))
            IsDataLoaded = false;

        ImGui::SameLine();
        if (ImGui::InputFloat("Alpha Value", &alpha, 0.1f, 1.0f, "%.2f"))
            IsDataLoaded = false;

        ImGui::Separator();
        // drawImGuiInterface();

        if (ImGui::Checkbox("Show Processed SLAM", &showprocessSLAM))
            IsDataLoaded = false;
    }
    ImGui::End();
}

int main(int argc, char **argv)
{
    // Initialize ROS 2
    rclcpp::init(argc, argv);
    py::scoped_interpreter guard{};
    // python test
    try
    {
        py::module_::import("sys").attr("path").attr("append")("/home/sujee/sick_gui/scripts");

        // Import your Python module and call a function
        auto process_image = py::module_::import("open3dml_demo").attr("process_image");
        std::string processed_image = process_image("input_image.png").cast<std::string>();
        std::cout << "Processed image saved at: " << processed_image << std::endl;
    }
    catch (py::error_already_set &e)
    {
        std::cerr << "Python error: " << e.what() << std::endl;
    }
    // Initialize GLFW and ImGui
    if (!glfwInit())
        return -1;
    GLFWwindow *window = glfwCreateWindow(2160, 1920, "Sick Mine Viewer", nullptr, nullptr);
    if (!window)
        return -1;
    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetScrollCallback(window, scrollCallback);

    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Slider parameters for zipping through timestamps
    float sliderValue = 0.0f;
    if (!pointCloudFrames.empty())
    {
        sliderValue = static_cast<float>(currentFrame);
    }

    while (!glfwWindowShouldClose(window))
    {
        // Poll events and start new ImGui frame
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("LiDAR Controls");

        if (ImGui::Button("Open Bag File"))
        {
            // Open the file dialog for .db3 files
            IGFD::FileDialogConfig config;
            config.path = "/home/sujee/camera_plus_lidar/test2/test2";
            config.fileName = "";                     // No default file name
            config.filePathName = "";                 // No pre-set file path
            config.countSelectionMax = 1;             // Allow selecting only one file
            config.flags = ImGuiFileDialogFlags_None; // Default dialog flags
            config.sidePaneWidth = 250.0f;            // Default side pane width

            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseBagFileDlg",  // Dialog key
                "Select a Bag File", // Dialog title
                ".db3",              // File extension filter
                config               // FileDialogConfig object
            );
        }

        // Handle the file dialog
        if (ImGuiFileDialog::Instance()->Display("ChooseBagFileDlg"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                bagFilePath = ImGuiFileDialog::Instance()->GetFilePathName();

                // Reload the data from the new file
                // /rtabmap/map Type: nav_msgs/msg/OccupancyGrid
                // /rtabmap/cloud_map Type:sensor_msgs/msg/PointCloud2
                // /rtabmap/mapPath Type: nav_msgs/msg/Path

                parsePointCloudDataFromBag(bagFilePath, "/cloud_unstructured_fullframe", "/camera/image_raw", "/sick_scansegment_xd/imu", "/rtabmap/map", "/rtabmap/cloud_map", "/rtabmap/mapPath");

                std::cout << "Size of imageFrames: " << imageFrames.size() << std::endl;
                std::cout << "Size of imuFrames: " << imuFrames.size() << std::endl;
                sliderValue = 0.0f; // Reset slider
                currentFrame = 0;   // Reset frame index

                // reset other fields here as well for filtered comments and reset the size of the timestamps
            }
            ImGuiFileDialog::Instance()->Close();
        }
        ImGui::Separator();

        if (!pointCloudFrames.empty())
        {
            if (ImGui::Button("Reset View"))
            {
                cameraAngleX = 0.0f;
                cameraAngleY = 0.0f;
                zoomLevel = 15.0f;
            }

            ImGui::Text("3D Point Cloud and Camera Image Viewer");
            if (ImGui::SliderFloat("Frame", &sliderValue, 0.0f, static_cast<float>(pointCloudFrames.size() - 1), "%.0f"))
            {
                ImGui::Separator(); // Optional: Adds a separator line for visual clarity
                ImGui::Text("LiDAR Point Cloud Section");
                ImGui::Separator(); // Another separator for the heading section
                currentFrame = static_cast<int>(sliderValue);
                IsDataLoaded = false;
            }
            if (!IsDataLoaded)
            {
                IsDataLoaded = true;
                if (!showprocessSLAM)
                {
                    auto cloud = ConvertToOpen3DPointCloud(pointCloudFrames[currentFrame]);
                    // voxel grid downsampling
                    auto downpcd = cloud->VoxelDownSample(0.05);

                    // Perform DBSCAN clustering
                    double eps = 0.5;   // Distance threshold for clustering
                    int min_points = 2; // Minimum points to form a cluster
                    bool print_progress = false;

                    std::vector<int> cluster_labels = downpcd->ClusterDBSCAN(eps, min_points, print_progress);

                    // Output clustering results
                    max_label = *std::max_element(cluster_labels.begin(), cluster_labels.end());
                    if (showClusters)
                    {
                        // set the cluster color when showCluster is clicked would show the result on the ui
                        for (size_t i = 0; i < pointCloudFrames[currentFrame].size(); ++i)
                        {
                            if (cluster_labels[i] == -1)
                            {
                                // Noise points: Assign a specific color (e.g., red)
                                pointCloudFrames[currentFrame][i].r = 1.0f;
                                pointCloudFrames[currentFrame][i].g = 0.0f;
                                pointCloudFrames[currentFrame][i].b = 0.0f;
                            }
                            else
                            {
                                // Clustered points: Assign a color based on the cluster label
                                double hue = static_cast<double>(cluster_labels[i]) / (max_label + 1);
                                // Example color scheme: Map hue to RGB
                                Eigen::Vector3d rgb = Eigen::Vector3d(hue, 0.5, 0.5); // Adjust this mapping as needed
                                pointCloudFrames[currentFrame][i].r = static_cast<float>(rgb(0));
                                pointCloudFrames[currentFrame][i].g = static_cast<float>(rgb(1));
                                pointCloudFrames[currentFrame][i].b = static_cast<float>(rgb(2));
                            }
                        }
                    }
                    // Plane segmentation parameters
                    double distance_threshold = 0.01; // Maximum distance to the plane
                    int ransac_n = 3;                 // Number of points to estimate a plane
                    int num_iterations = 1000;        // Number of iterations for RANSAC

                    // Segment the largest plane
                    std::tuple<Eigen::Vector4d, std::vector<size_t>> result =
                        downpcd->SegmentPlane(distance_threshold, ransac_n, num_iterations);

                    Eigen::Vector4d plane_model = std::get<0>(result);
                    std::vector<size_t> inliers = std::get<1>(result);

                    // Plane equation coefficients
                    double a = plane_model[0];
                    double b = plane_model[1];
                    double c = plane_model[2];
                    double d = plane_model[3];

                    // // Print the plane equation
                    // std::cout << "Plane equation: "
                    //           << a << "x + "
                    //           << b << "y + "
                    //           << c << "z + "
                    //           << d << " = 0"
                    //           << std::endl;

                    if (showPlane)
                    {
                        for (size_t i = 0; i < pointCloudFrames[currentFrame].size(); ++i)
                        {
                            bool is_inlier = std::find(inliers.begin(), inliers.end(), i) != inliers.end();
                            if (is_inlier)
                            {
                                // Color the inliers as red
                                pointCloudFrames[currentFrame][i].r = 1.0f;
                                pointCloudFrames[currentFrame][i].g = 0.0f;
                                pointCloudFrames[currentFrame][i].b = 0.0f;
                            }
                            else
                            {
                                // Optional: Color outliers differently if needed, here setting them as blue
                                pointCloudFrames[currentFrame][i].r = 0.0f;
                                pointCloudFrames[currentFrame][i].g = 0.0f;
                                pointCloudFrames[currentFrame][i].b = 1.0f;
                            }
                        }
                    }

                    // Create mesh from the point cloud using the alpha shape
                    mesh = open3d::geometry::TriangleMesh::CreateFromPointCloudAlphaShape(*downpcd, alpha);

                    // Compute vertex normals
                    mesh->ComputeVertexNormals();
                }
                else
                {
                    int step = pointCloudFrames.size() / cloudMapFrames.size();
                    int newFrame = std::min((currentFrame / step), static_cast<int>(cloudMapFrames.size() - 1));
                    auto cloud = ConvertToOpen3DPointCloud(cloudMapFrames[newFrame]);
                    // voxel grid downsampling
                    auto downpcd = cloud->VoxelDownSample(0.05);

                    // Perform DBSCAN clustering
                    double eps = 0.5;   // Distance threshold for clustering
                    int min_points = 2; // Minimum points to form a cluster
                    bool print_progress = false;

                    std::vector<int> cluster_labels = downpcd->ClusterDBSCAN(eps, min_points, print_progress);

                    // Output clustering results
                    max_label = *std::max_element(cluster_labels.begin(), cluster_labels.end());
                    if (showClusters)
                    {
                        // set the cluster color when showCluster is clicked would show the result on the ui
                        for (size_t i = 0; i < cloudMapFrames[newFrame].size(); ++i)
                        {
                            if (cluster_labels[i] == -1)
                            {
                                // Noise points: Assign a specific color (e.g., red)
                                cloudMapFrames[newFrame][i].r = 1.0f;
                                cloudMapFrames[newFrame][i].g = 0.0f;
                                cloudMapFrames[newFrame][i].b = 0.0f;
                            }
                            else
                            {
                                // Clustered points: Assign a color based on the cluster label
                                double hue = static_cast<double>(cluster_labels[i]) / (max_label + 1);
                                // Example color scheme: Map hue to RGB
                                Eigen::Vector3d rgb = Eigen::Vector3d(hue, 0.5, 0.5); // Adjust this mapping as needed
                                cloudMapFrames[newFrame][i].r = static_cast<float>(rgb(0));
                                cloudMapFrames[newFrame][i].g = static_cast<float>(rgb(1));
                                cloudMapFrames[newFrame][i].b = static_cast<float>(rgb(2));
                            }
                        }
                    }
                    // Plane segmentation parameters
                    double distance_threshold = 0.01; // Maximum distance to the plane
                    int ransac_n = 3;                 // Number of points to estimate a plane
                    int num_iterations = 1000;        // Number of iterations for RANSAC

                    // Segment the largest plane
                    std::tuple<Eigen::Vector4d, std::vector<size_t>> result =
                        downpcd->SegmentPlane(distance_threshold, ransac_n, num_iterations);

                    Eigen::Vector4d plane_model = std::get<0>(result);
                    std::vector<size_t> inliers = std::get<1>(result);

                    // Plane equation coefficients
                    double a = plane_model[0];
                    double b = plane_model[1];
                    double c = plane_model[2];
                    double d = plane_model[3];

                    // // Print the plane equation
                    // std::cout << "Plane equation: "
                    //           << a << "x + "
                    //           << b << "y + "
                    //           << c << "z + "
                    //           << d << " = 0"
                    //           << std::endl;

                    if (showPlane)
                    {
                        for (size_t i = 0; i < cloudMapFrames[newFrame].size(); ++i)
                        {
                            bool is_inlier = std::find(inliers.begin(), inliers.end(), i) != inliers.end();
                            if (is_inlier)
                            {
                                // Color the inliers as red
                                cloudMapFrames[newFrame][i].r = 1.0f;
                                cloudMapFrames[newFrame][i].g = 0.0f;
                                cloudMapFrames[newFrame][i].b = 0.0f;
                            }
                            else
                            {
                                // Optional: Color outliers differently if needed, here setting them as blue
                                cloudMapFrames[newFrame][i].r = 0.0f;
                                cloudMapFrames[newFrame][i].g = 0.0f;
                                cloudMapFrames[newFrame][i].b = 1.0f;
                            }
                        }
                    }

                    // Create mesh from the point cloud using the alpha shape
                    mesh = open3d::geometry::TriangleMesh::CreateFromPointCloudAlphaShape(*downpcd, alpha);

                    // Compute vertex normals
                    mesh->ComputeVertexNormals();
                }
            }
            // Compute normals
            // downpcd->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.01, 30));
            // downpcd->OrientNormalsConsistentTangentPlane(30);
            // std::vector<double> radii = {0.005, 0.01, 0.02, 0.04};

            // Create mesh using ball pivoting
            // mesh = open3d::geometry::TriangleMesh::CreateFromPointCloudBallPivoting(*downpcd, radii);
            // std::vector<double> densities;
            // std::tie(mesh, densities) = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(*downpcd, 9);

            // ToggleVisualizer();

            if (currentFrame >= 0 && currentFrame < imageFrames.size())
            {
                createTextureFromImage(imageFrames[currentFrame]);
            }
            ImGui::Text("Timestamp: %.2f", pointCloudTimestamps[currentFrame]); // Display the timestamp
            ImGui::Separator();

            // Load or clear comment buffer for the current timestamp
            if (timestampComments.find(currentFrame) != timestampComments.end())
            {
                strncpy(commentBuffer, timestampComments[currentFrame].c_str(), sizeof(commentBuffer));
            }
            else
            {
                memset(commentBuffer, 0, sizeof(commentBuffer));
            }

            // Text area for editing comments
            ImGui::Text("Edit the comment for this timestamp:");
            if (ImGui::InputTextMultiline("##comment", commentBuffer, sizeof(commentBuffer), ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 4)))
            {
                timestampComments[currentFrame] = std::string(commentBuffer);
            }

            ImGui::Separator();

            // Search bar
            ImGui::Begin("Search Annotations");
            ImGui::Text("Search:");
            ImGui::InputText("##search", searchBuffer, sizeof(searchBuffer));

            FilterComments();
            ImGui::Text("Comment History");
            // Display annotations in a table
            if (ImGui::BeginTable("AnnotationsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable))
            {
                ImGui::TableSetupColumn("Timestamp", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Comment", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                for (const auto &entry : filteredComments)
                {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);

                    // Add clickable timestamp
                    if (ImGui::Selectable(std::to_string(entry.first).c_str(), false, ImGuiSelectableFlags_SpanAllColumns))
                    {
                        currentFrame = entry.first;
                        sliderValue = static_cast<float>(currentFrame); // Sync slider with clicked timestamp
                    }

                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextWrapped("%s", entry.second.c_str());
                }
                ImGui::EndTable();
            }
            ImGui::End();
        }
        ImGui::End();
        showMetricsInUI();

        // OpenGL rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        // Left viewport: Point Cloud
        glViewport(0, 0, 1080, 1920);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        float aspectRatio = 1080.0f / 1920.0f;
        float top = tan(45.0f * 3.14159f / 360.0f) * 0.1f;
        float right = top * aspectRatio;
        glFrustum(-right, right, -top, top, 0.1f, 100.0f);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -zoomLevel);
        glRotatef(cameraAngleY, 1.0f, 0.0f, 0.0f);
        glRotatef(cameraAngleX, 0.0f, 1.0f, 0.0f);

        // Heading for LiDAR (Left Section)
        ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_Always);   // Positioning the heading in the left section
        ImGui::SetNextWindowSize(ImVec2(200, 50), ImGuiCond_Always); // Size of the heading area
        ImGui::Begin("LiDAR Heading", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        ImGui::Text("LiDAR Point Cloud"); // Heading for LiDAR
        ImGui::End();

        if (currentFrame >= 0 && currentFrame < pointCloudFrames.size())
        {
            renderPointCloud(pointCloudFrames[currentFrame], pointSize);
        }
        renderAxes();

        // Image Rendering
        glViewport(600, 900, 1000, 500);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Reset OpenGL states for image rendering
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        // Heading for Camera (Right Section)
        ImGui::SetNextWindowPos(ImVec2(1100, 20), ImGuiCond_Always); // Positioning the heading in the right section
        ImGui::SetNextWindowSize(ImVec2(200, 50), ImGuiCond_Always); // Size of the heading area
        ImGui::Begin("Camera Heading", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        ImGui::Text("Camera Image"); // Heading for Camera
        ImGui::End();
        // Render the image
        if (!imageFrames.empty() && currentFrame >= 0 && currentFrame < imageFrames.size())
        {
            renderImage();
        }

        glEnable(GL_DEPTH_TEST);

        // Render ImGui on top
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    cleanup();

    rclcpp::shutdown();
    return 0;
}