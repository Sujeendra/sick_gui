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
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <fstream>
#include <implot.h>

using namespace rtabmap;
namespace py = pybind11;

// LiDAR data structure
struct Point3D
{
    float x, y, z, intensity;
    float r, g, b; // Color for each point
    bool clicked;  // Flag to check if the point has been clicked

    Point3D(float x, float y, float z, float intensity)
        : x(x), y(y), z(z), intensity(intensity), r(1.0f), g(1.0f), b(1.0f), clicked(false) {}
};
struct IMUData
{
    Eigen::Quaternionf orientation;
    Eigen::Vector3f angularVelocity;
    Eigen::Vector3f linearAcceleration;
};
cv::dnn::Net net;
std::vector<std::string> classNames;
float progressValues[9];

// Function to load the YOLOv3 model
void loadYOLOModel(cv::dnn::Net &net, std::vector<std::string> &classNames)
{
    // Paths to the model files
    std::string modelWeights = "/home/sujee/sick_gui/model/yolov3.weights";
    std::string modelConfig = "/home/sujee/sick_gui/model/yolov3.cfg";
    std::string classNamesFile = "/home/sujee/sick_gui/model/coco.names";

    // Load the model
    net = cv::dnn::readNet(modelWeights, modelConfig);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Load class names
    std::ifstream classNamesStream(classNamesFile);
    std::string className;
    while (std::getline(classNamesStream, className))
    {
        classNames.push_back(className);
    }
}
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
        std::cout << "Transform Matrix: \n"
                  << transform << std::endl;

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
bool projectLidar = false;
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
static char userInput[256] = "";             // Buffer to store user input
static std::vector<std::string> chatHistory; // Store chat history

// Function to render the ImGui interface
void RenderImGuiInterface()
{
    // Start a new ImGui window
    ImGui::Begin("Mesh Export");

    // Button to open the file dialog
    if (ImGui::Button("Export Mesh"))
    {
        IGFD::FileDialogConfig config;
        config.path = "/home/sujee/stl";
        config.fileName = "";                     // No default file name
        config.filePathName = "";                 // No pre-set file path
        config.countSelectionMax = 1;             // Allow selecting only one file
        config.flags = ImGuiFileDialogFlags_None; // Default dialog flags
        config.sidePaneWidth = 250.0f;            // Default side pane width
        // Open the file dialog
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlg", "Choose File", ".ply", config);
    }

    // Display the file dialog if it's open
    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlg"))
    {
        // If the user selects a file
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            // Get the file path and name
            std::string file_path = ImGuiFileDialog::Instance()->GetFilePathName();

            // Export the mesh to the selected file
            if (open3d::io::WriteTriangleMesh(file_path, *mesh))
            {
                // Success message
                ImGui::OpenPopup("Export Success");
            }
            else
            {
                // Error message
                ImGui::OpenPopup("Export Failed");
            }
        }

        // Close the file dialog
        ImGuiFileDialog::Instance()->Close();
    }

    // Popup for success message
    if (ImGui::BeginPopupModal("Export Success", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("Mesh exported successfully!");
        if (ImGui::Button("OK"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Popup for error message
    if (ImGui::BeginPopupModal("Export Failed", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("Failed to export mesh!");
        if (ImGui::Button("OK"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // End the ImGui window
    ImGui::End();
}
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
void detectObjectsAndLog(
    const std::string &outputTextFile)
{
    // Open the output text file for writing
    std::ofstream outFile(outputTextFile);
    if (!outFile.is_open())
    {
        std::cerr << "Error: Unable to open output file: " << outputTextFile << std::endl;
        return;
    }

    // Iterate over each frame
    for (size_t i = 0; i < imageFrames.size(); ++i)
    {
        const cv::Mat &image = imageFrames[i];

        // Prepare the input blob for the model
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // Perform forward pass to get detections
        std::vector<cv::Mat> detections;
        net.forward(detections, net.getUnconnectedOutLayersNames());

        // Log the timestamp
        outFile << "At timestamp " << i << ", detected objects:\n";

        // Process detections
        for (const auto &detection : detections)
        {
            for (int j = 0; j < detection.rows; ++j)
            {
                float confidence = detection.at<float>(j, 4);
                if (confidence > 0.5)
                { // Filter out weak detections
                    int classId = static_cast<int>(detection.at<float>(j, 5));
                    std::string className = classNames[classId];

                    // Get bounding box coordinates
                    float centerX = detection.at<float>(j, 0) * image.cols;
                    float centerY = detection.at<float>(j, 1) * image.rows;
                    float width = detection.at<float>(j, 2) * image.cols;
                    float height = detection.at<float>(j, 3) * image.rows;

                    int left = static_cast<int>(centerX - width / 2);
                    int top = static_cast<int>(centerY - height / 2);
                    int right = static_cast<int>(centerX + width / 2);
                    int bottom = static_cast<int>(centerY + height / 2);

                    // Ensure the bounding box is within the image boundaries
                    left = std::max(0, left);
                    top = std::max(0, top);
                    right = std::min(image.cols - 1, right);
                    bottom = std::min(image.rows - 1, bottom);

                    // Crop the bounding box region from the image
                    cv::Mat bboxRegion = image(cv::Range(top, bottom), cv::Range(left, right));

                    // Calculate the mean RGB values of the bounding box region
                    cv::Scalar meanRGB = cv::mean(bboxRegion);

                    // Log the detected object, its confidence, and RGB values
                    outFile << " - " << className << ": " << confidence
                            << ", RGB: (" << meanRGB[2] << ", " << meanRGB[1] << ", " << meanRGB[0] << ")\n";

                    // Optional: Draw bounding box and label on the image (if needed)
                    cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);
                    std::string label = cv::format("%s: %.2f", className.c_str(), confidence);
                    cv::putText(image, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // Add a separator between timestamps
        outFile << "\n";
    }

    // Close the output file
    outFile.close();
    std::cout << "Detection results saved to: " << outputTextFile << std::endl;
}
// Function to perform object detection and draw bounding boxes
void detectObjects(cv::Mat &image)
{
    // Prepare the input blob for the model
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // Perform forward pass to get detections
    std::vector<cv::Mat> detections;
    net.forward(detections, net.getUnconnectedOutLayersNames());

    // Process detections
    for (const auto &detection : detections)
    {
        for (int i = 0; i < detection.rows; ++i)
        {
            float confidence = detection.at<float>(i, 4);
            if (confidence > 0.5)
            { // Filter out weak detections
                int classId = static_cast<int>(detection.at<float>(i, 5));
                float centerX = detection.at<float>(i, 0) * image.cols;
                float centerY = detection.at<float>(i, 1) * image.rows;
                float width = detection.at<float>(i, 2) * image.cols;
                float height = detection.at<float>(i, 3) * image.rows;

                // Calculate bounding box coordinates
                int left = static_cast<int>(centerX - width / 2);
                int top = static_cast<int>(centerY - height / 2);
                int right = static_cast<int>(centerX + width / 2);
                int bottom = static_cast<int>(centerY + height / 2);

                // Draw bounding box
                cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);

                // Put label
                std::string label = cv::format("%s: %.2f", classNames[classId].c_str(), confidence);
                cv::putText(image, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
}

void createTextureFromImage(const cv::Mat &image)
{
    if (image.empty())
    {
        // std::cerr << "Error: Image is empty!" << std::endl;
        return;
    }

    // Perform object detection
    cv::Mat imageWithDetections = image.clone();
    detectObjects(imageWithDetections);

    if (imageTexture)
        glDeleteTextures(1, &imageTexture);

    glGenTextures(1, &imageTexture);
    glBindTexture(GL_TEXTURE_2D, imageTexture);

    // Convert image to RGB if it's in BGR format (OpenCV default)
    cv::Mat rgbImage;
    if (imageWithDetections.channels() == 3)
    {
        cv::cvtColor(imageWithDetections, rgbImage, cv::COLOR_BGR2RGB);
    }
    else
    {
        rgbImage = imageWithDetections; // If already in RGB format, no need for conversion
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
void RenderChatBox()
{
    ImGui::Begin("Chat Interface");

    // Chat box
    ImGui::InputText("##ChatInput", userInput, IM_ARRAYSIZE(userInput), ImGuiInputTextFlags_EnterReturnsTrue); // Text input field
    ImGui::SameLine();                                                                                         // Place the button on the same line
    if (ImGui::Button("Send") || ImGui::IsKeyPressed(ImGuiKey_Enter))
    { // Send button or Enter key
        if (strlen(userInput) > 0)
        {
            // Add user input to chat history
            chatHistory.push_back("You: " + std::string(userInput));
            // python test
            try
            {
                py::module_::import("sys").attr("path").attr("append")("/home/sujee/sick_gui/scripts/sick_mine_llm");

                // Import your Python module and call a function
                auto process_chat = py::module_::import("main").attr("process_chat");
                std::string response = process_chat(std::string(userInput)).cast<std::string>();
                chatHistory.push_back("LLM: " + response);

                // Clear the input field
                memset(userInput, 0, sizeof(userInput));
            }
            catch (py::error_already_set &e)
            {
                std::cerr << "Python error: " << e.what() << std::endl;
                // Clear the input field
                memset(userInput, 0, sizeof(userInput));
            }
        }
    }

    // Scrollable grid with vertical columns
    // Scrollable table with two columns
    // if (ImGui::BeginTable("chatTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY))
    // {
    //     // Set up columns
    //     ImGui::TableSetupColumn("Timestamp", ImGuiTableColumnFlags_WidthStretch);
    //     ImGui::TableSetupColumn("Comment", ImGuiTableColumnFlags_WidthStretch);
    //     ImGui::TableHeadersRow(); // Add headers

    //     // Add rows
    //     for (int i = 0; i < 3; i++)
    //     {
    //         ImGui::TableNextRow(); // Move to the next row

    //         // Column 1: Clickable Timestamp
    //         ImGui::TableSetColumnIndex(0);
    //         if (ImGui::Selectable(std::to_string(i).c_str(), false, ImGuiSelectableFlags_SpanAllColumns))
    //         {
    //             // Handle click: Update currentFrame and sliderValue
    //             // currentFrame = entry.first;
    //             // sliderValue = static_cast<float>(currentFrame); // Sync slider with clicked timestamp
    //         }

    //         // Column 2: Comment
    //         ImGui::TableSetColumnIndex(1);
    //         ImGui::TextWrapped("%s", "test input"); // Display comment with word wrapping
    //     }

    //     ImGui::EndTable();
    // }

    // Chat response text area
    ImGui::BeginChild("ChatResponse", ImVec2(0, 200), true); // Fixed height for the chat response area
    for (const auto &message : chatHistory)
    {
        ImGui::TextWrapped("%s", message.c_str()); // Display each message with word wrapping
    }
    ImGui::EndChild();

    // End ImGui window
    ImGui::End();
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
                if (projectLidar)
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
        int count = 0;
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
// Function to draw a color-coded gauge
void DrawGauge(const char *label, float value, float min, float max)
{
    if (ImPlot::BeginPlot(label, ImVec2(300, 250), ImPlotFlags_NoLegend | ImPlotFlags_NoMenus))
    {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxesLimits(-1, 1, -1, 1);

        int num_segments = 5;
        float angles[] = {-M_PI / 2, -M_PI / 4, 0, M_PI / 4, M_PI / 2};
        ImVec4 colors[] = {ImVec4(1, 0, 0, 1), ImVec4(1, 0.5, 0, 1), ImVec4(1, 1, 0, 1), ImVec4(0.5, 1, 0, 1), ImVec4(0, 1, 0, 1)};

        for (int i = 0; i < num_segments; i++)
        {
            float arc_x[] = {cos(angles[i]), cos(angles[i + 1])};
            float arc_y[] = {sin(angles[i]), sin(angles[i + 1])};
            ImPlot::PushStyleColor(ImPlotCol_Line, colors[i]);
            ImPlot::PlotLine("##Arc", arc_x, arc_y, 2);
            ImPlot::PopStyleColor();
        }

        // Draw arrow needle
        float theta = ((value - min) / (max - min)) * M_PI - M_PI / 2;
        float needle_x[] = {0, cos(theta)};
        float needle_y[] = {0, sin(theta)};
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0, 0, 0, 1)); // Black arrow
        ImPlot::PlotLine("Needle", needle_x, needle_y, 2);
        ImPlot::PopStyleColor();

        ImPlot::EndPlot();
    }
}
void ShowProgressBars()
{

    // Labels for progress bars
    const char *labels[] = {
        "Wall Convergence", "Roof Sagging", "Floor Uplift",
        "Slope Gradient", "Surface Roughness", "Obstacle Density",
        "Airborne Dust", "Visibility", "Hazardous Areas"};

    // Create progress bars in ImGui window
    ImGui::Begin("Mine Safety Parameters");
    for (int i = 0; i < 9; ++i)
    {
        ImGui::Text("%s", labels[i]);
        ImVec4 color;
        if (progressValues[i] < 0.30f)
        {
            color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f); // Red
        }
        else if (progressValues[i] < 0.50f)
        {
            color = ImVec4(1.0f, 0.5f, 0.0f, 1.0f); // Orange
        }
        else
        {
            color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f); // Green
        }
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, color);
        ImGui::ProgressBar(progressValues[i], ImVec2(0.0f, 0.0f));
        ImGui::PopStyleColor();
    }
    ImGui::End();
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
        ImGui::Separator();
        if (ImGui::Checkbox("Project Lidar Point on Camera Image", &projectLidar))
            IsDataLoaded = false;
    }
    ImGui::End();
    ShowProgressBars();
}
void Render2DMap()
{
    // Check if mapFrames is available
    if (mapFrames.empty())
    {
        // std::cout << "Error: mapFrames is empty!" << std::endl;
        return;
    }

    // Calculate the current frame for the occupancy grid
    int step = pointCloudFrames.size() / mapFrames.size();
    int newFrame = std::min((currentFrame / step), static_cast<int>(mapFrames.size() - 1));

    // Get the current occupancy grid
    const auto &occupancyGrid = mapFrames[newFrame];

    // Check if the grid data is valid
    if (occupancyGrid.data.empty())
    {
        std::cout << "Error: Occupancy grid data is empty!" << std::endl;
        return;
    }

    // Static variable to track if the window is open
    // static bool is_open = false;
    // if (!is_open)
    // {
    //     is_open = true;
    //     // Create a new ImGui window in the bottom-right corner
    //     ImGui::SetNextWindowPos(ImVec2(2000, 1200), ImGuiCond_Always); // Position at bottom-right
    //     ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_Always);  // Size of the 2D map window
    // }

    ImGui::Begin("2D Map");

    // Get the draw list for the current window
    ImDrawList *drawList = ImGui::GetWindowDrawList();

    // Define the origin and scale for the 2D map
    ImVec2 origin = ImGui::GetCursorScreenPos(); // Top-left corner of the window
    float scale = 1.5f;                          // Scale factor to fit the grid in the window (adjust as needed)

    // Grid metadata
    int width = occupancyGrid.info.width;
    int height = occupancyGrid.info.height;

    // Render the occupancy grid
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            // Get the occupancy value for the current cell
            int8_t occupancy = occupancyGrid.data[y * width + x];

            // Calculate the position of the cell in the window
            ImVec2 cellPos = ImVec2(origin.x + x * scale, origin.y + y * scale);

            // Determine the color based on the occupancy value
            ImU32 color;
            if (occupancy == -1)
            {
                color = IM_COL32(128, 128, 128, 255); // Gray for unknown
            }
            else if (occupancy >= 50)
            {
                color = IM_COL32(255, 0, 0, 255); // Red for occupied
            }
            else
            {
                color = IM_COL32(0, 255, 0, 255); // Green for free
            }

            // Draw the cell as a filled rectangle
            drawList->AddRectFilled(cellPos, ImVec2(cellPos.x + scale, cellPos.y + scale), color);
        }
    }

    // Render the robot's path from mapPaths
    if (!mapPaths.empty() && newFrame < mapPaths.size())
    {
        const auto &path = mapPaths[newFrame].poses; // Get the path for the current frame

        // Convert path points to screen coordinates
        std::vector<ImVec2> screenPath;
        for (const auto &pose : path)
        {
            float gridX = (pose.pose.position.x - occupancyGrid.info.origin.position.x) / occupancyGrid.info.resolution;
            float gridY = (pose.pose.position.y - occupancyGrid.info.origin.position.y) / occupancyGrid.info.resolution;
            ImVec2 screenPos = ImVec2(origin.x + gridX * scale, origin.y + gridY * scale);
            screenPath.push_back(screenPos);
        }

        // Draw the path as a connected line
        if (screenPath.size() > 1)
        {
            for (size_t i = 1; i < screenPath.size(); ++i)
            {
                drawList->AddLine(screenPath[i - 1], screenPath[i], IM_COL32(0, 0, 255, 255), 2.0f); // Blue line for the path
            }
        }

        // Draw an arrow to indicate the robot's current direction
        if (screenPath.size() >= 2)
        {
            ImVec2 direction = ImVec2(screenPath.back().x - screenPath[screenPath.size() - 2].x,
                                      screenPath.back().y - screenPath[screenPath.size() - 2].y);
            float angle = atan2(direction.y, direction.x); // Calculate the angle of the direction
            ImVec2 arrowEnd = ImVec2(screenPath.back().x + 10.0f * cos(angle),
                                     screenPath.back().y + 10.0f * sin(angle));
            drawList->AddLine(screenPath.back(), arrowEnd, IM_COL32(255, 255, 0, 255), 2.0f); // Yellow arrow for direction
        }
    }

    ImGui::End();
}
// Load extrinsic matrix from YAML file
Eigen::Matrix4d loadExtrinsicMatrix(const std::string &yamlPath)
{
    YAML::Node config = YAML::LoadFile(yamlPath);
    if (!config["extrinsic_matrix"])
    {
        throw std::runtime_error("YAML file does not contain 'extrinsic_matrix' key.");
    }

    Eigen::Matrix4d T;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            try
            {
                T(i, j) = config["extrinsic_matrix"][i * 4 + j].as<double>();
            }
            catch (const YAML::TypedBadConversion<double> &e)
            {
                throw std::runtime_error("Failed to parse extrinsic_matrix: Invalid value at row " +
                                         std::to_string(i) + ", column " + std::to_string(j));
            }
        }
    }
    return T;
}

// Load camera calibration from YAML file
void loadCameraCalibration(const std::string &yamlPath, cv::Mat &cameraMatrix, cv::Mat &distCoeffs)
{
    YAML::Node config = YAML::LoadFile(yamlPath);
    if (!config["camera_matrix"] || !config["distortion_coefficients"])
    {
        throw std::runtime_error("YAML file does not contain required keys.");
    }

    // Load camera matrix
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            cameraMatrix.at<double>(i, j) = config["camera_matrix"]["data"][i * 3 + j].as<double>();
        }
    }

    // Load distortion coefficients
    distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
    for (int i = 0; i < 5; ++i)
    {
        distCoeffs.at<double>(0, i) = config["distortion_coefficients"]["data"][i].as<double>();
    }
}

// Project LiDAR points to image plane and extract RGB values
void projectLidarToImage(std::vector<Point3D> &lidarPoints,
                         const Eigen::Matrix4d &T_lidar_to_cam,
                         const cv::Mat &cameraMatrix,
                         const cv::Mat &distCoeffs,
                         cv::Mat &image)
{
    // Convert LiDAR points to homogeneous coordinates
    std::vector<cv::Point3d> points3d;
    for (auto &point : lidarPoints)
    {
        Eigen::Vector4d point_homogeneous(point.x, point.y, point.z, 1.0);
        Eigen::Vector4d point_cam = T_lidar_to_cam * point_homogeneous;
        if (point_cam.z() > 0)
        { // Only consider points in front of the camera
            points3d.emplace_back(point_cam.x(), point_cam.y(), point_cam.z());
        }
    }

    // Project 3D points to 2D image plane
    std::vector<cv::Point2d> imagePoints;
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); // Rotation vector (zero for no rotation)
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F); // Translation vector (zero for no translation)
    cv::projectPoints(points3d, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // Draw projected points on the image and extract RGB values
    for (size_t i = 0; i < imagePoints.size(); ++i)
    {
        const auto &point = imagePoints[i];
        if (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows)
        {

            // Extract RGB values from the image
            cv::Vec3b rgb = image.at<cv::Vec3b>(point.y, point.x);

            // Assign RGB values to the corresponding LiDAR point
            lidarPoints[i].r = rgb[2] / 255.0f; // Red
            lidarPoints[i].g = rgb[1] / 255.0f; // Green
            lidarPoints[i].b = rgb[0] / 255.0f; // Blue

            // Draw the projected point on the image
            cv::circle(image, point, 2, cv::Scalar(0, 255, 0), -1); // Green circle
        }
    }
}
std::map<float, float> computeWallWidth(const std::vector<Point3D> &pointCloud)
{
    std::map<float, std::pair<float, float>> ySections; // Stores min and max X for each Y-section
    std::map<float, float> avgWidth;                    // Stores computed width per Y-section

    // Identify min and max X values at each Y-section
    for (const auto &point : pointCloud)
    {
        float y = point.y;
        if (ySections.find(y) == ySections.end())
        {
            ySections[y] = {point.x, point.x}; // Initialize min and max X
        }
        else
        {
            ySections[y].first = std::min(ySections[y].first, point.x);   // Update min X
            ySections[y].second = std::max(ySections[y].second, point.x); // Update max X
        }
    }

    // Compute average width per Y-section
    for (const auto &section : ySections)
    {
        float y = section.first;
        float width = section.second.second - section.second.first;
        avgWidth[y] = width;
    }

    return avgWidth;
}

// Function to calculate percentage change between two scans
void compareWallWidthChanges(const std::map<float, float> &oldWidth, const std::map<float, float> &newWidth)
{
    for (const auto &newSection : newWidth)
    {
        float y = newSection.first;
        if (oldWidth.find(y) != oldWidth.end())
        {
            float oldW = oldWidth.at(y);
            float newW = newSection.second;
            float percentageChange = ((newW - oldW) / oldW) * 100.0f;

            std::cout << "Y-section: " << y
                      << " | Old Width: " << oldW
                      << " | New Width: " << newW
                      << " | % Change: " << percentageChange << "%"
                      << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    // Initialize ROS 2
    loadYOLOModel(net, classNames);
    // Load calibration data
    std::string extrinsicYaml = "/home/sujee/sick_gui/config/camera_extrinsic_calibration.yaml";
    std::string cameraYaml = "/home/sujee/sick_gui/config/camera_intrinsic_calibration.yaml";

    Eigen::Matrix4d T_lidar_to_cam = loadExtrinsicMatrix(extrinsicYaml);
    cv::Mat cameraMatrix, distCoeffs;
    loadCameraCalibration(cameraYaml, cameraMatrix, distCoeffs);
    rclcpp::init(argc, argv);
    py::scoped_interpreter guard{};

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
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Slider parameters for zipping through timestamps
    float sliderValue = 0.0f;
    if (!pointCloudFrames.empty())
    {
        sliderValue = static_cast<float>(currentFrame);
    }
    // Initialize random seed

    std::srand(std::time(nullptr));

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
                // store data in rag
                // std::string outputTextFile = "detection_results.txt";
                // detectObjectsAndLog(outputTextFile);
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
                currentFrame = static_cast<int>(sliderValue);
                IsDataLoaded = false;
                for (int i = 0; i < 9; ++i)
                {
                    progressValues[i] = static_cast<float>(std::rand() % 101) / 100.0f;
                }
            }
            // Add buttons for incrementing and decrementing the slider value
            if (ImGui::Button("-"))
            {
                sliderValue = std::max(sliderValue - 1.0f, 0.0f); // Decrease by 1, but don't go below 0
                currentFrame = static_cast<int>(sliderValue);
            }
            ImGui::SameLine(); // Place the next widget on the same line
            if (ImGui::Button("+"))
            {
                sliderValue = std::min(sliderValue + 1.0f, static_cast<float>(pointCloudFrames.size() - 1));

                currentFrame = static_cast<int>(sliderValue);
            }
            cv::Mat image;
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
                // Project LiDAR points to image
                if (projectLidar)
                {
                    image = imageFrames[currentFrame].clone();
                    projectLidarToImage(pointCloudFrames[currentFrame], T_lidar_to_cam, cameraMatrix, distCoeffs, image);
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
                if (projectLidar)
                    createTextureFromImage(image);
                else
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
        RenderChatBox();
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
        // Render the 2D map in the bottom-right corner
        Render2DMap();
        RenderImGuiInterface();
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