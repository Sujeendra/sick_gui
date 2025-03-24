#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
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
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <GL/gl.h>

// LiDAR data structure
struct Point3D
{
    float x, y, z, intensity;
};

// Global variables for camera
float cameraAngleX = 0.0f; // Horizontal rotation
float cameraAngleY = 0.0f; // Vertical rotation
float zoomLevel = 15.0f;   // Zoom distance
bool isDragging = false;   // Mouse drag state
double lastMouseX, lastMouseY;
GLuint imageTexture;
std::vector<cv::Mat> imageFrames; // Vector to hold image frames

// Vector to hold point cloud data for each frame
std::vector<std::vector<Point3D>> pointCloudFrames;
std::vector<double> pointCloudTimestamps; // Store timestamps for each frame
int currentFrame = 0;
float pointSize = 5.0f;

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

void renderPointCloud(const std::vector<Point3D> &points, float pointSize)
{
    glPointSize(pointSize);
    glBegin(GL_POINTS);
    for (const auto &point : points)
    {
        // Normalize intensity to [0, 1] and map it to RGB colors
        float intensityNormalized = std::min(1.0f, std::max(0.0f, point.intensity));

        // Map intensity to RGB color (low intensity = blue, high intensity = red)
        float red = intensityNormalized;
        float green = 0.0f;
        float blue = 1.0f - intensityNormalized;

        glColor3f(red, green, blue);
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
}

void parsePointCloudDataFromBag(const std::string &bagFile, const std::string &pointCloudTopic, const std::string &imageTopic)
{
    rosbag2_cpp::Reader reader;
    reader.open({bagFile, "sqlite3"});
    int count = 0;

    while (reader.has_next())
    {
        auto msg = reader.read_next();

        if (msg->topic_name == pointCloudTopic)
        {
            rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
            rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serializer;
            auto pointCloudMsg = std::make_shared<sensor_msgs::msg::PointCloud2>();
            serializer.deserialize_message(&serialized_msg, pointCloudMsg.get());

            // Process point cloud (same as before)
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
            pointCloudTimestamps.push_back(count);
            count++;
        }
        else if (msg->topic_name == imageTopic)
        {
            rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
            rclcpp::Serialization<sensor_msgs::msg::Image> serializer;
            auto imageMsg = std::make_shared<sensor_msgs::msg::Image>();
            serializer.deserialize_message(&serialized_msg, imageMsg.get());

            // Convert ROS 2 Image message to OpenCV Mat
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(imageMsg, "bgr8");
            imageFrames.push_back(cv_ptr->image);
        }
    }
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

// Mouse callback to track dragging
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            isDragging = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
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

int main(int argc, char **argv)
{
    // Initialize ROS 2
    rclcpp::init(argc, argv);

    // Open the bag file and extract data
    parsePointCloudDataFromBag("/home/sujee/camera_plus_lidar/test/test_0.db3",
                               "/cloud_unstructured_fullframe",
                               "/camera/image_raw");

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

        // ImGui controls
        if (!pointCloudFrames.empty())
        {
            ImGui::Begin("LiDAR Controls");
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
                if (currentFrame >= 0 && currentFrame < imageFrames.size())
                {
                    createTextureFromImage(imageFrames[currentFrame]);
                }
            }
            ImGui::Text("Timestamp: %.2f", pointCloudTimestamps[currentFrame]); // Display the timestamp
            ImGui::End();
        }

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

        if (currentFrame >= 0 && currentFrame < pointCloudFrames.size())
        {
            renderPointCloud(pointCloudFrames[currentFrame], pointSize);
        }
        renderAxes();

        // Image Rendering
        glViewport(1080, 1000, 1000, 500);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Reset OpenGL states for image rendering
        glDisable(GL_DEPTH_TEST);          // Disable depth testing
        glDisable(GL_LIGHTING);            // Disable lighting
        glBindTexture(GL_TEXTURE_2D, 0);   // Unbind textures
        glDisable(GL_TEXTURE_2D);          // Disable texturing
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f); // Reset color to white

        // Render the image
        if (!imageFrames.empty() && currentFrame >= 0 && currentFrame < imageFrames.size())
        {
            renderImage();
        }

        // Re-enable depth testing if needed later
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
    rclcpp::shutdown();
    return 0;
}
