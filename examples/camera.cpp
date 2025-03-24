#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <GL/gl.h>

GLuint imageTexture;
std::vector<cv::Mat> imageFrames; // Vector to hold image frames
int currentFrame = 0;

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

void clearScreen()
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // Set clear color to white
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void parseImageDataFromBag(const std::string &bagFile, const std::string &imageTopic)
{
    rosbag2_cpp::Reader reader;
    reader.open({bagFile, "sqlite3"});
    while (reader.has_next())
    {
        auto msg = reader.read_next();

        if (msg->topic_name == imageTopic)
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

int main(int argc, char **argv)
{
    // Initialize ROS 2
    rclcpp::init(argc, argv);

    // Open the bag file and extract image data
    parseImageDataFromBag("/home/sujee/camera_plus_lidar/test/test_0.db3", "/camera/image_raw");

    // Initialize GLFW and ImGui
    if (!glfwInit())
        return -1;
    GLFWwindow *window = glfwCreateWindow(2160, 1920, "Image Viewer", nullptr, nullptr);
    if (!window)
        return -1;
    glfwMakeContextCurrent(window);

    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Slider parameters for zipping through timestamps
    float sliderValue = 0.0f;
    if (!imageFrames.empty())
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
        if (!imageFrames.empty())
        {
            ImGui::Begin("Image Controls");
            if (ImGui::SliderFloat("Frame", &sliderValue, 0.0f, static_cast<float>(imageFrames.size() - 1), "%.0f"))
            {
                currentFrame = static_cast<int>(sliderValue);
                if (currentFrame >= 0 && currentFrame < imageFrames.size())
                {
                    createTextureFromImage(imageFrames[currentFrame]);
                }
            }
            ImGui::End();
        }

        // OpenGL rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        // Set up orthographic projection for image
        glViewport(0, 0, 2160, 1920); // Full screen for the image
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f); // Orthographic projection for 2D image

        // Render the image if a valid frame is loaded
        if (!imageFrames.empty() && currentFrame >= 0 && currentFrame < imageFrames.size())
        {
            renderImage();
        }

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
