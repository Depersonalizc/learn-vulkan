#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <cstring>


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

class HelloTriangleApplication {
public:
    void run() {
        initWindow();  // Setup GLFW
        initVulkan();  // Create instances
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;
    VkInstance instance;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Do not create an OpenGL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // Disable resizing for now, as it requires special care

        // width, height, title, monitor, only_relevant_to_opengl
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void validateRequiredExtensions(const char **requiredExtensions, uint32_t requiredExtensionCount) {
        // Retrieve a list of supported extensions, each VkExtensionProperties struct contains name and version
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);            // Get the number of extensions
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());  // Get info about extensions
        //std::cout << "available extensions:\n";
        //for (const auto& extension : extensions)
        //    std::cout << '\t' << extension.extensionName << '\n';

        for (int i = 0; i < requiredExtensionCount; i++) {
            bool found = false;
            for (const auto& ext : extensions) {
                if (std::string(requiredExtensions[i]) == std::string(ext.extensionName))
                    found = true;
            }
            if (!found)
                throw std::runtime_error("Oops, required extension not found");
        }
    }

    void createInstance() {
        // VkApplicationInfo provides useful information to the driver in order to optimize our specific application
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        // VkInstanceCreateInfo tells driver which global extensions and validation layers to use
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);  // required extensions
        validateRequiredExtensions(glfwExtensions, glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        createInfo.enabledLayerCount = 0;


        // vkCreateXXX(pXXXCreateInfo, pAllocator, pXXX)
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
            throw std::runtime_error("failed to create instance");
    }

    void initVulkan() {
        createInstance();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            // TODO: Draw current frame

        }
    }

    void cleanup() {
        std::cout << "Performing cleanup...\n";

        // NOTE: All objects created by the instance must be destroyed before this line
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();

        std::cout << "Bye!\n";
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
