#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <string>


#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};


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
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    /* Main Routines */

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Do not create an OpenGL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // Disable resizing for now, as it requires special care

        // width, height, title, monitor, only_relevant_to_opengl
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();

        pickPhysicalDevice();
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

    /* Helpers for initVulkan */

    void createInstance() {
        // Check validation layers support
        if (enableValidationLayers && !checkValidationLayerSupport())
            throw std::runtime_error("validation layers requested, but some not supported");


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
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);  // required extensions
        validateRequiredExtensions(glfwExtensions, glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = validationLayers.size();
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }


        // vkCreateXXX(pXXXCreateInfo, pAllocator, pXXX)
        VkResult res = vkCreateInstance(&createInfo, nullptr, &instance);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to create instance, error ") + string_VkResult(res)
            );
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        if (!deviceCount)
            throw std::runtime_error("failed to find GPUs with Vulkan support");

        std::vector<VkPhysicalDevice> physDevs(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, physDevs.data());

        for (const auto& dev : physDevs) {
            if (isDeviceSuitable(dev)) {
                physicalDevice = dev;
                return;
            }
        }

        throw std::runtime_error("Failed to find a suitable GPU");
    }

    bool isDeviceSuitable(const VkPhysicalDevice physDevice) const {
        return true;
    }

    /* Error Checking */

    void validateRequiredExtensions(const char **requiredExtensions, uint32_t requiredExtensionCount) {
        // Retrieve a list of supported extensions, each VkExtensionProperties struct contains name and version
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);            // Get the number of extensions
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());  // Get info about extensions

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

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const auto& layer : validationLayers) {
            bool found = false;
            for (const auto& availableLayer : availableLayers) {
                if (std::string(layer) == availableLayer.layerName) {
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }

        return true;
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
