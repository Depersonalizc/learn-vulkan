#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <cassert>
#include <optional>
#include <set>


#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif


struct QueueFamilyIndices {
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> graphicsFamily;

    inline bool isComplete() const { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
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
    VkSurfaceKHR surface;
    VkInstance instance;

    VkDevice device;
    VkQueue graphicsQueue, presentQueue;
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

        createSurface();  // Create surface right after instance creation, 
                          // since it could affect physical device selection
        pickPhysicalDevice();
        createLogicalDevice();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            // TODO: Draw current frame

        }
    }

    void cleanup() {
        std::cout << "Performing cleanup...\n";

        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        // NOTE: All objects created by the instance must be destroyed before this line
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();

        std::cout << "Bye!\n";
    }

    /* Helpers for initVulkan */

    // Instance creation - specify extensions, validation layers, etc.
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
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);  // extensions required by glfw
        validateRequiredExtensions(glfwExtensions, glfwExtensionCount); // make sure all extensions requried are supported by Vulkan

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
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

    // Pick the physical device with the supports we need
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        if (!deviceCount)
            throw std::runtime_error("failed to find GPUs with Vulkan support");

        std::vector<VkPhysicalDevice> physDevs(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, physDevs.data());

        // Simply pick the first suitable device
        for (const auto& physDev : physDevs) {
            if (isPhysDeviceSuitable(physDev)) {
                physicalDevice = physDev;
                return;
            }
        }

        throw std::runtime_error("Failed to find a suitable GPU");
    }

    bool isPhysDeviceSuitable(const VkPhysicalDevice physDevice) const {
        QueueFamilyIndices indices = findQueueFamilies(physDevice);
        
        return indices.isComplete() && checkPhysDeviceExtensionSupport(physDevice);
    }

    QueueFamilyIndices findQueueFamilies(const VkPhysicalDevice physDevice) const {
        // Find all queue families that we need (indices within the supported families of the device)
        QueueFamilyIndices indices;  // init with no value

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, queueFamilies.data());

        // Find the supporting queue family indices
        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (indices.isComplete())
                break;

            // Check graphics support
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                indices.graphicsFamily = i;

            // Check present (KHR surface) support
            VkBool32 supportKHRSurface = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physDevice, i, surface, &supportKHRSurface);
            if (supportKHRSurface)
                indices.presentFamily = i;

            i++;
        }

        return indices;
    }

    bool checkPhysDeviceExtensionSupport(const VkPhysicalDevice physDevice) const {
        using namespace std;
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physDevice, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> supportedExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physDevice, nullptr, &extensionCount, supportedExtensions.data());
        
        for (const auto& deviceExtension : deviceExtensions) {
            bool extensionFound = false;
            for (const auto& supportedExtension : supportedExtensions) {
                if (string(supportedExtension.extensionName) == deviceExtension) {
                    extensionFound = true;
                    break;
                }
            }
            if (!extensionFound)
                return false;
        }
        return true;
    }


    // Logical device
    void createLogicalDevice() {
        // Specify the queues to be created for the device
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);  // find queue family index (among those supported by the physical device)
        assert(indices.isComplete(), "Physical device should have supporting queue families");

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> queueFamilyIndices = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePrio = 1.0f;
        for (auto queueFamilyIndex : queueFamilyIndices) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.pQueuePriorities = &queuePrio;
            queueCreateInfo.queueCount = 1;  // queues to be created in the queue family (spcified by `indices`)
            queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        
        // Speicify device features, left empty for now
        VkPhysicalDeviceFeatures deviceFeatures{};
        
        // Create the logical device with vkCreateDevice; Need to specify extensions, validation layers 
        // just as in vkCreateInstance, but the infos are now device-specific
        VkDeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        // Pointers to the queue createInfos and device features
        deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
        deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

        // Device extensions and validation layers
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
        // Device validation layers have been deprecated, but we include for backward compatability
        if (enableValidationLayers) {
            deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            deviceCreateInfo.enabledLayerCount = 0;
        }

        VkResult res = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to create device, error ") + string_VkResult(res)
            );
        }

        // Get the handles for the device queues
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    void createSurface() {
        // Platform-specific surface creation, conveniently handled by GLFW
        // Ex. vkCreateWin32SurfaceKHR on Windows, and vkCreateXcbSurfaceKHR on Linux
        VkResult res = glfwCreateWindowSurface(instance, window, nullptr, &surface);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to create window surface, error ") + string_VkResult(res)
            );
        }
    }

    /* Error Checking */

    void validateRequiredExtensions(const char **requiredExtensions, uint32_t requiredExtensionCount) {
        // Retrieve a list of supported extensions, each VkExtensionProperties struct contains name and version
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);            // Get the number of extensions
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());  // Get info about extensions

        for (uint32_t i = 0; i < requiredExtensionCount; i++) {
            bool found = false;
            for (const auto& ext : extensions) {
                if (std::string(requiredExtensions[i]) == ext.extensionName)
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
