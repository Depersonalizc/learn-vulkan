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

//#include <cstdint>  // uint32_t
#include <limits>     // std::numeric_limits
#include <algorithm>  // std::clamp


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

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;

    inline bool isAdequate() const { return !formats.empty() && !presentModes.empty(); }
};

// Screen-coordinate dimensions (<= pixel-coordinate)
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
    /* Instance */
    VkInstance instance;

    /* Window & Surface */
    GLFWwindow* window;
    VkSurfaceKHR surface;

    /* Devices & Queues */
    VkDevice device;
    VkQueue graphicsQueue, presentQueue;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    /* Swap Chains */
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    VkFormat swapChainImageFormat;          // format of the surface associated with the swap chain
    VkExtent2D swapChainExtent;

private:
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

        createSurface();          // Create surface right after instance creation, 
                                  // as it could affect physical device selection
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();        // Create swap chain for the device
        createImageViews();       // Create views to images of the swap chain
        createGraphicsPipeline();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            // TODO: Draw current frame

        }
    }

    void cleanup() {
        std::cout << "Performing cleanup...\n";
        
        for (const auto& imageView : swapChainImageViews)
            vkDestroyImageView(device, imageView, nullptr);

        vkDestroySwapchainKHR(device, swapChain, nullptr);
        // NOTE: All objects created by the device must be destroyed before this line
        vkDestroyDevice(device, nullptr);

        vkDestroySurfaceKHR(instance, surface, nullptr);
        // NOTE: All objects created by the instance must be destroyed before this line
        vkDestroyInstance(instance, nullptr);
        
        glfwDestroyWindow(window);
        glfwTerminate();

        std::cout << "Bye!\n";
    }

    /* Helpers for initVulkan */
    // Create an instance - specify extensions, validation layers, etc.
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

    /* Physical Device */
    // Pick the physical device with all the supports we need
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

    // Find a suitable physical device that supports
    //  - Graphics and Present queue families
    //  - all of `deviceExtensions`
    //  - all of 
    bool isPhysDeviceSuitable(const VkPhysicalDevice physDevice) const {
        QueueFamilyIndices indices = findQueueFamilies(physDevice);
        return indices.isComplete()
            && checkPhysDeviceExtensionSupport(physDevice)      // device's extension supports
            && querySwapChainSupport(physDevice).isAdequate();  // device's swap chain support
    }

    QueueFamilyIndices findQueueFamilies(const VkPhysicalDevice physDevice) const {
        // Find all queue families that we need (indices within the supported families of the device)
        // We want graphics and present (KHR surface) supports.
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
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physDevice, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> supportedExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physDevice, nullptr, &extensionCount, supportedExtensions.data());
        
        for (const auto& deviceExtension : deviceExtensions) {  // Check support for deviceExtensions we need
            bool extensionFound = false;
            for (const auto& supportedExtension : supportedExtensions) {
                if (std::string(supportedExtension.extensionName) == deviceExtension) {
                    extensionFound = true;
                    break;
                }
            }
            if (!extensionFound)
                return false;
        }
        return true;
    }


    /* Swap Chain */
    void createSwapChain() {
        auto swapChainSupports = querySwapChainSupport(physicalDevice);

        auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupports.formats);
        auto presentMode = chooseSwapPresentMode(swapChainSupports.presentModes);
        auto extent = chooseSwapExtent(swapChainSupports.capabilities);
        uint32_t minImageCount = swapChainSupports.capabilities.minImageCount + 1;
        if (swapChainSupports.capabilities.maxImageCount)
            minImageCount = std::min(minImageCount, swapChainSupports.capabilities.maxImageCount);

        // Create the swap chain
        VkSwapchainCreateInfoKHR swapChainCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = minImageCount,
            .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,  // #layers each image consists of, always 1 unless going steroscopic
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,                // directly render to this chain

            .preTransform = swapChainSupports.capabilities.currentTransform,  // No additional transformations
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,              // Ignore alpha channel

            .presentMode = presentMode,
            .clipped = VK_TRUE,                                               // Ignore obscured pixels
            .oldSwapchain = VK_NULL_HANDLE,                                   // Handle to previous swap chain if -
                                                                              // - we're recreating one at run time
        };


        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
        if (indices.graphicsFamily.value() != indices.presentFamily.value()) {
            /**
              * VK_SHARING_MODE_CONCURRENT: An image can be used across multiple 
              *     queue families without explicit ownership transfers.
            **/
            swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
            swapChainCreateInfo.queueFamilyIndexCount = 2;
        }
        else {
            /**
              * VK_SHARING_MODE_EXCLUSIVE: An image is owned by one queue family at a time.
              *     Ownership must be explicitly transferred between queue families. Better performance
            **/
            swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            swapChainCreateInfo.pQueueFamilyIndices = nullptr;
            swapChainCreateInfo.queueFamilyIndexCount = 0;
        }

        // Create the swap chain
        VkResult res = vkCreateSwapchainKHR(device, &swapChainCreateInfo, nullptr, &swapChain);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to create swap chain, error ") + string_VkResult(res)
            );
        }

        // Retrieve swap chain images handles
        uint32_t imageCount = 0;
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        // Store the swap extent and surface image format
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    // Query physical device's swap chain support
    SwapChainSupportDetails querySwapChainSupport(const VkPhysicalDevice physDevice) const {
        SwapChainSupportDetails details;

        // Get surface capabilities
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDevice, surface, &details.capabilities);

        // Get surface formats
        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &formatCount, nullptr);
        if (formatCount) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &formatCount, details.formats.data());
        }

        // Get surface presentation modes (of varying optimality)
        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physDevice, surface, &presentModeCount, nullptr);
        if (presentModeCount) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(physDevice, surface, &formatCount, details.presentModes.data());
        }

        return details;
    }

    // Choose surface format for the swap chain: 8-bit BGRA pixel format and sRGB color space 
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& surfaceFormats) const {
        assert(surfaceFormats.size() && "Number of available surface formats is 0");
        for (const auto& surfaceFormat : surfaceFormats) {
            if (surfaceFormat.format == VK_FORMAT_B8G8R8A8_SRGB
                && surfaceFormat.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR) 
            {
                return surfaceFormat;
            }
        }
        return surfaceFormats[0];
    }

    // Choose presentation mode for the swap chain
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& presentModes) const {
        assert(presentModes.size() && "Number of available presentation modes is 0");
        for (const auto& presentMode : presentModes) {
            if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
                return presentMode;
        }
        return VK_PRESENT_MODE_FIFO_KHR;  // guaranteed support
    }

    // Choose extent for the swap chain
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())  // Best match
            return capabilities.currentExtent;

        int widthPx, heightPx;
        glfwGetFramebufferSize(window, &widthPx, &heightPx);

        VkExtent2D extentPx { 
            std::clamp(
                static_cast<uint32_t>(widthPx), 
                capabilities.minImageExtent.width, 
                capabilities.maxImageExtent.width
            ), 
            std::clamp(
                static_cast<uint32_t>(heightPx),
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height
            )
        };
        
        return extentPx;
    }

    // Create views of the images of the swap chain
    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());
        for (int i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo imageViewCreateInfo {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = swapChainImages[i],
                
                // viewType and format specify how image data should be interpreted
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = swapChainImageFormat,
                
                // Swizzle color channels
                .components = {
                    VK_COMPONENT_SWIZZLE_IDENTITY,  // r 
                    VK_COMPONENT_SWIZZLE_IDENTITY,  // g
                    VK_COMPONENT_SWIZZLE_IDENTITY,  // b
                    VK_COMPONENT_SWIZZLE_IDENTITY,  // a
                },
                
                // Describes purpose of the image and which part should be accessed
                .subresourceRange = {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,  // used as color target
                    .baseMipLevel   = 0,                          // no mip-mapping
                    .levelCount     = 1,                          // no mip-mapping
                    .baseArrayLayer = 0,                          // ?
                    .layerCount     = 1,                          // 1, unless going stereographic
                },
            };

            // Create image view
            VkResult res = vkCreateImageView(device, &imageViewCreateInfo, nullptr, &swapChainImageViews[i]);
            if (res != VK_SUCCESS) {
                throw std::runtime_error(
                    std::string("failed to create image view, error ") + string_VkResult(res)
                );
            }
        }
    }

    /* Logical Device */
    void createLogicalDevice() {
        // Specify the queues to be created for the device
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);  // find queue family index (among those supported by the physical device)
        assert(indices.isComplete() && "Physical device should have supporting queue families");

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        
        // Potentially two queue families, for graphics and presentation queues resp.
        std::set<uint32_t> queueFamilyIndices { 
            indices.graphicsFamily.value(), 
            indices.presentFamily.value() 
        };

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
        // just as in vkCreateInstance, except now the infos are device-specific
        VkDeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        
        // Pointers to the queue createInfos and device features
        deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
        deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

        // Device extensions and validation layers
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
        
        // Device validation layers have been deprecated, but we include here for backward compatability
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

    /* Graphics Pipeline */
    void createGraphicsPipeline() {
        
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
