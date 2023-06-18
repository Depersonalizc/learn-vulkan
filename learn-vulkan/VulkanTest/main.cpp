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
#include <fstream>


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
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkFormat swapChainImageFormat;          // format of the surface associated with the swap chain
    VkExtent2D swapChainExtent;

    /* Pipeline */
    VkPipeline graphicsPipeline;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;

    /* Command Buffers */
    VkCommandBuffer commandBuffer;
    VkCommandPool commandPool;

    /* Sync Primitives */
    VkSemaphore imageAvailableSemaphore;  // Signaled when an image is acquired from swapchain, ready for render
    VkSemaphore renderFinishedSemaphore;  // Signaled when rendering is finished and presentation can execute
    VkFence inFlightFence;  // Avoid overwriting the command buffer when presentation not finished

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
        
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffer();
        createSyncPrimitives();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        // Since many calls in drawFrame are asynchronous, we might be destroying resources on GPU when
        // they are still in use for drawing / presentation. So we wait until device finishes execution. 
        vkDeviceWaitIdle(device);
    }

    void drawFrame() {
        // Wait for the previous frame to finish
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        // Reset the fence to unsignaled
        vkResetFences(device, 1, &inFlightFence);

        uint32_t imageIndex;
        // Acquire the next image. When finished, SIGNAL [1] the imageAvailableSemaphore
        vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        // Empty the command buffer
        vkResetCommandBuffer(commandBuffer, 0);

        // Record command buffer: Draw into imageIndex
        recordCommandBuffer(commandBuffer, imageIndex);

        // Submit the command buffer
        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
        };

        // Wait semaphores: GPU should wait until image has been acquired [1] from swapchain
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;  // GPU waits on imageAvailableSemaphore to be signaled
        submitInfo.pWaitDstStageMask = waitStages;    // Tell GPU to only start waiting when it's time to write to the color attachment
        
        // Signal semaphores: GPU should SIGNAL [2] the renderFinishedSemaphore when finished
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        // SUBMIT
        // Signal `inFlightFence` when the graphics cmd buffer finishes.
        // This allows the CPU to know the command buffer is safe to overwrite (so it can proceed)
        VkResult res = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to submit command buffer, error ") + string_VkResult(res)
            );
        }

        // Submit request of presenting on swapchain to the present queue
        VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = signalSemaphores,  // wait until [2] draw command buffer done
            .swapchainCount = 1,
            .pSwapchains = &swapChain,
            .pImageIndices = &imageIndex,  // present the image the previous command buffer just wrote to
            .pResults = nullptr,  // check if presentation successful
        };
        vkQueuePresentKHR(presentQueue, &presentInfo);
    }

    void cleanup() {
        std::cout << "Performing cleanup...\n";

        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroyFence(device, inFlightFence, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);

        for (const auto &framebuffer : swapChainFramebuffers)
            vkDestroyFramebuffer(device, framebuffer, nullptr);

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

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
    //  - All of `deviceExtensions`
    //  - desired swap chain functionalities 
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

    // Create frame buffers, one for each VkImageView (attachment)
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkFramebufferCreateInfo framebufferInfo{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = renderPass,
                .attachmentCount = 1,
                .pAttachments = &swapChainImageViews[i],
                .width = swapChainExtent.width,
                .height = swapChainExtent.height,
                .layers = 1,  // #layers in swapchain images
            };

            VkResult res = vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]);
            if (res != VK_SUCCESS) {
                throw std::runtime_error(
                    std::string("failed to create frame buffer, error ") + string_VkResult(res)
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
        VkResult res;

        auto vertShaderSpv = readFile("VulkanTest/shaders/spv/triangle_vert.spv");
        auto fragShaderSpv = readFile("VulkanTest/shaders/spv/triangle_frag.spv");

        // Shader modules: Thin wrapper around the spir-v byte code
        VkShaderModule vertShaderModule = createShaderModule(vertShaderSpv);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderSpv);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertShaderModule,
            .pName = "main",
        };
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragShaderModule,
            .pName = "main",
        };
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        /* Fix-function modules */

        // Vertex Input: Specify format of vertex data, including
        //  - Binding: spacing, whether data is per-vertex or per-instance
        //  - Attributes: Types of attributes, which binding to load them and at which offset
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

        // Input Assembly: 
        // - What kind of geometry will be drawn from the vertices (topology)
        // - Primitive restart?
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        // Viewport and Scissors: Make them dynamic state
        std::vector<VkDynamicState> dynamicStates{
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        // Since these are dynamic state, we only need to specify count at creation time 
        // w/o passing in the createInfo. Will be actually set up at drawing time
        VkPipelineDynamicStateCreateInfo dynamicStateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };
        VkPipelineViewportStateCreateInfo viewportStateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };

        // Rasterizer
        VkPipelineRasterizationStateCreateInfo rasterizerInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,          // Do fragments beyond viewport min-max depth get clamped, or discarded?
            .rasterizerDiscardEnable = VK_FALSE,   // Disable output to the framebuffer?
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,           // depth bias (disbale for now)
            //.depthBiasConstantFactor = 0.0f,       // Optional
            //.depthBiasClamp = 0.0f,                // Optional
            //.depthBiasSlopeFactor = 0.0f,          // Optional
            .lineWidth = 1.0f,
        };

        // Multisampling (disable for now)
        VkPipelineMultisampleStateCreateInfo multisamplingInfo{};
        multisamplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisamplingInfo.sampleShadingEnable = VK_FALSE;
        multisamplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisamplingInfo.minSampleShading = 1.0f; // Optional
        multisamplingInfo.pSampleMask = nullptr; // Optional
        multisamplingInfo.alphaToCoverageEnable = VK_FALSE; // Optional
        multisamplingInfo.alphaToOneEnable = VK_FALSE; // Optional

        // Color blending (disable for now)
        // Config per attached framebuffer
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
        // Global config for all framebuffers. Will disable the blending specified per attached framebuffer
        VkPipelineColorBlendStateCreateInfo colorBlendingInfo{};
        colorBlendingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendingInfo.logicOpEnable = VK_FALSE;
        colorBlendingInfo.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlendingInfo.attachmentCount = 1;
        colorBlendingInfo.pAttachments = &colorBlendAttachment;
        colorBlendingInfo.blendConstants[0] = 0.0f; // Optional
        colorBlendingInfo.blendConstants[1] = 0.0f; // Optional
        colorBlendingInfo.blendConstants[2] = 0.0f; // Optional
        colorBlendingInfo.blendConstants[3] = 0.0f; // Optional

        // Pipeline layout: Specify uniforms
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 0,                // Optional
            .pSetLayouts = nullptr,             // Optional
            .pushConstantRangeCount = 0,        // Optional
            .pPushConstantRanges = nullptr,     // Optional
        };

        res = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to create pipeline layout, error ") + string_VkResult(res)
            );
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{
            // Shader stages
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shaderStages,

            // Fixed-function stages
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportStateInfo,
            .pRasterizationState = &rasterizerInfo,
            .pMultisampleState = &multisamplingInfo,
            .pDepthStencilState = nullptr,  // optional
            .pColorBlendState = &colorBlendingInfo,
            .pDynamicState = &dynamicStateInfo,

            .layout = pipelineLayout,
            .renderPass = renderPass,
            .subpass = 0,  // index of the subpass where this pipeline will be used

            // (Optional) Derive from existing base pipeline
            // Will only be used if VK_PIPELINE_CREATE_DERIVATIVE_BIT is set in .flags
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };

        // Second param: VkPipelineCache for storing and reusing data across multiple calls of vkCreateGraphicsPipelines 
        res = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to create graphics pipeline, error ") + string_VkResult(res)
            );
        }

        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
    }

    void createRenderPass() {
        // Single color buffer attachment
        // - Format as swapchain image format
        // - No supersampling
        // - Clear the buffer before rendering, store result in memory afterward
        // - Image layout should be for presentation in swap chain
        VkAttachmentDescription colorAttachment{
            .format = swapChainImageFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            // Color & depth data handling
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,    // op before rendering: clear framebuffer
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,  // op after rendering: store in memory
            // Stencil data handling (ignore now)

            // Images need to be transformed to specific layouts for 
            // the ops they will be involved next. Ex,
            // - VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Images used as color attachment
            // - VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: Images to be presented in the swap chain
            // - VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: Images to be used as destination for a memory copy operation
            // Don't care about the initial layout
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,     // which layout the image will have before the render pass begins
            // Want image to be ready for presentation using swap chain after render pass
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR  // transition to this layout after render pass finishes
        };

        // Subpasses with attachment references
        VkAttachmentReference colorAttachmentRef{
            .attachment = 0,  // Index in the VkAttachmentDescription array (we only have one attachment)
                              // referenced in `layout(location = 0) out vec4 outColor` in the fragment shader
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  // DURING subpass: Layout attachment as color buffer
        };
        VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,
        };

        // Subpass dependencies (TODO: understand this)
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;  // swap chain (reading from the image)
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };

        VkResult res = vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to create render pass, error ") + string_VkResult(res)
            );
        }

    }

    // shader helpers
    VkShaderModule createShaderModule(const std::vector<char>& shaderSpv) {
        VkShaderModuleCreateInfo shaderModuleCreateInfo {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = shaderSpv.size(),
            .pCode = reinterpret_cast<const uint32_t*>(shaderSpv.data())
        };

        VkShaderModule shaderModule;
        VkResult res = vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule);  // thin wrapper around the byte code
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to create shader module, error ") + string_VkResult(res)
            );
        }

        return shaderModule;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    /* Command buffers */
    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        
        VkCommandPoolCreateInfo commandPoolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,  // re-record command buffer every frame
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
        };

        VkResult res = vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to create command pool, error ") + string_VkResult(res)
            );
        }
    }

    void createCommandBuffer() {
        VkCommandBufferAllocateInfo cmdBufferAllocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        VkResult res = vkAllocateCommandBuffers(device, &cmdBufferAllocInfo, &commandBuffer);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to create command buffer, error ") + string_VkResult(res)
            );
        }
    }

    // Write commands we want to execute into `commandBuffer`
    // `imageIndex`: Index of the current swapchain image we want to write to
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        {   // Begin recording `commandBuffer`
            VkCommandBufferBeginInfo beginInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = 0,                   // optional
                .pInheritanceInfo = nullptr,  // optional, for secondary command buffers
            };
            VkResult res = vkBeginCommandBuffer(commandBuffer, &beginInfo);
            if (res != VK_SUCCESS) {
                throw std::runtime_error(
                    std::string("failed to begin recoring commands, error ") + string_VkResult(res)
                );
            }
        }

        // Command: begin render pass
        VkRenderPassBeginInfo renderPassBeginInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = renderPass,
            .framebuffer = swapChainFramebuffers[imageIndex],
            .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
        };
        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        renderPassBeginInfo.clearValueCount = 1;
        renderPassBeginInfo.pClearValues = &clearColor;  // for VK_ATTACHMENT_LOAD_OP_CLEAR
        vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Command: bind pipeline
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    
        // Command: dynamic viewport and scissor
        VkViewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(swapChainExtent.width),
            .height = static_cast<float>(swapChainExtent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{
            .offset = {0, 0},
            .extent = swapChainExtent,
        };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // Command: DRAW!
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        // Command: end render pass
        vkCmdEndRenderPass(commandBuffer);
        
        {   // Finish recording command buffer
            VkResult res = vkEndCommandBuffer(commandBuffer);
            if (res != VK_SUCCESS) {
                throw std::runtime_error(
                    std::string("failed to record command buffer, error ") + string_VkResult(res)
                );
            }
        }
    }

    void createSyncPrimitives() {
        VkSemaphoreCreateInfo semaphoreInfo{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,  // so that first vkWaitForFences doesn't hang forever
        };

        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization primitives!");
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
