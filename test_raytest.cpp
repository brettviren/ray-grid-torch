#include "RayTest.h"
#include <iostream>
#include <vector>
#include <iomanip> // For std::fixed, std::setprecision

int main() {
    std::cout << "Starting symmetric_views test...\n";

    // Set output precision for better readability of double tensors
    // std::cout << std::fixed << std::setprecision(4);

    // --- Test 1: Basic functionality with default parameters (CPU) ---
    std::cout << "\n--- Test 1: CPU Tensors (Default Parameters) ---\n";
    torch::Tensor pitches1 = WireCell::Spng::RayGrid::symmetric_views();

    std::cout << "\nGenerated pitches (shape " << pitches1.sizes() << ", device "
              << pitches1.device() << ", dtype " << pitches1.dtype() << "):\n"
              << pitches1 << std::endl;

    // --- Test 2: Custom parameters (CPU) ---
    std::cout << "\n--- Test 2: CPU Tensors (Custom Parameters) ---\n";
    double custom_width = 200.0;
    double custom_height = 150.0;
    double custom_pitch_mag = 5.0;
    double custom_angle = 0.7853981633974483; // 45 degrees

    torch::Tensor pitches2 = WireCell::Spng::RayGrid::symmetric_views(
        custom_width, custom_height, custom_pitch_mag, custom_angle);

    std::cout << "\nGenerated pitches (custom parameters, shape " << pitches2.sizes() << ", device "
              << pitches2.device() << ", dtype " << pitches2.dtype() << "):\n"
              << pitches2 << std::endl;

    // --- Test 3: Functionality with CUDA tensors (if available) ---
    if (torch::cuda::is_available()) {
        std::cout << "\n--- Test 3: CUDA Tensors ---\n";
        torch::Device cuda_device(torch::kCUDA);
        torch::TensorOptions cuda_options = torch::TensorOptions().device(cuda_device);

        torch::Tensor pitches_cuda = WireCell::Spng::RayGrid::symmetric_views(
            custom_width, custom_height, custom_pitch_mag, custom_angle, cuda_options);

        std::cout << "\nGenerated pitches (CUDA) shape: " << pitches_cuda.sizes() << ", device "
                  << pitches_cuda.device() << ", dtype " << pitches_cuda.dtype() << ":\n"
                  << pitches_cuda << std::endl;

    } else {
        std::cout << "\nCUDA not available. Skipping CUDA tests for symmetric_views.\n";
    }

    std::cout << "\nsymmetric_views test finished.\n";
    return 0;
}
