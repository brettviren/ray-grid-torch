#ifndef WIRECELL_SPNG_RAYTEST
#define WIRECELL_SPNG_RAYTEST

#include <map>
#include <string>
#include <torch/torch.h>

namespace WireCell::Spng::RayGrid {

    // Example data members dumped from the Python version of Coordinates for a
    // microboone-like detector.
    extern const std::map<std::string, torch::Tensor> ray_grid_coordinates_data;


    // Helper for comparing tensors
    bool are_tensors_close(const torch::Tensor& a, const torch::Tensor& b, double rtol = 1e-05, double atol = 1e-08);

} // namespace

#endif
