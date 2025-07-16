#ifndef WIRECELL_SPNG_RAYTEST
#define WIRECELL_SPNG_RAYTEST

#include <map>
#include <string>
#include <torch/torch.h>

namespace WireCell::Spng::RayGrid {

    // Example data members dumped from the Python version of Coordinates for a
    // microboone-like detector.
    extern const std::map<std::string, torch::Tensor> ray_grid_coordinates_data;

} // namespace

#endif
