// RayGrid.cpp
#include "RayGrid.h"

namespace WireCell {
namespace Spng {
namespace RayGrid {

// --- Helper functions from funcs.py ---

torch::Tensor pitch(const torch::Tensor& r0, const torch::Tensor& r1) {
    // rdir = r0[1] - r0[0]
    torch::Tensor rdir = r0.index({1}) - r0.index({0});
    // uperp = torch.tensor([-rdir[1], rdir[0]]) / torch.norm(rdir)
    torch::Tensor uperp = torch::tensor({-rdir.index({1}).item<double>(), rdir.index({0}).item<double>()}, torch::kDouble) / torch::norm(rdir);
    // cvec = r1[0]-r0[0]
    torch::Tensor cvec = r1.index({0}) - r0.index({0});
    // pdist = torch.dot(cvec, uperp)
    torch::Tensor pdist = torch::dot(cvec, uperp);
    return pdist * uperp;
}

torch::Tensor crossing(const torch::Tensor& r0, const torch::Tensor& r1) {
    torch::Tensor p1 = r0.index({0});
    torch::Tensor p2 = r0.index({1});
    torch::Tensor p3 = r1.index({0});
    torch::Tensor p4 = r1.index({1});

    double x1 = p1.index({0}).item<double>();
    double y1 = p1.index({1}).item<double>();
    double x2 = p2.index({0}).item<double>();
    double y2 = p2.index({1}).item<double>();
    double x3 = p3.index({0}).item<double>();
    double y3 = p3.index({1}).item<double>();
    double x4 = p4.index({0}).item<double>();
    double y4 = p4.index({1}).item<double>();

    double denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

    if (torch::isclose(torch::tensor(denominator), torch::tensor(0.0)).item<bool>()) {
        throw std::runtime_error("parallel lines do not cross");
    }

    double t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    torch::Tensor t = torch::tensor(t_numerator / denominator, torch::kDouble);

    torch::Tensor intersection_point = p1 + t * (p2 - p1);
    return intersection_point;
}

torch::Tensor vector(const torch::Tensor& ray) {
    return ray.index({1}) - ray.index({0});
}

torch::Tensor ray_direction(const torch::Tensor& ray) {
    torch::Tensor n = vector(ray);
    torch::Tensor d = torch::linalg_norm(n);
    // Handle division by zero for zero-length vectors
    if (d.item<double>() == 0.0) {
        return torch::zeros_like(n);
    }
    return n / d;
}


// --- Coordinates class implementation ---

Coordinates::Coordinates(const torch::Tensor& views) {
    init(views);
}

long Coordinates::nviews() const {
    return pitch_mag.size(0);
}

torch::Tensor Coordinates::bounding_box() const {
    // Python: if self.pitch_dir[0,0] == 0:
    // C++: self.pitch_dir.index({0, 0}).item<double>() == 0.0
    if (pitch_dir.index({0, 0}).item<double>() == 0.0) { // points up, hbounds is view 0
        torch::Tensor x0 = center.index({1, 0});
        torch::Tensor y0 = center.index({0, 1});
        torch::Tensor x1 = x0 + pitch_mag.index({1});
        torch::Tensor y1 = y0 + pitch_mag.index({0});
        return torch::stack({torch::stack({x0, x1}), torch::stack({y0, y1})});
    } else {
        torch::Tensor x0 = center.index({0, 1});
        torch::Tensor y0 = center.index({1, 0});
        torch::Tensor x1 = x0 + pitch_mag.index({0});
        torch::Tensor y1 = y0 + pitch_mag.index({1});
        return torch::stack({torch::stack({x0, x1}), torch::stack({y0, y1})});
    }
}

torch::Tensor Coordinates::point_pitches(const torch::Tensor& points) const {
    if (points.dim() != 2 || points.size(1) != 2) {
        throw std::invalid_argument("Input 'points' must be a tensor of shape (nbatch, 2).");
    }

    // (nbatch, 2)
    // long nbatch = points.size(0);

    // Reshape points to (nbatch, 1, 2) for broadcasting with view data
    // (nbatch, 1, 2)
    torch::Tensor points_reshaped = points.unsqueeze(1);

    // Reshape center to (1, Nview, 2) for broadcasting with points
    // (1, Nview, 2)
    torch::Tensor center_reshaped = center.unsqueeze(0);

    // Calculate vector from each view's center to each point
    // Result will be (nbatch, Nview, 2)
    // (nbatch, 1, 2) - (1, Nview, 2) -> (nbatch, Nview, 2)
    torch::Tensor vec_center_to_point = points_reshaped - center_reshaped;

    // Reshape pitch_dir to (1, Nview, 2) for broadcasting
    // (1, Nview, 2)
    torch::Tensor pitch_dir_reshaped = pitch_dir.unsqueeze(0);

    // Calculate the dot product along the last dimension to get pitches.
    // This is a batched dot product: sum((nbatch, Nview, 2) * (1, Nview, 2)) over dim 2
    // Result will be (nbatch, Nview)
    torch::Tensor pitches = torch::sum(vec_center_to_point * pitch_dir_reshaped, /*dim=*/2);

    return pitches;
}

torch::Tensor Coordinates::point_indices(const torch::Tensor& points) const {
    // Calculate the floating-point pitch locations for each point in each view
    // (nbatch, nview)
    torch::Tensor pitches_per_view = this->point_pitches(points);

    // Get nviews from the calculated pitches_per_view (or self.pitch_mag)
    // long nviews_val = pitches_per_view.size(1);

    // Reshape pitch_mag to (1, Nview) for broadcasting with pitches_per_view
    // (nbatch, Nview) / (1, Nview) -> (nbatch, Nview)
    // Note: self.pitch_mag is (Nview,)
    torch::Tensor pitch_mag_reshaped = pitch_mag.unsqueeze(0);

    // Calculate pitch indices using the existing pitch_index logic: floor(pitch / pitch_mag)
    // (nbatch, nview)
    torch::Tensor indices_float = pitches_per_view / pitch_mag_reshaped;

    // Apply floor and convert to long integer type
    // (nbatch, nview)
    torch::Tensor pitch_indices = torch::floor(indices_float).to(torch::kLong);

    return pitch_indices;
}

torch::Tensor Coordinates::ray_crossing(const torch::Tensor& one, const torch::Tensor& two) const {
    // one and two can be scalar (2,) or batched (nbatch, 2)
    // view1, ray1 = one
    // view2, ray2 = two

    // Extract view and ray indices. Use .select(dim, index) for single dimension access.
    // For batched input, this will broadcast correctly.
    torch::Tensor view1 = one.select(-1, 0);
    torch::Tensor ray1 = one.select(-1, 1);
    torch::Tensor view2 = two.select(-1, 0);
    torch::Tensor ray2 = two.select(-1, 1);

    // Ensure indices are long for indexing
    view1 = view1.to(torch::kLong);
    ray1 = ray1.to(torch::kLong);
    view2 = view2.to(torch::kLong);
    ray2 = ray2.to(torch::kLong);

    // r00 = self.zero_crossings[view1, view2]
    // w12 = self.ray_jump[view1, view2]
    // w21 = self.ray_jump[view2, view1]

    // Use advanced indexing. If view1/view2 are scalar, this will be scalar.
    // If view1/view2 are 1D tensors, this will result in a batched tensor.
    torch::Tensor r00 = zero_crossings.index({view1, view2});
    torch::Tensor w12 = ray_jump.index({view1, view2});
    torch::Tensor w21 = ray_jump.index({view2, view1});

    // return r00 + ray2 * w12 + ray1 * w21;
    // Need to unsqueeze ray1 and ray2 to match the dimensions of w12, w21, r00 for broadcasting
    // If r00, w12, w21 are (..., 2), then ray1/ray2 need to be (..., 1)
    if (ray1.dim() < r00.dim() - 1) { // Check if ray1/ray2 are scalar or 1D when r00 is 2D or 3D
        ray1 = ray1.unsqueeze(-1);
        ray2 = ray2.unsqueeze(-1);
    }
    return r00 + ray2.to(r00.dtype()) * w12 + ray1.to(r00.dtype()) * w21;
}

torch::Tensor Coordinates::pitch_location(const torch::Tensor& one, const torch::Tensor& two,
                                         const torch::Tensor& view) const {
    // view1, ray1 = one
    // view2, ray2 = two
    torch::Tensor view1 = one.select(-1, 0);
    torch::Tensor ray1 = one.select(-1, 1);
    torch::Tensor view2 = two.select(-1, 0);
    torch::Tensor ray2 = two.select(-1, 1);

    // Ensure indices are long for indexing
    view1 = view1.to(torch::kLong);
    ray1 = ray1.to(torch::kLong);
    view2 = view2.to(torch::kLong);
    ray2 = ray2.to(torch::kLong);
    torch::Tensor k_view = view.to(torch::kLong); // The 'view' argument corresponds to 'ik' in Python's 'a' and 'b' tensors

    // return self.b[view1, view2, view] 
    //     + ray2 * self.a[view1, view2, view] 
    //     + ray1 * self.a[view2, view1, view]

    // Use advanced indexing.
    // If view1/view2/k_view are scalar, this will be scalar.
    // If view1/view2/k_view are 1D tensors, this will result in a batched tensor.
    torch::Tensor b_val = b.index({view1, view2, k_view});
    torch::Tensor a_val_12 = a.index({view1, view2, k_view});
    torch::Tensor a_val_21 = a.index({view2, view1, k_view});

    // Ensure ray1 and ray2 are broadcastable with the result of indexing
    // If b_val, a_val_12, a_val_21 are scalar, ray1/ray2 should remain scalar.
    // If they are 1D, ray1/ray2 should remain 1D.
    return b_val + ray2.to(b_val.dtype()) * a_val_12 + ray1.to(b_val.dtype()) * a_val_21;
}

torch::Tensor Coordinates::pitch_index(const torch::Tensor& pitch_val, const torch::Tensor& view_idx) const {
    // return torch.floor(pitch/self.pitch_mag[view]).to(torch.long)
    torch::Tensor view_idx_long = view_idx.to(torch::kLong);
    torch::Tensor mag_at_view = pitch_mag.index({view_idx_long});
    return torch::floor(pitch_val / mag_at_view).to(torch::kLong);
}

void Coordinates::init(const torch::Tensor& pitches_in) {
    long nviews_val = pitches_in.size(0);

    // 1D (l) the magnitude of the pitch of view l.
    // pvrel = pitches[:,1,:] - pitches[:,0,:]
    torch::Tensor pvrel = pitches_in.index({torch::indexing::Slice(), 1, torch::indexing::Slice()}) -
                          pitches_in.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});
    // self.pitch_mag = torch.sqrt(pvrel[:,0]**2 + pvrel[:,1]**2)
    pitch_mag = torch::sqrt(torch::pow(pvrel.index({torch::indexing::Slice(), 0}), 2) +
                           torch::pow(pvrel.index({torch::indexing::Slice(), 1}), 2));

    // 2D (l,c) the pitch direction 2D coordinates c of view l.
    // self.pitch_dir = pvrel / self.pitch_mag.reshape(5,1)
    // Assuming nviews is 5 for the reshape, but it should be dynamic
    pitch_dir = pvrel / pitch_mag.reshape({nviews_val, 1});

    // 2D (l,c) the 2D coordinates c of the origin point of view l
    // self.center = pitches[:,0,:]
    center = pitches_in.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});

    // self.ray_dir = torch.vstack((-self.pitch_dir[:,1], self.pitch_dir[:,0])).T
    ray_dir = torch::stack({-pitch_dir.index({torch::indexing::Slice(), 1}),
                            pitch_dir.index({torch::indexing::Slice(), 0})}).transpose(0, 1);

    // ray0 = torch.vstack((self.center - self.ray_dir, self.center + self.ray_dir)).reshape(2,-1,2)
    torch::Tensor ray0_part1 = center - ray_dir;
    torch::Tensor ray0_part2 = center + ray_dir;
    torch::Tensor ray0 = torch::stack({ray0_part1, ray0_part2}).reshape({2, -1, 2});

    // ray1 = torch.vstack((ray0[0] + pvrel, ray0[1] + pvrel)).reshape(2,-1,2)
    torch::Tensor ray1_part1 = ray0.index({0}) + pvrel;
    torch::Tensor ray1_part2 = ray0.index({1}) + pvrel;
    torch::Tensor ray1 = torch::stack({ray1_part1, ray1_part2}).reshape({2, -1, 2});

    // 3D (l,m,c) crossing point 2D coordinates c of "ray 0" of views l and m.
    zero_crossings = torch::zeros({nviews_val, nviews_val, 2}, torch::kDouble);

    // 3D (l,m,c) difference vector coordinates c between two consecutive
    // m-view crossings along l ray direction.  between crossings of rays of view m.
    ray_jump = torch::zeros({nviews_val, nviews_val, 2}, torch::kDouble);

    // The Ray Grid tensor representations.
    a = torch::zeros({nviews_val, nviews_val, nviews_val}, torch::kDouble);
    b = torch::zeros({nviews_val, nviews_val, nviews_val}, torch::kDouble);

    // Cross-view things
    for (long il = 0; il < nviews_val; ++il) {
        torch::Tensor rl0 = ray0.index({torch::indexing::Slice(), il, torch::indexing::Slice()});
        torch::Tensor rl1 = ray1.index({torch::indexing::Slice(), il, torch::indexing::Slice()});

        for (long im = 0; im < nviews_val; ++im) {
            torch::Tensor rm0 = ray0.index({torch::indexing::Slice(), im, torch::indexing::Slice()});
            torch::Tensor rm1 = ray1.index({torch::indexing::Slice(), im, torch::indexing::Slice()});

            // Special case diagonal values
            if (il == im) {
                // self.zero_crossings[il,im] = self.center[il]
                zero_crossings.index_put_({il, im}, center.index({il}));
                // self.ray_jump[il,im] = funcs.ray_direction(rl0)
                ray_jump.index_put_({il, im}, ray_direction(rl0));
                continue;
            }

            if (il < im) {
                // Fill in both triangles in one go to exploit the symmetry of this:
                try {
                    torch::Tensor p = crossing(rl0, rm0);
                    zero_crossings.index_put_({il, im}, p);
                    zero_crossings.index_put_({im, il}, p);
                    ray_jump.index_put_({il, im}, crossing(rl0, rm1) - p);
                    ray_jump.index_put_({im, il}, crossing(rm0, rl1) - p);
                } catch (const std::runtime_error& e) {
                    // Python: print(f'skipping parallel view pair: {il=} {im=}')
                    // In C++, we'll just print to stderr or a log.
                    std::cerr << "skipping parallel view pair: il=" << il << " im=" << im << ": " << e.what() << std::endl;
                    // Continue loop, as in Python's 'continue'
                }
            }
        }
    }

    // Triple layer things
    for (long ik = 0; ik < nviews_val; ++ik) {
        torch::Tensor pk = pitch_dir.index({ik});
        torch::Tensor cp = torch::dot(center.index({ik}), pk);

        for (long il = 0; il < nviews_val; ++il) {
            if (il == ik) {
                continue;
            }

            for (long im = 0; im < il; ++im) {
                if (im == ik) {
                    continue;
                }

                torch::Tensor rlmpk = torch::dot(zero_crossings.index({il, im}), pk);
                torch::Tensor wlmpk = torch::dot(ray_jump.index({il, im}), pk);
                torch::Tensor wmlpk = torch::dot(ray_jump.index({im, il}), pk);

                a.index_put_({il, im, ik}, wlmpk);
                a.index_put_({im, il, ik}, wmlpk);
                b.index_put_({il, im, ik}, rlmpk - cp);
                b.index_put_({im, il, ik}, rlmpk - cp);
            }
        }
    }
}

} // namespace RayGrid
} // namespace Spng
} // namespace WireCell
