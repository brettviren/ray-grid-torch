#include "RayTiling.h"
#include "RayTest.h"
#include "Stopwatch.h"

using namespace WireCell::Spng::RayGrid;

const double pitch_magnitude = 5;
const double gaussian = 3;
const double width = 4000;
const double height = 4000;


int main()
{
    Stopwatch sw;

    auto views = symmetric_views(width, height, pitch_magnitude);
    assert(views.size(0) == 5);
    
    std::cerr << "Made symmetric views in " << sw.restart() << " us, views=\n" << views << "\n";

    Coordinates coords(views);
    std::cerr << "Made coordinates " << sw.restart() << " us\n";

    auto points = random_groups(1000, 10, gaussian, {0.0, 0.0}, {width, height});
    std::cerr << "Made points in " << sw.restart() << " us, points.shape=" << points.sizes() << "\n";

    auto activities = fill_activity(coords, points);
    assert(activities.size() == 5);

    std::cerr << "Made activities in " << sw.restart() << " us\n";

    const size_t ntries = 1001;
    for (size_t tries = 0; tries < ntries; ++tries) {
        auto blobs = trivial_blobs();
        if (!tries) {
            std::cerr << "trivial has " << blobs.size(0) << " blobs\n";
        }
        for (size_t view = 2; view < 5; ++view) {
            blobs = apply_activity(coords, blobs, activities[view]);
            if (!tries) {
                std::cerr << "view " << view << " has " << blobs.size(0) << " blobs\n";
            }
            assert (blobs.size(0) > 0);
        }            
        if (!tries) {
            std::cerr << "Made " << blobs.size(0) << " blobs in " << sw.restart() << " us\n";
        }
    }
    double us = sw.restart();

    std::cerr << "Repeated " << ntries-1 << " in " << us << " us, " << ((ntries-1) / (1e-6*us)) <<  " Hz\n";

    return 0;
}
