#include "msdfgen.h"
#include <cstdint>
#include <memory>

struct Vector2 {
    double x;
    double y;

    bool operator==(const Vector2 &rhs) const noexcept;
    bool operator!=(const Vector2 &rhs) const noexcept;
};

std::unique_ptr<msdfgen::Shape> create_shape() {
    return std::make_unique<msdfgen::Shape>();
}

bool contour_is_edges_empty(msdfgen::Contour const* contour) {
    return contour->edges.empty();
}

void contour_add_edge2(msdfgen::Contour* contour, Vector2 p0, Vector2 p1) {
    contour->addEdge(msdfgen::EdgeHolder(*(msdfgen::Vector2*)&p0,*(msdfgen::Vector2*)&p1));
}

void contour_add_edge3(msdfgen::Contour* contour, Vector2 p0, Vector2 p1, Vector2 p2) {
    contour->addEdge(msdfgen::EdgeHolder(*(msdfgen::Vector2*)&p0,*(msdfgen::Vector2*)&p1, *(msdfgen::Vector2*)&p2));
}

void contour_add_edge4(msdfgen::Contour* contour, Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3) {
    contour->addEdge(msdfgen::EdgeHolder(*(msdfgen::Vector2*)&p0,*(msdfgen::Vector2*)&p1, *(msdfgen::Vector2*)&p2, *(msdfgen::Vector2*)&p3));
}

void shape_check_last_contour(msdfgen::Shape& shape) {
    if (!shape.contours.empty() && shape.contours.back().edges.empty()) {
        shape.contours.pop_back();
    }
}

void generateMSDF(float* data, uint32_t width, uint32_t height, Vector2 offset, msdfgen::Shape const& shape, double range) {
    msdfgen::BitmapRef<float, 3> ref(data, width, height);
    msdfgen::Projection proj(msdfgen::Vector2(1.0), msdfgen::Vector2(offset.x, offset.y));
    msdfgen::generateMSDF(ref, shape, proj, range);
}
