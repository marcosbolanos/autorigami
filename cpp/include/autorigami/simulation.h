#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <Eigen/Core>

namespace autorigami {

using PointMatrix = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using PinnedMask = Eigen::Array<bool, Eigen::Dynamic, 1>;
using CurveIndices = Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>;

/// Rigid-body model of centerlines that can move along z only.
///
/// Each vertex is one sphere on a vertical rail: its x and y never change.
/// Lengths are in the units of the input polyline (nm in this project).
struct AxialGravityParameters {
    /// Sphere radius. Two vertices touch when they are `2 * radius` apart.
    float collision_radius;
    /// Vertex pairs of one curve whose index difference is at most this never
    /// collide. Vertices of different curves always collide.
    std::size_t ignored_adjacent_vertices;
    /// Stiffness of the spring on the height difference of consecutive
    /// vertices of one curve. It acts as string tension along the curve.
    float tension_stiffness;
    float tension_damping;
    /// Height of the horizontal floor that supports the curve.
    float floor_height;
    /// Acceleration along negative z.
    float gravity;
    float linear_damping;
    /// Largest speed at which the solver separates overlapping spheres.
    float maximum_depenetration_velocity;
    /// Largest speed of any sphere.
    float maximum_linear_velocity;
    unsigned int position_iterations;
    unsigned int velocity_iterations;
    unsigned int thread_count;
};

/// A PhysX scene in which centerlines fall along z until contacts stop them.
class AxialGravitySimulation {
public:
    /// `points` holds the vertices of all curves, one curve after the other.
    /// `pinned[i]` is true for a vertex that never moves. Pinned vertices
    /// support the others. `curve_indices[i]` is the curve of vertex `i`.
    AxialGravitySimulation(
        const PointMatrix& points,
        const PinnedMask& pinned,
        const CurveIndices& curve_indices,
        const AxialGravityParameters& parameters
    );
    ~AxialGravitySimulation();
    AxialGravitySimulation(const AxialGravitySimulation&) = delete;
    AxialGravitySimulation& operator=(const AxialGravitySimulation&) = delete;

    /// Advance the scene by `step_count` steps of `time_step` each.
    void step(std::size_t step_count, float time_step);

    /// Change the acceleration along negative z.
    void set_gravity(float gravity);

    /// Current vertex positions, one row per input vertex.
    [[nodiscard]] PointMatrix points() const;

    /// Largest speed of any vertex.
    [[nodiscard]] float maximum_speed() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace autorigami
