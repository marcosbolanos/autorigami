#include "autorigami/simulation.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <PxPhysicsAPI.h>

namespace autorigami {

namespace {

using physx::PxFilterData;
using physx::PxFilterFlag;
using physx::PxFilterFlags;
using physx::PxFilterObjectAttributes;
using physx::PxPairFlag;
using physx::PxPairFlags;
using physx::PxReal;
using physx::PxTransform;
using physx::PxU32;
using physx::PxVec3;

constexpr PxU32 vertex_tag = 1;

/// One PhysX foundation may exist per process, so every simulation shares it.
struct PhysicsContext {
    physx::PxDefaultAllocator allocator;
    physx::PxDefaultErrorCallback error_callback;
    physx::PxFoundation* foundation;
    physx::PxPhysics* physics;

    PhysicsContext() {
        foundation = PxCreateFoundation(
            PX_PHYSICS_VERSION, allocator, error_callback
        );
        if (foundation == nullptr) {
            throw std::runtime_error("PhysX foundation creation failed");
        }
        physics = PxCreatePhysics(
            PX_PHYSICS_VERSION, *foundation, physx::PxTolerancesScale()
        );
        if (physics == nullptr || !PxInitExtensions(*physics, nullptr)) {
            throw std::runtime_error("PhysX initialization failed");
        }
    }
};

PhysicsContext& physics_context() {
    // Intentionally never destroyed: PhysX objects must outlive every scene.
    static auto* context = new PhysicsContext();
    return *context;
}

/// Discard contacts between vertices that are neighbors along the curve.
///
/// `word0` is the vertex index, `word1` marks curve vertices and `word2` is
/// the curve index. The constant block holds the largest index difference
/// that never collides. Vertices of different curves always collide.
PxFilterFlags curve_filter_shader(
    PxFilterObjectAttributes /*attributes0*/,
    PxFilterData data0,
    PxFilterObjectAttributes /*attributes1*/,
    PxFilterData data1,
    PxPairFlags& pair_flags,
    const void* constant_block,
    PxU32 /*constant_block_size*/
) {
    if (data0.word1 == vertex_tag && data1.word1 == vertex_tag
        && data0.word2 == data1.word2) {
        const PxU32 ignored_adjacent_vertices =
            *static_cast<const PxU32*>(constant_block);
        const PxU32 index_difference = data0.word0 > data1.word0
            ? data0.word0 - data1.word0
            : data1.word0 - data0.word0;
        if (index_difference <= ignored_adjacent_vertices) {
            return PxFilterFlag::eKILL;
        }
    }
    pair_flags = PxPairFlag::eCONTACT_DEFAULT;
    return PxFilterFlag::eDEFAULT;
}

}  // namespace

struct AxialGravitySimulation::State {
    physx::PxDefaultCpuDispatcher* dispatcher = nullptr;
    physx::PxScene* scene = nullptr;
    physx::PxMaterial* material = nullptr;
    physx::PxRigidStatic* floor = nullptr;
    std::vector<physx::PxRigidDynamic*> vertices;
    std::vector<physx::PxJoint*> joints;
    PxU32 ignored_adjacent_vertices = 0;

    ~State() {
        // Joints refer to actors, so release the joints first.
        for (auto* joint : joints) {
            joint->release();
        }
        for (auto* vertex : vertices) {
            vertex->release();
        }
        if (floor != nullptr) {
            floor->release();
        }
        if (scene != nullptr) {
            scene->release();
        }
        if (material != nullptr) {
            material->release();
        }
        if (dispatcher != nullptr) {
            dispatcher->release();
        }
    }
};

AxialGravitySimulation::AxialGravitySimulation(
    const PointMatrix& points,
    const PinnedMask& pinned,
    const CurveIndices& curve_indices,
    const AxialGravityParameters& parameters
) : state_(std::make_unique<State>()) {
    if (points.rows() < 2 || pinned.size() != points.rows()
        || curve_indices.size() != points.rows()) {
        throw std::invalid_argument(
            "a curve needs at least two vertices, and one pin flag and one "
            "curve index per vertex"
        );
    }
    if (parameters.collision_radius <= 0.0F
        || parameters.tension_stiffness < 0.0F
        || parameters.thread_count == 0) {
        throw std::invalid_argument("invalid axial gravity parameters");
    }

    auto& physics = *physics_context().physics;
    state_->ignored_adjacent_vertices =
        static_cast<PxU32>(parameters.ignored_adjacent_vertices);
    state_->dispatcher =
        physx::PxDefaultCpuDispatcherCreate(parameters.thread_count);

    physx::PxSceneDesc description(physics.getTolerancesScale());
    description.gravity = PxVec3(0.0F, 0.0F, -parameters.gravity);
    description.cpuDispatcher = state_->dispatcher;
    description.filterShader = curve_filter_shader;
    description.filterShaderData = &state_->ignored_adjacent_vertices;
    description.filterShaderDataSize = sizeof(PxU32);
    description.solverType = physx::PxSolverType::eTGS;
    description.broadPhaseType = physx::PxBroadPhaseType::ePABP;
    state_->scene = physics.createScene(description);
    if (state_->scene == nullptr) {
        throw std::runtime_error("PhysX scene creation failed");
    }
    // Frictionless and without restitution: contacts only stop the fall.
    state_->material = physics.createMaterial(0.0F, 0.0F, 0.0F);
    state_->floor = physx::PxCreatePlane(
        physics,
        physx::PxPlane(0.0F, 0.0F, 1.0F, -parameters.floor_height),
        *state_->material
    );
    state_->scene->addActor(*state_->floor);

    const auto vertex_count = static_cast<std::size_t>(points.rows());
    const physx::PxSphereGeometry sphere(parameters.collision_radius);
    const physx::PxRigidDynamicLockFlags rail =
        physx::PxRigidDynamicLockFlag::eLOCK_LINEAR_X
        | physx::PxRigidDynamicLockFlag::eLOCK_LINEAR_Y
        | physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_X
        | physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y
        | physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z;
    state_->vertices.reserve(vertex_count);
    for (std::size_t index = 0; index < vertex_count; ++index) {
        const auto row = static_cast<Eigen::Index>(index);
        auto* vertex = physics.createRigidDynamic(PxTransform(
            PxVec3(points(row, 0), points(row, 1), points(row, 2))
        ));
        auto* shape = physx::PxRigidActorExt::createExclusiveShape(
            *vertex, sphere, *state_->material
        );
        shape->setSimulationFilterData(
            PxFilterData(
                static_cast<PxU32>(index), vertex_tag, curve_indices[row], 0
            )
        );
        vertex->setMass(1.0F);
        if (pinned[row]) {
            vertex->setRigidBodyFlag(physx::PxRigidBodyFlag::eKINEMATIC, true);
        }
        vertex->setRigidDynamicLockFlags(rail);
        vertex->setLinearDamping(parameters.linear_damping);
        vertex->setMaxDepenetrationVelocity(
            parameters.maximum_depenetration_velocity
        );
        vertex->setMaxLinearVelocity(parameters.maximum_linear_velocity);
        vertex->setSolverIterationCounts(
            parameters.position_iterations, parameters.velocity_iterations
        );
        // A slowly settling curve must not be frozen by the sleep system.
        vertex->setSleepThreshold(0.0F);
        state_->scene->addActor(*vertex);
        state_->vertices.push_back(vertex);
    }

    // A spring on the height difference of consecutive vertices acts as
    // string tension: it keeps the height profile smooth between supports.
    const PxTransform center(physx::PxIdentity);
    const physx::PxD6JointDrive tension(
        parameters.tension_stiffness,
        parameters.tension_damping,
        PX_MAX_F32,
        false
    );
    state_->joints.reserve(vertex_count - 1);
    for (std::size_t index = 0; index + 1 < vertex_count; ++index) {
        const auto row = static_cast<Eigen::Index>(index);
        if (pinned[row] && pinned[row + 1]) {
            continue;  // Both vertices are immovable.
        }
        if (curve_indices[row] != curve_indices[row + 1]) {
            continue;  // The end of one curve and the start of the next.
        }
        auto* joint = physx::PxD6JointCreate(
            physics,
            state_->vertices[index],
            center,
            state_->vertices[index + 1],
            center
        );
        for (const auto axis : { physx::PxD6Axis::eX,
                                 physx::PxD6Axis::eY,
                                 physx::PxD6Axis::eZ,
                                 physx::PxD6Axis::eTWIST,
                                 physx::PxD6Axis::eSWING1,
                                 physx::PxD6Axis::eSWING2 }) {
            joint->setMotion(axis, physx::PxD6Motion::eFREE);
        }
        joint->setDrive(physx::PxD6Drive::eZ, tension);
        state_->joints.push_back(joint);
    }
}

AxialGravitySimulation::~AxialGravitySimulation() = default;

void AxialGravitySimulation::step(
    const std::size_t step_count,
    const float time_step
) {
    if (time_step <= 0.0F) {
        throw std::invalid_argument("time_step must be positive");
    }
    for (std::size_t step = 0; step < step_count; ++step) {
        state_->scene->simulate(time_step);
        state_->scene->fetchResults(true);
    }
}

void AxialGravitySimulation::set_gravity(const float gravity) {
    state_->scene->setGravity(PxVec3(0.0F, 0.0F, -gravity));
}

PointMatrix AxialGravitySimulation::points() const {
    PointMatrix output(static_cast<Eigen::Index>(state_->vertices.size()), 3);
    for (std::size_t index = 0; index < state_->vertices.size(); ++index) {
        const PxVec3 position = state_->vertices[index]->getGlobalPose().p;
        output.row(static_cast<Eigen::Index>(index))
            << position.x, position.y, position.z;
    }
    return output;
}

float AxialGravitySimulation::maximum_speed() const {
    PxReal maximum = 0.0F;
    for (const auto* vertex : state_->vertices) {
        maximum = std::max(maximum, vertex->getLinearVelocity().magnitude());
    }
    return maximum;
}

}  // namespace autorigami
