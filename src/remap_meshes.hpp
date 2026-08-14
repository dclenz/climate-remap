#ifndef MFA_REMAP_MESHES_HPP
#define MFA_REMAP_MESHES_HPP

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <diy/thirdparty/fmt/format.h>
#include <mfa/types.hpp>
#include <moab/Core.hpp>
#include <moab/ParallelComm.hpp>

// Owns the independent source and target MOAB databases used by a remap.
// Separate databases are required because the input files may define the same
// tag name with incompatible types or component counts.
template <typename T>
class RemapMeshes
{
    static_assert(std::is_same_v<T, double>,
                  "Climate remapping expects double-precision MOAB field tags");

  public:
    static constexpr int GeometryDimension = 3;

    explicit RemapMeshes(MPI_Comm communicator) :
        sourceMb_(std::make_unique<moab::Core>()),
        sourcePc_(std::make_unique<moab::ParallelComm>(sourceMb_.get(), communicator)),
        targetMb_(std::make_unique<moab::Core>()),
        targetPc_(std::make_unique<moab::ParallelComm>(targetMb_.get(), communicator))
    {}

    RemapMeshes(const RemapMeshes&) = delete;
    RemapMeshes& operator=(const RemapMeshes&) = delete;

    moab::Interface& source() { return *sourceMb_; }
    const moab::Interface& source() const { return *sourceMb_; }
    moab::Interface& target() { return *targetMb_; }
    const moab::Interface& target() const { return *targetMb_; }

    const moab::EntityHandle& sourceFileSet() const { return sourceFileSet_; }
    const moab::EntityHandle& targetFileSet() const { return targetFileSet_; }

    const moab::Range& sourceVertices() const { return sourceVertices_; }
    const moab::Range& sourceElements() const { return sourceElements_; }
    const moab::Range& targetVertices() const { return targetVertices_; }
    const moab::Range& targetElements() const { return targetElements_; }

    const std::map<moab::EntityHandle, int>& sourceElementIndices() const
    {
        return sourceElementIndices_;
    }

    enum class Database { Source, Target };

    moab::Tag requiredTag(Database database, const std::string& name)
    {
        moab::Tag tag;
        require(interface(database).tag_get_handle(name.c_str(), tag),
                "get " + name + " tag");
        return tag;
    }

    moab::Tag sourceTag(const std::string& name) { return requiredTag(Database::Source, name); }
    moab::Tag targetTag(const std::string& name) { return requiredTag(Database::Target, name); }

    int tagLength(Database database, moab::Tag tag, const std::string& name)
    {
        int length = 0;
        require(interface(database).tag_get_length(tag, length),
                "read " + name + " tag length");
        if (length <= 0)
            fail(name + " tag has no components");
        return length;
    }

    template <typename Value>
    std::vector<Value> readTag(Database database, moab::Tag tag, const moab::Range& entities,
                                int length, const std::string& name)
    {
        const int actualLength = tagLength(database, tag, name);
        if (length != actualLength)
            fail(name + " tag has unexpected component count " + std::to_string(actualLength) +
                 "; expected " + std::to_string(length));
        moab::DataType actualType;
        require(interface(database).tag_get_data_type(tag, actualType),
                "read " + name + " tag type");
        const moab::DataType expectedType = std::is_integral_v<Value> ?
            moab::MB_TYPE_INTEGER : moab::MB_TYPE_DOUBLE;
        if (actualType != expectedType)
            fail(name + " tag has an incompatible data type");
        std::vector<Value> values(entities.size() * length);

        require(interface(database).tag_get_data(tag, entities, values.data()),
                "read " + name + " tag values");
        return values;
    }

    moab::Tag createDenseTag(Database database, const std::string& name, int length,
                             moab::DataType type)
    {
        if (length <= 0)
            fail("create " + name + " tag with invalid length");
        moab::Tag tag;
        require(interface(database).tag_get_handle(
                    name.c_str(), length, type, tag, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT),
                "create " + name + " tag");
        return tag;
    }

    template <typename Value>
    void writeTag(Database database, moab::Tag tag, const moab::Range& entities,
                  const Value* values, const std::string& name)
    {
        require(interface(database).tag_set_data(tag, entities, values),
                "write " + name + " tag values");
    }

    void writeMesh(Database database, const std::string& filename,
                   const moab::EntityHandle& meshset)
    {
        require(interface(database).write_mesh(filename.c_str(), &meshset, 1),
                "write mesh file '" + filename + "'");
    }

    void loadSource(const std::string& filename)
    {
        if (sourceLoaded_)
            fail("load a second source mesh");
        fmt::print(stderr, "RemapMeshes: Reading source file {}\n", filename);
        require(source().create_meshset(moab::MESHSET_SET, sourceFileSet_),
                "create source file set");
        require(source().load_file(filename.c_str(), &sourceFileSet_),
                "load source file");
        require(source().get_entities_by_type(sourceFileSet_, moab::MBVERTEX, sourceVertices_),
                "read source vertices");
        require(source().get_entities_by_dimension(sourceFileSet_, 2, sourceElements_),
                "read source elements");

        sourceCoordinates_.resize(sourceVertices_.size() * GeometryDimension);
        require(source().get_coords(sourceVertices_, sourceCoordinates_.data()),
                "read source coordinates");

        sourceElementIndices_.clear();
        int index = 0;
        for (moab::EntityHandle element : sourceElements_)
            sourceElementIndices_[element] = index++;
        sourceLoaded_ = true;
    }

    void loadTarget(const std::string& filename)
    {
        if (targetLoaded_)
            fail("load a second target mesh");
        fmt::print(stderr, "RemapMeshes: Reading target file {}\n", filename);
        require(target().create_meshset(moab::MESHSET_SET, targetFileSet_),
                "create target file set");
        require(target().load_file(filename.c_str(), &targetFileSet_),
                "load target file");
        require(target().get_entities_by_type(targetFileSet_, moab::MBVERTEX, targetVertices_),
                "read target vertices");

        targetCoordinates_.resize(targetVertices_.size() * GeometryDimension);
        require(target().get_coords(targetVertices_, targetCoordinates_.data()),
                "read target coordinates");

        moab::Range allTargetElements;
        require(target().get_entities_by_dimension(targetFileSet_, 2, allTargetElements),
                "read target elements");

        moab::Tag maskTag = targetTag("ROMS_Mask");
        const int maskComponents = tagLength(Database::Target, maskTag, "ROMS_Mask");
        std::vector<int> mask = readTag<int>(
            Database::Target, maskTag, allTargetElements, maskComponents, "ROMS_Mask");

        targetElements_.clear();
        for (size_t index = 0; index < allTargetElements.size(); ++index)
        {
            if (mask[index * maskComponents] == 1)
                targetElements_.insert(allTargetElements[index]);
        }
        fmt::print(stderr, "Initial target elements: {}, ROMS_Mask==1 target elements: {}\n",
                   allTargetElements.size(), targetElements_.size());
        if (targetElements_.empty())
            fail("select any wet target elements");
        targetLoaded_ = true;
    }

    VectorX<T> sourceVertexMins() const { return coordinateBounds(sourceCoordinates_).first; }
    VectorX<T> sourceVertexMaxs() const { return coordinateBounds(sourceCoordinates_).second; }
    VectorX<T> targetVertexMins() const { return coordinateBounds(targetCoordinates_).first; }
    VectorX<T> targetVertexMaxs() const { return coordinateBounds(targetCoordinates_).second; }

  private:
    moab::Interface& interface(Database database)
    {
        return database == Database::Source ? source() : target();
    }

    static void fail(const std::string& operation)
    {
        fmt::print(stderr, "ERROR: RemapMeshes failed to {}.\n", operation);
        std::exit(1);
    }

    static void require(moab::ErrorCode result, const std::string& operation)
    {
        if (result != moab::MB_SUCCESS)
            fail(operation);
    }

    static std::pair<VectorX<T>, VectorX<T>> coordinateBounds(const std::vector<T>& coordinates)
    {
        if (coordinates.size() < GeometryDimension)
            fail("compute bounds for an empty mesh");

        VectorX<T> mins(GeometryDimension);
        VectorX<T> maxs(GeometryDimension);
        for (int dimension = 0; dimension < GeometryDimension; ++dimension)
            mins(dimension) = maxs(dimension) = coordinates[dimension];

        for (size_t index = 0; index < coordinates.size() / GeometryDimension; ++index)
        {
            for (int dimension = 0; dimension < GeometryDimension; ++dimension)
            {
                const T value = coordinates[GeometryDimension * index + dimension];
                mins(dimension) = std::min(mins(dimension), value);
                maxs(dimension) = std::max(maxs(dimension), value);
            }
        }
        return {mins, maxs};
    }

    std::unique_ptr<moab::Core> sourceMb_;
    std::unique_ptr<moab::ParallelComm> sourcePc_;
    std::unique_ptr<moab::Core> targetMb_;
    std::unique_ptr<moab::ParallelComm> targetPc_;

    moab::EntityHandle sourceFileSet_{0};
    moab::EntityHandle targetFileSet_{0};
    moab::Range sourceVertices_;
    moab::Range sourceElements_;
    moab::Range targetVertices_;
    moab::Range targetElements_;
    std::map<moab::EntityHandle, int> sourceElementIndices_;
    std::vector<T> sourceCoordinates_;
    std::vector<T> targetCoordinates_;
    bool sourceLoaded_ = false;
    bool targetLoaded_ = false;
};

#endif // MFA_REMAP_MESHES_HPP
