#ifndef MFA_REMAP_DOMAIN_HPP
#define MFA_REMAP_DOMAIN_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <diy/master.hpp>
#include <mfa/mfa.hpp>

#include "remap_coverage.hpp"
#include "remap_meshes.hpp"

template <typename T>
class RemapDomain
{
  public:
    struct Configuration
    {
        int verbose = 0;
        std::vector<std::string> variableNames;
        bool deterministicTargetPlane = false;
        bool combineScienceVariables = false;
    };

    static constexpr int GeometryDimension = 3;
    static constexpr int ParameterDimension = 3;
    static constexpr T EarthRadius = 6371220.0;

    RemapDomain(RemapMeshes<T>& meshes, Configuration configuration) :
        meshes_(meshes), configuration_(std::move(configuration))
    {}

    VectorXi pointSetModelDimensions() const
    {
        VectorXi dimensions = configuration_.combineScienceVariables ?
            VectorXi(2) : VectorXi::Ones(configuration_.variableNames.size() + 1);
        dimensions(0) = GeometryDimension;
        if (configuration_.combineScienceVariables)
            dimensions(1) = configuration_.variableNames.size();
        return dimensions;
    }

    template <typename V>
    void readTargetData(const diy::Master::ProxyWithLink&, const std::string& filename)
    {
        meshes_.loadTarget(filename);

        MatrixX<T> locations(meshes_.targetElements().size(), GeometryDimension);
        std::vector<V> coord(GeometryDimension);
        int i = 0;
        for (auto it = meshes_.targetElements().begin(), end = meshes_.targetElements().end();
             it != end; ++it, ++i)
        {
            meshes_.target().get_coords(&(*it), 1, coord.data());
            for (int j = 0; j < GeometryDimension; j++)
                locations(i, j) = static_cast<T>(coord[j]) * EarthRadius;
        }
        planeMap_ = computeTargetPlaneMap(locations);
        targetBoundingSpace_ = mfa::BoundingSpace<T>(planeMap_, locations);
        targetSurfaceReady_ = true;
    }

    template <typename V>
    void readSourceData(const diy::Master::ProxyWithLink& cp, const std::string& filename)
    {
        requireState(targetSurfaceReady_, "Target surface must be read before source data.");
        meshes_.loadSource(filename);
        if (configuration_.verbose >= 1)
            targetBoundingSpace_.print("Target Bounding Space");

        numSourceCells_ = 0;
        numSourceElements_ = 0;
        maxLevel_ = 0;
        sourceCellIDs_.clear();
        sourceElementLevels_.clear();
        selectedSourceElements_.clear();
        selectSourceElements(targetBoundingSpace_);

        MatrixX<T> locations(numSourceElements_, GeometryDimension);
        std::vector<T> coord(GeometryDimension);
        int i = 0;
        for (auto it = selectedSourceElements_.begin(), end = selectedSourceElements_.end();
             it != end; ++it, ++i)
        {
            meshes_.source().get_coords(&(*it), 1, coord.data());
            for (int j = 0; j < GeometryDimension; j++)
                locations(i, j) = coord[j];
        }
        sourceBoundingSpace_ = mfa::BoundingSpace<T>(planeMap_, locations);
        totalBoundingSpace_ = sourceBoundingSpace_.merge(targetBoundingSpace_);
        targetBoundingSpace_.print("Target Bounding Space");
        sourceBoundingSpace_.print("Source Bounding Space");
        totalBoundingSpace_.print("Total Bounding Space");

        computeSourceLevelInfo();
        computeSourceCellInfo();
        sourceInput_ = std::make_unique<mfa::PointSet<T>>(
            ParameterDimension, pointSetModelDimensions(), numSourceCells_);

        using Database = typename RemapMeshes<T>::Database;
        moab::Tag sshTag = meshes_.sourceTag("timeDaily_avg_ssh");
        moab::Tag thicknessTag = meshes_.sourceTag("timeDaily_avg_layerThickness_3d");
        const int tagLength = meshes_.tagLength(
            Database::Source, thicknessTag, "timeDaily_avg_layerThickness_3d");
        std::vector<T> ssh = meshes_.template readTag<T>(
            Database::Source, sshTag, selectedSourceElements_, 1, "timeDaily_avg_ssh");
        std::vector<T> thickness = meshes_.template readTag<T>(
            Database::Source, thicknessTag, selectedSourceElements_, tagLength,
            "timeDaily_avg_layerThickness_3d");

        int elementID = 0;
        minSourceHeight_ = std::numeric_limits<T>::max();
        maxSourceHeight_ = std::numeric_limits<T>::lowest();
        sourceHeights_.resize(numSourceCells_);
        std::vector<T> heightSums(maxLevel_, 0.0);
        std::vector<int> heightCounts(maxLevel_, 0);
        std::vector<V> centroid(GeometryDimension);
        for (auto it = selectedSourceElements_.begin(), end = selectedSourceElements_.end();
             it != end; ++it, ++elementID)
        {
            meshes_.source().get_coords(&(*it), 1, centroid.data());
            VectorX<T> normal = VectorX<T>::Zero(GeometryDimension);
            T norm2 = 0.0;
            for (int k = 0; k < GeometryDimension; k++)
            {
                normal(k) = static_cast<T>(centroid[k]);
                norm2 += normal(k) * normal(k);
            }
            if (norm2 <= 0.0)
            {
                fmt::print(stderr, "ERROR: Degenerate centroid while building source geometry.\n");
                std::exit(1);
            }
            normal /= std::sqrt(norm2);

            T cumulativeThickness = 0.0;
            for (int level = 0; level < sourceElementLevels_[elementID]; level++)
            {
                if (level >= tagLength)
                {
                    fmt::print(stderr, "ERROR: sourceElementLevels exceeds layerThickness tag length.\n");
                    std::exit(1);
                }
                int cellID = sourceCellIDs_[elementID][level];
                T layerThickness = thickness[elementID * tagLength + level];
                const T top = level == 0 ? ssh[elementID] : -cumulativeThickness;
                const T bottom = -(cumulativeThickness + layerThickness);
                T axialCentroid = 0.5 * (top + bottom);
                for (int k = 0; k < GeometryDimension; k++)
                    sourceInput_->domain(cellID, k) =
                        static_cast<T>(centroid[k]) + axialCentroid * normal(k);
                sourceHeights_(cellID) = axialCentroid;
                heightSums[level] += axialCentroid;
                heightCounts[level]++;
                cumulativeThickness += layerThickness;
            }
            if (ssh[elementID] > maxSourceHeight_)
                maxSourceHeight_ = ssh[elementID];
            if (-1 * cumulativeThickness < minSourceHeight_)
                minSourceHeight_ = -1 * cumulativeThickness;
        }

        sourceLevelMeanHeights_.clear();
        sourceLevelMeanHeights_.reserve(maxLevel_);
        for (int level = 0; level < maxLevel_; level++)
        {
            if (heightCounts[level] > 0)
                sourceLevelMeanHeights_.push_back(heightSums[level] / heightCounts[level]);
        }
        for (int i = 0; i < configuration_.variableNames.size(); i++)
            addSourceVariable<V>(cp, filename, configuration_.variableNames[i], i);
        sourcePointSetReady_ = true;
    }

    template <typename V>
    void readTargetDataFull()
    {
        requireState(targetSurfaceReady_, "Target surface must be read before target profiles.");
        using Database = typename RemapMeshes<T>::Database;
        int elementCount = meshes_.targetElements().size();
        moab::Tag sshTag = meshes_.targetTag("SSH");
        moab::Tag bathymetryTag = meshes_.targetTag("bathymetry");
        moab::Tag locationTag = meshes_.targetTag("ROMS3DLocations");
        const int tagLength = meshes_.tagLength(
            Database::Target, locationTag, "ROMS3DLocations");
        std::vector<T> ssh = meshes_.template readTag<T>(
            Database::Target, sshTag, meshes_.targetElements(), 1, "SSH");
        std::vector<T> bathymetry = meshes_.template readTag<T>(
            Database::Target, bathymetryTag, meshes_.targetElements(), 1, "bathymetry");
        std::vector<T> locations = meshes_.template readTag<T>(
            Database::Target, locationTag, meshes_.targetElements(), tagLength,
            "ROMS3DLocations");

        minTargetHeight_ = std::numeric_limits<T>::max();
        maxTargetHeight_ = std::numeric_limits<T>::lowest();
        targetHeights_.resize(elementCount * tagLength);
        targetInput_ = std::make_unique<mfa::PointSet<T>>(
            ParameterDimension, pointSetModelDimensions(), elementCount * tagLength);
        std::vector<V> coord(GeometryDimension);
        VectorX<T> centroid(GeometryDimension);
        int i = 0;
        for (auto it = meshes_.targetElements().begin(), end = meshes_.targetElements().end();
             it != end; ++it, ++i)
        {
            meshes_.target().get_coords(&(*it), 1, coord.data());
            for (int k = 0; k < GeometryDimension; k++)
                centroid(k) = coord[k] * EarthRadius;
            const VectorX<T> normal = centroid.normalized();
            for (int level = 0; level < tagLength; level++)
            {
                int idx = i * tagLength + level;
                T height = -1 * locations[idx];
                targetInput_->domain.block(idx, 0, 1, GeometryDimension) =
                    (centroid + height * normal).transpose();
                targetHeights_(idx) = height;
            }
            if (ssh[i] > maxTargetHeight_)
                maxTargetHeight_ = ssh[i];
            if (-1 * bathymetry[i] < minTargetHeight_)
                minTargetHeight_ = -1 * bathymetry[i];
        }
        targetPointSetReady_ = true;
    }

    void setParameterizations()
    {
        requireState(sourcePointSetReady_ && targetPointSetReady_,
                     "Source and target point sets must be ready before parameterization.");
        minHeight_ = std::min(minSourceHeight_, minTargetHeight_);
        maxHeight_ = std::max(maxSourceHeight_, maxTargetHeight_);
        requireState(maxHeight_ > minHeight_, "Source/target height range must be nonzero.");
        MatrixX<T> sourceParams(sourceInput_->npts, ParameterDimension);
        MatrixX<T> targetParams(targetInput_->npts, ParameterDimension);
        mfa::AffineMap<T> paramMap(totalBoundingSpace_);
        MatrixX<T> params2d;
        MatrixX<T> xyz = sourceInput_->domain.leftCols(GeometryDimension);
        paramMap.transform(xyz, params2d, true);
        sourceParams.leftCols(2) = params2d;
        sourceParams.col(2) = (sourceHeights_.array() - minHeight_) / (maxHeight_ - minHeight_);
        xyz = targetInput_->domain.leftCols(GeometryDimension);
        paramMap.transform(xyz, params2d, true);
        targetParams.leftCols(2) = params2d;
        targetParams.col(2) = (targetHeights_.array() - minHeight_) / (maxHeight_ - minHeight_);
        if (configuration_.verbose >= 2)
            fmt::print(stderr, "Setting source parameters\n");
        sourceInput_->set_custom_params_unstructured(sourceParams, 1e-6);
        if (configuration_.verbose >= 2)
            fmt::print(stderr, "Setting target parameters\n");
        targetInput_->set_custom_params_unstructured(targetParams, 1e-6);
    }

    void classifySourceCoverage()
    {
        requireState(sourcePointSetReady_ && targetPointSetReady_,
                     "Source and target point sets must be ready before coverage classification.");
        CoverageAnalyzer<T> analyzer;
        CoverageAnalysisResult<T> result = analyzer.analyze(
            meshes_, targetBoundingSpace_, EarthRadius, selectedSourceElements_,
            sourceElementIsCore_, sourceElementLevels_, sourceCellIDs_, sourceHeights_,
            targetHeights_, numSourceCells_);
        sourceCoverageFractions_ = std::move(result.coverageFractions);
        sourceCoverageClasses_ = std::move(result.coverageClasses);
        sourcePointEligible_ = std::move(result.pointEligible);
        fmt::print(stderr,
                   "MPAS wet-ROMS coverage: full={} partial={} uncovered={} halo={}; eligible active levels={}\n",
                   result.fullCount, result.partialCount, result.uncoveredCount, result.haloCount,
                   result.eligiblePointCount);
    }

    mfa::PointSet<T>& sourceInput() { return *sourceInput_; }
    mfa::PointSet<T>& targetInput() { return *targetInput_; }
    std::unique_ptr<mfa::PointSet<T>> releaseSource() { return std::move(sourceInput_); }
    std::unique_ptr<mfa::PointSet<T>> releaseTarget() { return std::move(targetInput_); }
    const VectorX<T>& targetHeights() const { return targetHeights_; }
    const std::vector<T>& sourceLevelMeanHeights() const { return sourceLevelMeanHeights_; }
    T minHeight() const { return minHeight_; }
    T maxHeight() const { return maxHeight_; }
    const moab::Range& selectedSourceElements() const { return selectedSourceElements_; }
    const std::vector<int>& sourceElementLevels() const { return sourceElementLevels_; }
    const std::vector<std::vector<int>>& sourceCellIDs() const { return sourceCellIDs_; }
    const std::vector<T>& sourceCoverageFractions() const { return sourceCoverageFractions_; }
    const std::vector<int>& sourceCoverageClasses() const { return sourceCoverageClasses_; }
    const std::vector<char>& sourcePointEligible() const { return sourcePointEligible_; }
    const std::vector<std::string>& variableNames() const { return configuration_.variableNames; }
    int maximumLevel() const { return maxLevel_; }

  private:
    static void requireState(bool condition, const char* message)
    {
        if (!condition)
        {
            fmt::print(stderr, "ERROR: {}\n", message);
            std::exit(1);
        }
    }

    mfa::AffMap<T> computeTargetPlaneMap(const MatrixX<T>& locations) const
    {
        VectorX<T> center = locations.colwise().mean();
        VectorX<T> normal = configuration_.deterministicTargetPlane ? center.normalized() :
                             mfa::estimateSurfaceNormal<T>(locations);
        auto [a, b] = mfa::getPlaneVectors<T>(normal);
        if (configuration_.verbose >= 2)
        {
            fmt::print(stderr, "Target Orientation:\n");
            fmt::print(stderr, "  a: {}\n", mfa::print_vec(a));
            fmt::print(stderr, "  b: {}\n", mfa::print_vec(b));
            fmt::print(stderr, "  n: {}\n", mfa::print_vec(normal));
        }
        return mfa::AffMap<T>(2, 3, center, {a, b});
    }

    bool elementIntersectsSourceReadBbox(
        moab::EntityHandle element, const mfa::BoundingSpace<T>& space) const
    {
        const moab::EntityHandle* connectivity = nullptr;
        int count = 0;
        moab::ErrorCode result = meshes_.source().get_connectivity(
            element, connectivity, count);
        if (result != moab::MB_SUCCESS || !connectivity || count == 0)
        {
            fmt::print(stderr, "ERROR: Failed to get element connectivity in elementIntersectsSourceReadBbox().\n");
            std::exit(1);
        }
        std::vector<T> coordinates(GeometryDimension * count);
        result = meshes_.source().get_coords(connectivity, count, coordinates.data());
        if (result != moab::MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to get element coordinates in elementIntersectsSourceReadBbox().\n");
            std::exit(1);
        }
        VectorX<T> mins = VectorX<T>::Constant(space.spaceDim, std::numeric_limits<T>::max());
        VectorX<T> maxs = VectorX<T>::Constant(space.spaceDim, std::numeric_limits<T>::lowest());
        VectorX<T> xyz(GeometryDimension);
        VectorX<T> rotated;
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < GeometryDimension; j++)
                xyz(j) = coordinates[GeometryDimension * i + j];
            space.toRefSpace(xyz, rotated);
            mins = mins.cwiseMin(rotated);
            maxs = maxs.cwiseMax(rotated);
        }
        for (int j = 0; j < space.spaceDim; j++)
        {
            if (maxs(j) < space.refMins(j) || mins(j) > space.refMaxs(j))
                return false;
        }
        VectorX<T> planePoint;
        space.toPhysicalSpace(rotated, planePoint);
        T distanceToPlane = (xyz - planePoint).norm();
        return distanceToPlane <= EarthRadius / 2;
    }

    std::vector<int> getElementNeighborsBySharedEdge(moab::EntityHandle element) const
    {
        const moab::EntityHandle* connectivity = nullptr;
        int count = 0;
        moab::ErrorCode result = meshes_.source().get_connectivity(
            element, connectivity, count);
        if (result != moab::MB_SUCCESS || !connectivity || count == 0)
        {
            fmt::print(stderr, "ERROR: Failed to get element connectivity in getElementNeighborsBySharedEdge().\n");
            std::exit(1);
        }
        std::unordered_map<int, int> sharedVertexCount;
        for (int i = 0; i < count; i++)
        {
            moab::EntityHandle vertex = connectivity[i];
            moab::Range adjacent;
            result = meshes_.source().get_adjacencies(&vertex, 1, 2, false, adjacent);
            if (result != moab::MB_SUCCESS)
            {
                fmt::print(stderr, "ERROR: Failed to get vertex adjacencies in getElementNeighborsBySharedEdge().\n");
                std::exit(1);
            }
            for (moab::EntityHandle candidate : adjacent)
            {
                if (candidate == element)
                    continue;
                auto found = meshes_.sourceElementIndices().find(candidate);
                if (found != meshes_.sourceElementIndices().end())
                    sharedVertexCount[found->second]++;
            }
        }
        std::vector<int> neighbors;
        neighbors.reserve(sharedVertexCount.size());
        for (const auto& entry : sharedVertexCount)
        {
            if (entry.second >= 2)
                neighbors.push_back(entry.first);
        }
        return neighbors;
    }

    void selectSourceElements(const mfa::BoundingSpace<T>& space)
    {
        int totalElements = meshes_.sourceElements().size();
        std::vector<moab::EntityHandle> handles(totalElements);
        int index = 0;
        for (moab::EntityHandle element : meshes_.sourceElements())
            handles[index++] = element;
        std::vector<char> selected(totalElements, 0);
        std::vector<char> coreSelected(totalElements, 0);
        std::vector<int> coreIndices;
        coreIndices.reserve(totalElements);
        for (int i = 0; i < totalElements; i++)
        {
            if (elementIntersectsSourceReadBbox(handles[i], space))
            {
                selected[i] = 1;
                coreSelected[i] = 1;
                coreIndices.push_back(i);
            }
        }
        if (coreIndices.empty())
        {
            fmt::print(stderr, "ERROR: source read bbox does not intersect any source elements.\n");
            std::exit(1);
        }
        std::vector<int> boundaryCoreIndices;
        boundaryCoreIndices.reserve(coreIndices.size());
        for (int coreIndex : coreIndices)
        {
            auto neighbors = getElementNeighborsBySharedEdge(handles[coreIndex]);
            bool boundary = false;
            for (int neighbor : neighbors)
            {
                if (!selected[neighbor])
                {
                    boundary = true;
                    break;
                }
            }
            if (boundary)
                boundaryCoreIndices.push_back(coreIndex);
        }
        std::vector<char> haloSelected(totalElements, 0);
        for (int coreIndex : boundaryCoreIndices)
        {
            auto neighbors = getElementNeighborsBySharedEdge(handles[coreIndex]);
            for (int neighbor : neighbors)
            {
                haloSelected[neighbor] = 1;
                selected[neighbor] = 1;
            }
        }
        selectedSourceElements_.clear();
        sourceElementIsCore_.clear();
        int haloCount = 0;
        for (int i = 0; i < totalElements; i++)
        {
            if (selected[i])
            {
                selectedSourceElements_.insert(handles[i]);
                sourceElementIsCore_.push_back(coreSelected[i]);
                if (!coreSelected[i] && haloSelected[i])
                    haloCount++;
            }
        }
        numSourceElements_ = selectedSourceElements_.size();
        if (numSourceElements_ == 0)
        {
            fmt::print(stderr, "ERROR: Empty selected source element set after bbox/halo selection.\n");
            std::exit(1);
        }
        if (configuration_.verbose >= 1)
            fmt::print(stderr, "Selected source elements: {} (core={} halo={})\n",
                       numSourceElements_, coreIndices.size(), haloCount);
    }

    void computeSourceLevelInfo()
    {
        if (selectedSourceElements_.empty())
        {
            fmt::print(stderr, "ERROR: computeSourceLevelInfo called with empty selectedSourceElements.\n");
            std::exit(1);
        }
        using Database = typename RemapMeshes<T>::Database;
        moab::Tag tag = meshes_.sourceTag("maxLevelCell");
        numSourceElements_ = selectedSourceElements_.size();
        sourceElementLevels_ = meshes_.template readTag<int>(
            Database::Source, tag, selectedSourceElements_, 1, "maxLevelCell");
        maxLevel_ = 0;
        for (int level : sourceElementLevels_)
        {
            if (level > maxLevel_)
                maxLevel_ = level;
        }
    }

    void computeSourceCellInfo()
    {
        sourceCellIDs_.resize(numSourceElements_);
        for (int i = 0; i < numSourceElements_; i++)
        {
            sourceCellIDs_[i].resize(sourceElementLevels_[i]);
            for (int level = 0; level < sourceElementLevels_[i]; level++)
                sourceCellIDs_[i][level] = numSourceCells_ + level;
            numSourceCells_ += sourceElementLevels_[i];
        }
    }

    template <typename V>
    void addSourceVariable(const diy::Master::ProxyWithLink&, const std::string&,
                           const std::string& name, int variableIndex)
    {
        using Database = typename RemapMeshes<T>::Database;
        moab::Tag tag = meshes_.sourceTag(name);
        const int tagLength = meshes_.tagLength(Database::Source, tag, name);
        std::vector<T> data = meshes_.template readTag<T>(
            Database::Source, tag, selectedSourceElements_, tagLength, name);
        for (int element = 0; element < numSourceElements_; element++)
        {
            for (int level = 0; level < sourceElementLevels_[element]; level++)
            {
                int cell = sourceCellIDs_[element][level];
                if (tagLength == 1)
                    sourceInput_->domain(cell, GeometryDimension + variableIndex) = data[element];
                else
                {
                    if (level >= tagLength)
                    {
                        fmt::print(stderr, "ERROR: Requested level {} from tag '{}' length {}.\n",
                                   level, name, tagLength);
                        std::exit(1);
                    }
                    int fullCell = element * tagLength + level;
                    sourceInput_->domain(cell, GeometryDimension + variableIndex) = data[fullCell];
                }
            }
        }
    }

    RemapMeshes<T>& meshes_;
    Configuration configuration_;
    std::unique_ptr<mfa::PointSet<T>> sourceInput_;
    std::unique_ptr<mfa::PointSet<T>> targetInput_;
    mfa::AffMap<T> planeMap_;
    mfa::BoundingSpace<T> targetBoundingSpace_;
    mfa::BoundingSpace<T> sourceBoundingSpace_;
    mfa::BoundingSpace<T> totalBoundingSpace_;
    VectorX<T> sourceHeights_;
    VectorX<T> targetHeights_;
    std::vector<T> sourceLevelMeanHeights_;
    T minHeight_;
    T maxHeight_;
    T minSourceHeight_;
    T maxSourceHeight_;
    T minTargetHeight_;
    T maxTargetHeight_;
    int maxLevel_ = 0;
    int numSourceElements_ = 0;
    int numSourceCells_ = 0;
    moab::Range selectedSourceElements_;
    std::vector<int> sourceElementLevels_;
    std::vector<std::vector<int>> sourceCellIDs_;
    std::vector<char> sourceElementIsCore_;
    std::vector<T> sourceCoverageFractions_;
    std::vector<int> sourceCoverageClasses_;
    std::vector<char> sourcePointEligible_;
    bool targetSurfaceReady_ = false;
    bool sourcePointSetReady_ = false;
    bool targetPointSetReady_ = false;
};

#endif // MFA_REMAP_DOMAIN_HPP
