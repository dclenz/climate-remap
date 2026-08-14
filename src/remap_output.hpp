#ifndef MFA_REMAP_OUTPUT_HPP
#define MFA_REMAP_OUTPUT_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <mfa/mfa.hpp>

#include "remap_meshes.hpp"

template <typename T>
class RemapOutput
{
  public:
    using Database = typename RemapMeshes<T>::Database;

    RemapOutput(
        RemapMeshes<T>& meshes,
        const moab::Range& selectedSourceElements,
        const std::vector<int>& sourceElementLevels,
        const std::vector<std::vector<int>>& sourceCellIDs,
        const std::vector<T>& sourceCoverageFractions,
        const std::vector<int>& sourceCoverageClasses,
        const std::vector<char>& sourcePointEligible,
        const std::vector<std::string>& variableNames,
        int geometryDimension,
        int maximumLevel,
        mfa::PointSet<T>* sourceInput,
        mfa::PointSet<T>* forwardApproximation,
        mfa::PointSet<T>* roundtripApproximation) :
        meshes_(meshes),
        selectedSourceElements_(selectedSourceElements),
        sourceElementLevels_(sourceElementLevels),
        sourceCellIDs_(sourceCellIDs),
        sourceCoverageFractions_(sourceCoverageFractions),
        sourceCoverageClasses_(sourceCoverageClasses),
        sourcePointEligible_(sourcePointEligible),
        variableNames_(variableNames),
        geometryDimension_(geometryDimension),
        maximumLevel_(maximumLevel),
        sourceInput_(sourceInput),
        forwardApproximation_(forwardApproximation),
        roundtripApproximation_(roundtripApproximation)
    {}

    void writeLegacyTargetTags(mfa::PointSet<T>& approximation, int locationTagLength)
    {
        for (int variable = 0; variable < variableNames_.size(); variable++)
        {
            const std::string tagName = variableNames_[variable] + "_remap";
            moab::Tag remapTag = meshes_.createDenseTag(
                Database::Target, tagName, 1, moab::MB_TYPE_DOUBLE);
            std::vector<T> remapData(approximation.npts);
            for (int point = 0; point < approximation.npts; point++)
                remapData[point] = approximation.domain(point, geometryDimension_ + variable);
            meshes_.writeTag(Database::Target, remapTag, meshes_.targetElements(),
                             remapData.data(), tagName);
        }

        const int nLevels = 15;
        if (locationTagLength < nLevels)
        {
            fmt::print(stderr, "ERROR: Legacy output requires at least {} ROMS levels; found {}.\n",
                       nLevels, locationTagLength);
            std::exit(1);
        }
        for (int level = 0; level < nLevels; level++)
        {
            for (int variable = 0; variable < variableNames_.size(); variable++)
            {
                const std::string tagName = variableNames_[variable] + "_remap_" +
                                             std::to_string(level);
                moab::Tag remapTag = meshes_.createDenseTag(
                    Database::Target, tagName, 1, moab::MB_TYPE_DOUBLE);
                std::vector<T> remapData(approximation.npts / locationTagLength);
                for (int element = 0; element < meshes_.targetElements().size(); element++)
                    remapData[element] = approximation.domain(
                        element * locationTagLength + level, geometryDimension_ + variable);
            meshes_.writeTag(Database::Target, remapTag, meshes_.targetElements(),
                             remapData.data(), tagName);
            }
        }
    }

    void writeForwardRemapH5M(const std::string& filename)
    {
        if (!forwardApproximation_)
        {
            fmt::print(stderr, "ERROR: Cannot write forward profiles before remapping.\n");
            std::exit(1);
        }

        moab::Tag locationTag = meshes_.targetTag("ROMS3DLocations");
        const int profileLength = meshes_.tagLength(
            Database::Target, locationTag, "ROMS3DLocations");
        const size_t expectedPoints = meshes_.targetElements().size() * profileLength;
        if (static_cast<size_t>(forwardApproximation_->npts) != expectedPoints)
        {
            fmt::print(stderr, "ERROR: Cannot pack {} decoded points into {} target profiles of length {}.\n",
                       forwardApproximation_->npts, meshes_.targetElements().size(), profileLength);
            std::exit(1);
        }

        for (int variable = 0; variable < variableNames_.size(); variable++)
        {
            const std::string tagName = variableNames_[variable] + "_forward_remap";
            const T* profileData = forwardApproximation_->domain.col(
                geometryDimension_ + variable).data();
            for (size_t point = 0; point < expectedPoints; point++)
            {
                if (!std::isfinite(profileData[point]))
                {
                    fmt::print(stderr, "ERROR: Non-finite value in '{}' at decoded point {}.\n",
                               tagName, point);
                    std::exit(1);
                }
            }

            moab::Tag profileTag = meshes_.createDenseTag(
                Database::Target, tagName, profileLength, moab::MB_TYPE_DOUBLE);
            meshes_.writeTag(Database::Target, profileTag, meshes_.targetElements(),
                             profileData, tagName);
        }

        meshes_.writeMesh(Database::Target, filename, meshes_.targetFileSet());
        fmt::print(stderr, "Wrote {} complete forward profiles to '{}'.\n",
                   variableNames_.size(), filename);
    }

    void writeCoverageTags()
    {
        moab::Tag fractionTag = meshes_.createDenseTag(
            Database::Source, CoverageFractionTagName, 1, moab::MB_TYPE_DOUBLE);
        moab::Tag classTag = meshes_.createDenseTag(
            Database::Source, CoverageClassTagName, 1, moab::MB_TYPE_INTEGER);
        meshes_.writeTag(Database::Source, fractionTag, selectedSourceElements_,
                         sourceCoverageFractions_.data(), CoverageFractionTagName);
        meshes_.writeTag(Database::Source, classTag, selectedSourceElements_,
                         sourceCoverageClasses_.data(), CoverageClassTagName);
    }

    void writeEligibilityTags(int profileLength, const std::vector<int>& levelEligibility,
                              const std::vector<int>& cellEligibility)
    {
        moab::Tag levelTag = meshes_.createDenseTag(
            Database::Source, LevelEligibilityTagName, profileLength, moab::MB_TYPE_INTEGER);
        moab::Tag cellTag = meshes_.createDenseTag(
            Database::Source, CellEligibilityTagName, 1, moab::MB_TYPE_INTEGER);
        meshes_.writeTag(Database::Source, levelTag, selectedSourceElements_,
                         levelEligibility.data(), LevelEligibilityTagName);
        meshes_.writeTag(Database::Source, cellTag, selectedSourceElements_,
                         cellEligibility.data(), CellEligibilityTagName);
    }

    void writeRoundtripResults(const std::string& h5mFilename,
                               const std::string& metricsFilename)
    {
        if (!roundtripApproximation_ || !sourceInput_ ||
            sourcePointEligible_.size() != static_cast<size_t>(sourceInput_->npts))
        {
            fmt::print(stderr, "ERROR: Round-trip values and eligibility must be ready before writing results.\n");
            std::exit(1);
        }

        const T nan = std::numeric_limits<T>::quiet_NaN();
        moab::Tag maxLevelCellTag = meshes_.sourceTag("maxLevelCell");
        meshes_.tagLength(Database::Source, maxLevelCellTag, "maxLevelCell");
        moab::Tag layerThicknessTag = meshes_.sourceTag("timeDaily_avg_layerThickness_3d");
        const int outputProfileLength = meshes_.tagLength(
            Database::Source, layerThicknessTag, "timeDaily_avg_layerThickness_3d");
        if (outputProfileLength < maximumLevel_)
        {
            fmt::print(stderr, "ERROR: Invalid MPAS output profile length {}.\n", outputProfileLength);
            std::exit(1);
        }

        writeCoverageTags();

        moab::Tag globalIdTag = meshes_.sourceTag("GLOBAL_ID");
        std::vector<int> sourceGlobalIds = meshes_.template readTag<int>(
            Database::Source, globalIdTag, selectedSourceElements_, 1, "GLOBAL_ID");

        std::vector<Metric> overall(variableNames_.size());
        std::vector<std::vector<Metric>> perLevel(
            variableNames_.size(), std::vector<Metric>(maximumLevel_));
        std::vector<std::vector<T>> roundtripProfiles(
            variableNames_.size(),
            std::vector<T>(selectedSourceElements_.size() * outputProfileLength, nan));
        std::vector<std::vector<T>> signedErrorProfiles(
            variableNames_.size(),
            std::vector<T>(selectedSourceElements_.size() * outputProfileLength, nan));
        std::vector<std::vector<T>> absoluteErrorProfiles(
            variableNames_.size(),
            std::vector<T>(selectedSourceElements_.size() * outputProfileLength, nan));
        std::vector<int> eligibilityProfiles(
            selectedSourceElements_.size() * outputProfileLength, 0);
        std::vector<int> cellEligibility(selectedSourceElements_.size(), 0);

        int elementIndex = 0;
        for (moab::EntityHandle element : selectedSourceElements_)
        {
            (void)element;
            for (int level = 0; level < sourceElementLevels_[elementIndex]; level++)
            {
                const int pointIndex = sourceCellIDs_[elementIndex][level];
                const int profileIndex = elementIndex * outputProfileLength + level;
                eligibilityProfiles[profileIndex] = sourcePointEligible_[pointIndex];
                if (sourcePointEligible_[pointIndex])
                    cellEligibility[elementIndex] = 1;
                for (int variable = 0; variable < variableNames_.size(); variable++)
                {
                    const T original = sourceInput_->domain(
                        pointIndex, geometryDimension_ + variable);
                    const T roundtrip = roundtripApproximation_->domain(
                        pointIndex, geometryDimension_ + variable);
                    roundtripProfiles[variable][profileIndex] = roundtrip;
                    if (sourcePointEligible_[pointIndex])
                    {
                        const T error = roundtrip - original;
                        signedErrorProfiles[variable][profileIndex] = error;
                        absoluteErrorProfiles[variable][profileIndex] = std::abs(error);
                        updateMetric(overall[variable], original, roundtrip,
                                     sourceGlobalIds[elementIndex], level);
                        updateMetric(perLevel[variable][level], original, roundtrip,
                                     sourceGlobalIds[elementIndex], level);
                    }
                }
            }
            elementIndex++;
        }

        writeEligibilityTags(outputProfileLength, eligibilityProfiles, cellEligibility);

        for (int variable = 0; variable < variableNames_.size(); variable++)
        {
            const std::vector<std::pair<std::string, std::vector<T>*>> outputs = {
                {variableNames_[variable] + "_roundtrip", &roundtripProfiles[variable]},
                {variableNames_[variable] + "_roundtrip_signed_error", &signedErrorProfiles[variable]},
                {variableNames_[variable] + "_roundtrip_absolute_error", &absoluteErrorProfiles[variable]}
            };
            for (const auto& output : outputs)
            {
                moab::Tag tag = meshes_.createDenseTag(
                    Database::Source, output.first, outputProfileLength, moab::MB_TYPE_DOUBLE);
                meshes_.writeTag(Database::Source, tag, selectedSourceElements_,
                                 output.second->data(), output.first);
            }
        }

        moab::EntityHandle outputSet;
        moab::ErrorCode result = meshes_.source().create_meshset(moab::MESHSET_SET, outputSet);
        if (result == moab::MB_SUCCESS)
            result = meshes_.source().add_entities(outputSet, selectedSourceElements_);
        if (result != moab::MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to prepare round-trip MPAS output set.\n");
            std::exit(1);
        }
        meshes_.writeMesh(Database::Source, h5mFilename, outputSet);

        std::ofstream metrics(metricsFilename);
        if (!metrics)
        {
            fmt::print(stderr, "ERROR: Failed to open metrics file '{}'.\n", metricsFilename);
            std::exit(1);
        }
        metrics << "variable,level,count,bias,rmse,max_absolute_error,max_global_id,max_level,"
                   "original_min,original_max,roundtrip_min,roundtrip_max\n";
        metrics << std::setprecision(17);
        auto writeMetric = [&](const std::string& variable, const std::string& level,
                               const Metric& metric)
        {
            if (metric.count == 0)
                return;
            const T bias = metric.sumError / metric.count;
            const T rmse = std::sqrt(metric.sumSquaredError / metric.count);
            metrics << variable << ',' << level << ',' << metric.count << ',' << bias << ','
                    << rmse << ',' << metric.maxAbsoluteError << ',' << metric.maxGlobalId << ','
                    << metric.maxLevel << ',' << metric.originalMin << ',' << metric.originalMax << ','
                    << metric.roundtripMin << ',' << metric.roundtripMax << '\n';
        };
        for (int variable = 0; variable < variableNames_.size(); variable++)
        {
            writeMetric(variableNames_[variable], "overall", overall[variable]);
            for (int level = 0; level < maximumLevel_; level++)
                writeMetric(variableNames_[variable], std::to_string(level), perLevel[variable][level]);

            const Metric& metric = overall[variable];
            if (metric.count > 0)
            {
                fmt::print(stderr,
                           "Round-trip '{}': count={} bias={} RMSE={} max_abs={} at GLOBAL_ID {} level {}\n",
                           variableNames_[variable], metric.count, metric.sumError / metric.count,
                           std::sqrt(metric.sumSquaredError / metric.count), metric.maxAbsoluteError,
                           metric.maxGlobalId, metric.maxLevel);
            }
        }
        fmt::print(stderr, "Wrote round-trip fields to '{}' and metrics to '{}'.\n",
                   h5mFilename, metricsFilename);
    }

    void writeVTK()
    {
        fmt::print(stderr, "Writing remapped data to vtk...");
        meshes_.writeMesh(Database::Target, "remap_out.vtk", meshes_.targetFileSet());

        moab::EntityHandle sourceSubset;
        meshes_.source().create_meshset(moab::MESHSET_SET, sourceSubset);
        meshes_.source().add_entities(sourceSubset, selectedSourceElements_);

        const int nLevels = 15;
        const std::vector<std::string> sourceVariableNames = {
            "timeDaily_avg_activeTracers_temperature_3d",
            "timeDaily_avg_activeTracers_salinity_3d"
        };
        for (const std::string& variableName : sourceVariableNames)
        {
            moab::Tag sourceTag = meshes_.sourceTag(variableName);
            const int tagLength = meshes_.tagLength(Database::Source, sourceTag, variableName);
            if (tagLength < nLevels)
            {
                fmt::print(stderr, "ERROR: Source tag '{}' has length {}; expected at least {}.\n",
                           variableName, tagLength, nLevels);
                std::exit(1);
            }

            std::vector<T> sourceData = meshes_.template readTag<T>(
                Database::Source, sourceTag, selectedSourceElements_, tagLength, variableName);

            std::vector<T> levelData(selectedSourceElements_.size());
            for (int level = 0; level < nLevels; level++)
            {
                for (size_t index = 0; index < selectedSourceElements_.size(); index++)
                    levelData[index] = sourceData[index * tagLength + level];

                const std::string levelTagName = variableName + "_" + std::to_string(level);
                moab::Tag levelTag = meshes_.createDenseTag(
                    Database::Source, levelTagName, 1, moab::MB_TYPE_DOUBLE);
                meshes_.writeTag(Database::Source, levelTag, selectedSourceElements_,
                                 levelData.data(), levelTagName);
            }
        }

        meshes_.writeMesh(Database::Source, "source_subset.vtk", sourceSubset);
        fmt::print(stderr, "done.\n");
    }

  private:
    static constexpr const char* CoverageFractionTagName = "roundtrip_coverage_fraction";
    static constexpr const char* CoverageClassTagName = "roundtrip_coverage_class";
    static constexpr const char* LevelEligibilityTagName = "roundtrip_error_eligible";
    static constexpr const char* CellEligibilityTagName = "roundtrip_error_cell_eligible";

    struct Metric
    {
        long count = 0;
        int maxGlobalId = -1;
        int maxLevel = -1;
        T sumError = 0;
        T sumSquaredError = 0;
        T maxAbsoluteError = 0;
        T originalMin = std::numeric_limits<T>::max();
        T originalMax = std::numeric_limits<T>::lowest();
        T roundtripMin = std::numeric_limits<T>::max();
        T roundtripMax = std::numeric_limits<T>::lowest();
    };

    static void updateMetric(Metric& metric, T original, T roundtrip, int globalId, int level)
    {
        const T error = roundtrip - original;
        const T absoluteError = std::abs(error);
        metric.count++;
        metric.sumError += error;
        metric.sumSquaredError += error * error;
        metric.originalMin = std::min(metric.originalMin, original);
        metric.originalMax = std::max(metric.originalMax, original);
        metric.roundtripMin = std::min(metric.roundtripMin, roundtrip);
        metric.roundtripMax = std::max(metric.roundtripMax, roundtrip);
        if (absoluteError >= metric.maxAbsoluteError)
        {
            metric.maxAbsoluteError = absoluteError;
            metric.maxGlobalId = globalId;
            metric.maxLevel = level;
        }
    }

    RemapMeshes<T>& meshes_;
    const moab::Range& selectedSourceElements_;
    const std::vector<int>& sourceElementLevels_;
    const std::vector<std::vector<int>>& sourceCellIDs_;
    const std::vector<T>& sourceCoverageFractions_;
    const std::vector<int>& sourceCoverageClasses_;
    const std::vector<char>& sourcePointEligible_;
    const std::vector<std::string>& variableNames_;
    int geometryDimension_;
    int maximumLevel_;
    mfa::PointSet<T>* sourceInput_;
    mfa::PointSet<T>* forwardApproximation_;
    mfa::PointSet<T>* roundtripApproximation_;
};

#endif // MFA_REMAP_OUTPUT_HPP
