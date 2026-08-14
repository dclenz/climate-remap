#ifndef MFA_REMAP_COVERAGE_HPP
#define MFA_REMAP_COVERAGE_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <diy/thirdparty/fmt/format.h>
#include <mfa/mfa.hpp>

#include "remap_meshes.hpp"

enum class CoverageClass : int
{
    Uncovered = 0,
    Partial = 1,
    Full = 2,
    Halo = 3
};

template <typename T>
struct CoverageAnalysisResult
{
    std::vector<T> coverageFractions;
    std::vector<int> coverageClasses;
    std::vector<char> pointEligible;
    int fullCount = 0;
    int partialCount = 0;
    int uncoveredCount = 0;
    int haloCount = 0;
    int eligiblePointCount = 0;
};

template <typename T>
class CoverageAnalyzer
{
  public:
    CoverageAnalysisResult<T> analyze(
        RemapMeshes<T>& remapMeshes,
        const mfa::BoundingSpace<T>& targetBoundingSpace,
        T earthRadius,
        const moab::Range& selectedSourceElements,
        const std::vector<char>& sourceElementIsCore,
        const std::vector<int>& sourceElementLevels,
        const std::vector<std::vector<int>>& sourceCellIDs,
        const VectorX<T>& sourceHeights,
        const VectorX<T>& targetHeights,
        int numSourceCells) const
    {
        const int numSourceElements = static_cast<int>(selectedSourceElements.size());
        const int targetCount = remapMeshes.targetElements().size();
        if (targetCount == 0 || numSourceElements == 0)
        {
            fmt::print(stderr, "ERROR: Cannot classify coverage with empty source or target elements.\n");
            std::exit(1);
        }

        std::vector<Polygon2> targetPolygons;
        std::vector<BoostPolygon2> boostTargetPolygons;
        targetPolygons.reserve(targetCount);
        boostTargetPolygons.reserve(targetCount);
        T gridMinX = std::numeric_limits<T>::max();
        T gridMaxX = std::numeric_limits<T>::lowest();
        T gridMinY = std::numeric_limits<T>::max();
        T gridMaxY = std::numeric_limits<T>::lowest();
        for (moab::EntityHandle element : remapMeshes.targetElements())
        {
            targetPolygons.push_back(projectedElement(&remapMeshes.target(), element, earthRadius,
                                                       targetBoundingSpace));
            const Polygon2& polygon = targetPolygons.back();
            boostTargetPolygons.push_back(toBoostPolygon(polygon.points));
            gridMinX = std::min(gridMinX, polygon.minX);
            gridMaxX = std::max(gridMaxX, polygon.maxX);
            gridMinY = std::min(gridMinY, polygon.minY);
            gridMaxY = std::max(gridMaxY, polygon.maxY);
        }

        const int gridSize = std::max(1, static_cast<int>(std::sqrt(targetCount)));
        const T cellWidth = (gridMaxX - gridMinX) / gridSize;
        const T cellHeight = (gridMaxY - gridMinY) / gridSize;
        std::vector<std::vector<int>> bins(gridSize * gridSize);
        auto binIndex = [&](T value, T minimum, T width)
        {
            if (width <= 0)
                return 0;
            return std::max(0, std::min(gridSize - 1,
                static_cast<int>((value - minimum) / width)));
        };
        for (int targetIdx = 0; targetIdx < targetCount; targetIdx++)
        {
            const Polygon2& polygon = targetPolygons[targetIdx];
            const int minI = binIndex(polygon.minX, gridMinX, cellWidth);
            const int maxI = binIndex(polygon.maxX, gridMinX, cellWidth);
            const int minJ = binIndex(polygon.minY, gridMinY, cellHeight);
            const int maxJ = binIndex(polygon.maxY, gridMinY, cellHeight);
            for (int j = minJ; j <= maxJ; j++)
                for (int i = minI; i <= maxI; i++)
                    bins[j * gridSize + i].push_back(targetIdx);
        }

        const int profileLength = targetHeights.size() / targetCount;
        std::vector<T> targetProfileMins(targetCount, std::numeric_limits<T>::max());
        std::vector<T> targetProfileMaxs(targetCount, std::numeric_limits<T>::lowest());
        for (int targetIdx = 0; targetIdx < targetCount; targetIdx++)
        {
            const int firstPoint = targetIdx * profileLength;
            for (int level = 0; level < profileLength; level++)
            {
                targetProfileMins[targetIdx] = std::min(targetProfileMins[targetIdx],
                                                         targetHeights(firstPoint + level));
                targetProfileMaxs[targetIdx] = std::max(targetProfileMaxs[targetIdx],
                                                         targetHeights(firstPoint + level));
            }
        }

        CoverageAnalysisResult<T> result;
        result.coverageFractions.assign(numSourceElements, 0);
        result.coverageClasses.assign(numSourceElements, 0);
        result.pointEligible.assign(numSourceCells, 0);
        std::vector<int> candidateStamp(targetCount, -1);
        const T fullCoverageTolerance = static_cast<T>(1e-6);

        int sourceIdx = 0;
        for (moab::EntityHandle sourceElement : selectedSourceElements)
        {
            const Polygon2 sourcePolygon = projectedElement(&remapMeshes.source(), sourceElement, 1,
                                                             targetBoundingSpace);
            const T sourceArea = std::abs(signedPolygonArea(sourcePolygon.points));
            if (sourceArea <= 0)
            {
                fmt::print(stderr, "ERROR: Degenerate projected MPAS polygon {}.\n", sourceElement);
                std::exit(1);
            }

            const int minI = binIndex(sourcePolygon.minX, gridMinX, cellWidth);
            const int maxI = binIndex(sourcePolygon.maxX, gridMinX, cellWidth);
            const int minJ = binIndex(sourcePolygon.minY, gridMinY, cellHeight);
            const int maxJ = binIndex(sourcePolygon.maxY, gridMinY, cellHeight);
            std::vector<int> candidates;
            for (int j = minJ; j <= maxJ; j++)
            {
                for (int i = minI; i <= maxI; i++)
                {
                    for (int targetIdx : bins[j * gridSize + i])
                    {
                        if (candidateStamp[targetIdx] != sourceIdx)
                        {
                            candidateStamp[targetIdx] = sourceIdx;
                            candidates.push_back(targetIdx);
                        }
                    }
                }
            }

            T supportedMinHeight = std::numeric_limits<T>::lowest();
            T supportedMaxHeight = std::numeric_limits<T>::max();
            bool hasPositiveOverlap = false;
            BoostMultiPolygon2 coveredRegion;
            const BoostPolygon2 boostSourcePolygon = toBoostPolygon(sourcePolygon.points);
            for (int targetIdx : candidates)
            {
                const Polygon2& targetPolygon = targetPolygons[targetIdx];
                if (targetPolygon.maxX < sourcePolygon.minX || targetPolygon.minX > sourcePolygon.maxX ||
                    targetPolygon.maxY < sourcePolygon.minY || targetPolygon.minY > sourcePolygon.maxY)
                    continue;

                std::vector<BoostPolygon2> intersections;
                boost::geometry::intersection(boostSourcePolygon,
                                              boostTargetPolygons[targetIdx],
                                              intersections);
                T intersectionArea = 0;
                for (const BoostPolygon2& intersection : intersections)
                    intersectionArea += std::abs(boost::geometry::area(intersection));
                if (intersectionArea <= sourceArea * static_cast<T>(1e-12))
                    continue;

                if (coveredRegion.empty())
                {
                    for (const BoostPolygon2& intersection : intersections)
                        coveredRegion.push_back(intersection);
                }
                else
                {
                    BoostMultiPolygon2 additions;
                    for (const BoostPolygon2& intersection : intersections)
                    {
                        BoostMultiPolygon2 difference;
                        boost::geometry::difference(intersection, coveredRegion, difference);
                        additions.insert(additions.end(), difference.begin(), difference.end());
                    }
                    coveredRegion.insert(coveredRegion.end(), additions.begin(), additions.end());
                }
                hasPositiveOverlap = true;
                supportedMinHeight = std::max(supportedMinHeight, targetProfileMins[targetIdx]);
                supportedMaxHeight = std::min(supportedMaxHeight, targetProfileMaxs[targetIdx]);
            }

            T coveredArea = 0;
            for (const BoostPolygon2& polygon : coveredRegion)
                coveredArea += std::abs(boost::geometry::area(polygon));
            const T coverageFraction = std::max(static_cast<T>(0),
                std::min(static_cast<T>(1), coveredArea / sourceArea));
            result.coverageFractions[sourceIdx] = coverageFraction;
            if (!sourceElementIsCore[sourceIdx])
            {
                result.coverageClasses[sourceIdx] = static_cast<int>(CoverageClass::Halo);
                result.haloCount++;
            }
            else if (!hasPositiveOverlap || coverageFraction <= fullCoverageTolerance)
            {
                result.coverageClasses[sourceIdx] = static_cast<int>(CoverageClass::Uncovered);
                result.uncoveredCount++;
            }
            else if (coverageFraction < static_cast<T>(1) - fullCoverageTolerance)
            {
                result.coverageClasses[sourceIdx] = static_cast<int>(CoverageClass::Partial);
                result.partialCount++;
            }
            else
            {
                result.coverageClasses[sourceIdx] = static_cast<int>(CoverageClass::Full);
                result.fullCount++;
                if (supportedMinHeight <= supportedMaxHeight)
                {
                    for (int level = 0; level < sourceElementLevels[sourceIdx]; level++)
                    {
                        const int pointIdx = sourceCellIDs[sourceIdx][level];
                        const T height = sourceHeights(pointIdx);
                        if (height >= supportedMinHeight && height <= supportedMaxHeight)
                        {
                            result.pointEligible[pointIdx] = 1;
                            result.eligiblePointCount++;
                        }
                    }
                }
            }
            sourceIdx++;
        }

        return result;
    }

  private:
    struct Point2
    {
        T x;
        T y;
    };

    struct Polygon2
    {
        std::vector<Point2> points;
        T minX;
        T maxX;
        T minY;
        T maxY;
    };

    using BoostPoint2 = boost::geometry::model::d2::point_xy<T>;
    using BoostPolygon2 = boost::geometry::model::polygon<BoostPoint2>;
    using BoostMultiPolygon2 = boost::geometry::model::multi_polygon<BoostPolygon2>;

    static BoostPolygon2 toBoostPolygon(const std::vector<Point2>& points)
    {
        BoostPolygon2 polygon;
        for (const Point2& point : points)
            boost::geometry::append(polygon.outer(), BoostPoint2(point.x, point.y));
        boost::geometry::correct(polygon);
        return polygon;
    }

    static T signedPolygonArea(const std::vector<Point2>& polygon)
    {
        T area = 0;
        for (size_t i = 0; i < polygon.size(); i++)
        {
            const Point2& a = polygon[i];
            const Point2& b = polygon[(i + 1) % polygon.size()];
            area += a.x * b.y - b.x * a.y;
        }
        return area * static_cast<T>(0.5);
    }

    static Polygon2 projectedElement(
        moab::Interface* mb,
        moab::EntityHandle element,
        T coordinateScale,
        const mfa::BoundingSpace<T>& targetBoundingSpace)
    {
        const moab::EntityHandle* connectivity = nullptr;
        int count = 0;
        moab::ErrorCode rval = mb->get_connectivity(element, connectivity, count);
        if (rval != moab::MB_SUCCESS || !connectivity || count < 3)
        {
            fmt::print(stderr, "ERROR: Invalid polygon connectivity for entity {}.\n", element);
            std::exit(1);
        }

        std::vector<T> coordinates(count * RemapMeshes<T>::GeometryDimension);
        rval = mb->get_coords(connectivity, count, coordinates.data());
        if (rval != moab::MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to read polygon coordinates for entity {}.\n", element);
            std::exit(1);
        }

        Polygon2 polygon;
        polygon.minX = polygon.minY = std::numeric_limits<T>::max();
        polygon.maxX = polygon.maxY = std::numeric_limits<T>::lowest();
        polygon.points.reserve(count);
        for (int i = 0; i < count; i++)
        {
            VectorX<T> xyz(RemapMeshes<T>::GeometryDimension);
            for (int dim = 0; dim < RemapMeshes<T>::GeometryDimension; dim++)
                xyz(dim) = coordinates[i * RemapMeshes<T>::GeometryDimension + dim] * coordinateScale;
            VectorX<T> projected;
            targetBoundingSpace.toRefSpace(xyz, projected);
            Point2 point{projected(0), projected(1)};
            polygon.points.push_back(point);
            polygon.minX = std::min(polygon.minX, point.x);
            polygon.maxX = std::max(polygon.maxX, point.x);
            polygon.minY = std::min(polygon.minY, point.y);
            polygon.maxY = std::max(polygon.maxY, point.y);
        }
        return polygon;
    }
};

#endif // MFA_REMAP_COVERAGE_HPP
