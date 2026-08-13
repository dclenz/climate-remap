//--------------------------------------------------------------
// A custom block for climate remapping
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_CBLOCK
#define _MFA_CBLOCK

#include    <algorithm>
#include    <random>
#include    <stdio.h>
#include    <cmath>
#include    <limits>
#include    <fstream>
#include    <iomanip>
#include    <unordered_map>
#include    <boost/geometry.hpp>
#include    <boost/geometry/geometries/multi_polygon.hpp>
#include    <boost/geometry/geometries/point_xy.hpp>
#include    <boost/geometry/geometries/polygon.hpp>
#include    <mfa/mfa.hpp>
#include    <mfa/block_base.hpp>
#include    <diy/master.hpp>
#include    <diy/reduce-operations.hpp>
#include    <diy/decomposition.hpp>
#include    <diy/assigner.hpp>
#include    <diy/io/block.hpp>
#include    <diy/io/bov.hpp>
#include    <diy/pick.hpp>
#include    <highfive/H5DataSet.hpp>
#include    <highfive/H5DataSpace.hpp>
#include    <highfive/H5File.hpp>

#include    "moab/Core.hpp"
#include    "moab/ParallelComm.hpp"

using namespace std;
using namespace moab;

template <typename V>
struct MBReader
{
    const int geomDim;
    int cellDim;

    // Separate MOAB instances for source and target to avoid tag-definition
    // conflicts (e.g. refBottomDepth with different array lengths).
    Interface* sourceMb;
    ParallelComm* sourcePc;
    Interface* targetMb;
    ParallelComm* targetPc;
    EntityHandle sourceFileSet{0};
    EntityHandle targetFileSet{0};

    // Source data
    string sourceFilename;
    Range sourceVertices;
    Range sourceElements;
    map<EntityHandle, int> sourceEltIdxMap;
    vector<EntityHandle> sourceBdryFaces;   // boundary faces on the source mesh
    vector<V> sourceCoords;

    // Target data
    string targetFilename;
    Range  targetVertices;
    Range  targetElements;
    map<EntityHandle, int> targetEltIdxMap;
    vector<V> targetCoords;

    ErrorCode rval;

    MBReader(MPI_Comm local, int dim) :
        geomDim(3),
        cellDim(dim)
    {
        sourceMb = new Core;
        sourcePc = new ParallelComm(sourceMb, local);
        targetMb = new Core;
        targetPc = new ParallelComm(targetMb, local);
    }

    ~MBReader()
    {
        delete sourcePc;
        delete sourceMb;
        delete targetPc;
        delete targetMb;
    }

    void loadSourceMesh(string filename_)
    {
        fmt::print(stderr, "MBReader: Reading remap source file {}\n", filename_);
        sourceFilename = filename_;

        sourceMb->create_meshset(MESHSET_SET, sourceFileSet);
        rval = sourceMb->load_file(sourceFilename.c_str(), &sourceFileSet); MB_CHK_ERR_RET(rval);

        // Get vertices
        rval = sourceMb->get_entities_by_type(sourceFileSet, MBVERTEX, sourceVertices); MB_CHK_ERR_RET(rval);

        // Get vertex coordinates
        sourceCoords.resize(sourceVertices.size()*3);
        rval = sourceMb->get_coords(sourceVertices, sourceCoords.data()); MB_CHK_ERR_RET(rval);

        // Get 2D mesh element rage
        rval = sourceMb->get_entities_by_dimension(sourceFileSet, 2, sourceElements); MB_CHK_ERR_RET(rval);

        // Create map of element handles to range indices
        int idx = 0;
        for (auto eIt = sourceElements.begin(), end = sourceElements.end(); eIt != end; ++eIt)
        {
            sourceEltIdxMap[*eIt] = idx;
            idx++;
        }
    }

    void loadTargetMesh(string filename_)
    {
        fmt::print(stderr, "MBReader: Reading remap target file {}\n", filename_);
        targetFilename = filename_;

        targetMb->create_meshset(MESHSET_SET, targetFileSet);
        rval = targetMb->load_file(targetFilename.c_str(), &targetFileSet); MB_CHK_ERR_RET(rval);

        // Get vertices
        rval = targetMb->get_entities_by_type(targetFileSet, MBVERTEX, targetVertices); MB_CHK_ERR_RET(rval);

        // Get vertex coordinates
        targetCoords.resize(targetVertices.size()*3);
        rval = targetMb->get_coords(targetVertices, targetCoords.data()); MB_CHK_ERR_RET(rval);

        // Get 2D mesh element range
        Range initialTargetElements;
        rval = targetMb->get_entities_by_dimension(targetFileSet, 2, initialTargetElements); MB_CHK_ERR_RET(rval);

        // // Filter to only those elements with ROMS_Mask==1
        // Tag romsMaskTag;
        // vector<unsigned long> romsMaskData(initialTargetElements.size());
        // rval = targetMb->tag_get_handle("ROMS_Mask", romsMaskTag); MB_CHK_ERR_RET(rval);
        // rval = targetMb->tag_get_data(romsMaskTag, initialTargetElements, romsMaskData.data()); MB_CHK_ERR_RET(rval);
        // for (int i = 0; i < initialTargetElements.size(); i++)        {
        //     if (romsMaskData[i] == 1)
        //     {
        //         targetElements.insert(initialTargetElements[i]);
        //     }
        // }

        // fmt::print(stderr, "Initial target elements: {}, ROMS_Mask==1 target elements: {}\n", initialTargetElements.size(), targetElements.size());

        // Filter to only those elements with ROMS_Mask==1 (robust)
        Tag romsMaskTag;
        rval = targetMb->tag_get_handle("ROMS_Mask", romsMaskTag); MB_CHK_ERR_RET(rval);

        int tagLen = 0;
        rval = targetMb->tag_get_length(romsMaskTag, tagLen); MB_CHK_ERR_RET(rval);
        fmt::print(stderr, "ROMS_Mask tag length = {}\n", tagLen);

        size_t N = initialTargetElements.size();
        int stride = (tagLen > 0) ? tagLen : 1;
        vector<int> romsMaskData;
        romsMaskData.resize(N * stride);
        rval = targetMb->tag_get_data(romsMaskTag, initialTargetElements, romsMaskData.data()); MB_CHK_ERR_RET(rval);

        size_t count = 0;
        for (size_t i = 0; i < N; ++i)
        {
            unsigned long val = romsMaskData[i * stride]; // component 0
            if (val == 1)
            {
                targetElements.insert(initialTargetElements[i]);
                ++count;
            }
            if (i < 20)
                fmt::print(stderr, "entity {} -> ROMS_Mask = {}\n", (unsigned long)initialTargetElements[i], val);
        }

        fmt::print(stderr, "Initial target elements: {}, ROMS_Mask==1 target elements: {}\n", N, count);


        // Create map of element handles to range indices
        int idx = 0;
        for (auto eIt = targetElements.begin(), end = targetElements.end(); eIt != end; ++eIt)
        {
            targetEltIdxMap[*eIt] = idx;
            idx++;
        }
    }

    // // NOTE: In 3D, this only selects polygonal boundary faces, since this is the structure of the expected
    // //       MPAS data. However, 3D mesh boundary could be MBPOLYGON, MBQUAD, or MBTRI.
    // void loadSourceBdry()
    // {
    //     if (!sourceFileSet)
    //     {
    //         fmt::print(stderr, "MBReader Error: Cannot load source boundary, source file has not been read.\n");
    //         exit(1);
    //     }
        
    //     Range faces;
    //     vector<EntityHandle> neighbors;

    //     // Get set of all faces. For a 2D mesh, the bdry is edges. In 3D it could be several cell types.
    //     // Right now we only expect polygons but this should be made more generic.
    //     if (cellDim == 2)
    //     {
    //         rval = mb->get_entities_by_type(sourceFileSet, MBEDGE, faces); MB_CHK_ERR_RET(rval);
    //     }
    //     else if (cellDim == 3)
    //     {
    //         rval = mb->get_entities_by_type(sourceFileSet, MBPOLYGON, faces); MB_CHK_ERR_RET(rval);
    //     }
    //     else
    //     {
    //         fmt::print(stderr, "ERROR: Unsupported cellDim in MBReader::loadSourceBdry()\n");
    //         exit(1);
    //     }

    //     // Search through faces and only keep those with one parent cell
    //     for (auto it = faces.begin(), end = faces.end(); it != end; ++it)
    //     {
    //         neighbors.clear();
    //         mb->get_adjacencies( &(*it) , 1, 3, false, neighbors);
    //         if (neighbors.size() == 1)
    //         {
    //             sourceBdryFaces.push_back(*it);
    //         }
    //         else if (neighbors.size() == 0)
    //         {
    //             fmt::print(stderr, "ERROR: Found face with no parent cells\n");
    //             exit(1);
    //         }
    //     }
    // }

    void sourceVertexBounds(VectorX<V>& mins, VectorX<V>& maxs)
    {
        if (sourceVertices.size() == 0)
        {
            fmt::print(stderr, "Warning: Attempted to compute sourceVertexBounds but no vertices found.\n");
        }

        mins = VectorX<V>::Zero(geomDim);
        maxs = VectorX<V>::Zero(geomDim);
        for (int j = 0; j < geomDim; j++)
        {
            mins(j) = sourceCoords[j];
            maxs(j) = sourceCoords[j];
        }
        for (int i = 0; i < sourceVertices.size(); i++)
        {
            for (int j = 0; j < geomDim; j++)
            {
                if (sourceCoords[geomDim*i+j] < mins(j)) mins(j) = sourceCoords[geomDim*i+j];
                if (sourceCoords[geomDim*i+j] > maxs(j)) maxs(j) = sourceCoords[geomDim*i+j];
            }
        }
    }

    void targetVertexBounds(VectorX<V>& mins, VectorX<V>& maxs)
    {
        if (targetVertices.size() == 0)
        {
            fmt::print(stderr, "Warning: Attempted to compute targetVertexBounds but no vertices found.\n");
        }

        mins = VectorX<V>::Zero(geomDim);
        maxs = VectorX<V>::Zero(geomDim);
        for (int j = 0; j < geomDim; j++)
        {
            mins(j) = targetCoords[j];
            maxs(j) = targetCoords[j];
        }
        for (int i = 0; i < targetVertices.size(); i++)
        {
            for (int j = 0; j < geomDim; j++)
            {
                if (targetCoords[geomDim*i+j] < mins(j)) mins(j) = targetCoords[geomDim*i+j];
                if (targetCoords[geomDim*i+j] > maxs(j)) maxs(j) = targetCoords[geomDim*i+j];
            }
        }
    }
};



// Climate-block
template <typename T>
struct CBlock2 : public BlockBase<T>
{
    using Base = BlockBase<T>;
    using Base::dom_dim;
    using Base::pt_dim;
    using Base::core_mins;
    using Base::core_maxs;
    using Base::bounds_mins;
    using Base::bounds_maxs;
    using Base::overlaps;
    using Base::input;
    using Base::approx;
    using Base::errs;
    using Base::mfa;

    const int geomDim;
    int verbose;
    vector<string> varNames;

    bool dumpMatrices = false;
    bool addBdryData = true;
    bool writeLegacyRemapTags = true;
    bool useDeterministicTargetPlane = false;
    bool combineScienceVariables = false;

    // Source (MPAS) mesh: ECEF Cartesian in meters. Target (ROMS) mesh:
    // unit sphere. We rescale target coords into MPAS's ECEF frame on read
    // so source/target bounding boxes share a common scale. The MOAB
    // target mesh is left in unit-sphere units (writeVTK preserves the
    // original target frame).
    static constexpr T R_earth = 6371220.0;

    mfa::PointSet<T>*   mpas_input;
    mfa::PointSet<T>*   mpas_approx;
    mfa::PointSet<T>*   mpas_error;
    mfa::PointSet<T>*   roms_input;
    mfa::MFA<T>*        reverse_mfa;
    mfa::PointSet<T>*   roundtrip_approx;

    MBReader<T>* mbr;

    mfa::Bbox<T> sourceBox;
    mfa::Bbox<T> targetBox;
    mfa::Bbox<T> bbox;
    mfa::AffMap<T> planeMap;
    mfa::BoundingSpace<T> targetBoundingSpace;
    mfa::BoundingSpace<T> sourceBoundingSpace;
    mfa::BoundingSpace<T> totalBoundingSpace;
    VectorX<T> targetPlaneMins;
    VectorX<T> targetPlaneMaxs;
    VectorX<T> sourcePlaneMins;
    VectorX<T> sourcePlaneMaxs;
    
    VectorX<T> sourceHeights;   // sourceHeights[i] is the height of mpas_input->domain(i, 2) relative to average sea height. Positive is above sea level.
    VectorX<T> targetHeights;
    vector<T> sourceLevelMeanHeights;

    T minHeight;
    T maxHeight;
    T minSourceHeight;
    T maxSourceHeight;
    T minTargetHeight;
    T maxTargetHeight;

    VectorX<T> sourceMins;
    VectorX<T> sourceMaxs;
    VectorX<T> targetMins;
    VectorX<T> targetMaxs;
    VectorX<T> bboxMins;
    VectorX<T> bboxMaxs;

    int maxLevel;
    int numSourceElements;
    int numSourceCells;
    Range selectedSourceElements;
    vector<int> sourceElementLevels;    // [numSourceElements]
    vector<vector<int>> sourceCellIDs;  // [numSourceElements][maxLevel]
    vector<char> sourceElementIsCore;   // [numSourceElements]
    vector<T> sourceCoverageFractions;  // [numSourceElements]
    vector<int> sourceCoverageClasses;  // 0=uncovered, 1=partial, 2=full, 3=halo
    vector<char> sourcePointEligible;   // [numSourceCells]

    // zero-initialize pointers during default construction
    CBlock2() :
        Base(),
        geomDim(3),
        verbose(0),
        maxLevel(0),
        numSourceElements(0),
        numSourceCells(0),
        mpas_input(nullptr),
        mpas_approx(nullptr),
        mpas_error(nullptr),
        roms_input(nullptr),
        reverse_mfa(nullptr),
        roundtrip_approx(nullptr),
        mbr(nullptr)
    { 
        sourceMins = VectorX<T>::Zero(geomDim);
        sourceMaxs = VectorX<T>::Zero(geomDim);
        targetMins = VectorX<T>::Zero(geomDim);
        targetMaxs = VectorX<T>::Zero(geomDim);
        bboxMins = VectorX<T>::Zero(geomDim);
        bboxMaxs = VectorX<T>::Zero(geomDim);
    }

    virtual ~CBlock2()
    {
        delete mpas_input;
        delete mpas_approx;
        delete mpas_error;
        delete roms_input;
        delete reverse_mfa;
        delete roundtrip_approx;
        delete mbr;
    }

    static
        void* create()              { return mfa::create<CBlock2>(); }

    static
        void destroy(void* b)       { mfa::destroy<CBlock2>(b); }

    static
        void add(                                   // add the block to the decomposition
            int                 gid,                // block global id
            const Bounds<T>&    core,               // block bounds without any ghost added
            const Bounds<T>&    bounds,             // block bounds including any ghost region added
            const Bounds<T>&    domain,             // global data bounds
            const RCLink<T>&    link,               // neighborhood
            diy::Master&        master,             // diy master
            int                 dom_dim,            // domain dimensionality
            int                 pt_dim,             // point dimensionality
            T                   ghost_factor = 0.0) // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    {
        mfa::add<CBlock2, T>(gid, core, bounds, domain, link, master, dom_dim, pt_dim, ghost_factor);
    }

    static
        void save(const void* b_, diy::BinaryBuffer& bb)    { mfa::save<CBlock2, T>(b_, bb); }
    static
        void load(void* b_, diy::BinaryBuffer& bb)          { mfa::load<CBlock2, T>(b_, bb); }

    void initMOAB(MPI_Comm local, int dim)
    {
        mbr = new MBReader<T>(local, dim);
    }

    void computeBbox3D()
    {
        sourceBox = mfa::Bbox<T>(*mpas_input);
        targetBox = mfa::Bbox<T>(*roms_input);
        bbox = sourceBox.merge(targetBox);

        if (verbose >= 1) bbox.print();
    }

    void computeBbox2D()
    {
        // Compute bounding boxes for source and target data
        if (dom_dim == 2)   // if dom_dim < geomDim, assume the data lies on a plane
        {
            VectorX<T> n = mfa::estimateSurfaceNormal<T>(roms_input->domain.leftCols(3));
            auto [a, b] = mfa::getPlaneVectors<T>(n);

            if (verbose >= 2)
            {
                fmt::print(stderr, "Target Orientation:\n");
                fmt::print(stderr, "  a: {}\n", mfa::print_vec(a));
                fmt::print(stderr, "  b: {}\n", mfa::print_vec(b));
                fmt::print(stderr, "  n: {}\n", mfa::print_vec(n));
            }

            targetBox = mfa::Bbox<T>({a, b, n}, *roms_input);
            sourceBox = mfa::Bbox<T>(targetBox.basis, *mpas_input); // Source box gets the same orientation as target
        }
        else    // if dom_dim == geomDim, use axis-aligned bounding boxes
        {
            targetBox = mfa::Bbox<T>(*roms_input);
            sourceBox = mfa::Bbox<T>(*mpas_input);
        }

        // Compute superset of targetBox and sourceBox
        bbox = targetBox.merge(sourceBox); 
        
        // Ensure that the full bounding box is a superset of both domains
        // n.b. this debug step could be slow as it loops completely through both data sets
        // Bbox::merge round-trips coords through basis*basis^T which is numerically lossy at
        // ULP scale (~1.4e-9 at ECEF magnitudes); the default prec=1e-12 is ~1000x tighter
        // than ULP at this scale, so we relax it to a scale-aware tolerance.
        const T prec = R_earth * 1000 * std::numeric_limits<T>::epsilon();
        bbox.pad(prec);
        if (!bbox.doesContain(*mpas_input, 2, prec))
        {
            fmt::print(stderr, "ERROR: Bounding box does not contain all of source data\n");
            exit(1);
        }
        if (!bbox.doesContain(*roms_input, 2, prec))
        {
            fmt::print(stderr, "ERROR: Bounding box does not contain all of target data\n");
            exit(1);
        }

        if (verbose >= 1) bbox.print();
    }

    // Check if a source polygon intersects the source read bbox in xyz space.
    // The element box is formed from its vertex coordinates and tested in the
    // bbox's rotated basis for compatibility with oriented boxes.
    bool elementIntersectsSourceReadBbox(
            EntityHandle         element,
            const mfa::BoundingSpace<T>&  boundingSpace) const
    {
        // Get array of vertex element handles
        const EntityHandle* conn = nullptr;
        int nconn = 0;
        ErrorCode rval = mbr->sourceMb->get_connectivity(element, conn, nconn);
        if (rval != MB_SUCCESS || !conn || nconn == 0)
        {
            fmt::print(stderr, "ERROR: Failed to get element connectivity in elementIntersectsSourceReadBbox().\n");
            exit(1);
        }

        // Get vector of coordinates for all element vertices
        vector<T> coords(geomDim * nconn);
        rval = mbr->sourceMb->get_coords(conn, nconn, coords.data());
        if (rval != MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to get element coordinates in elementIntersectsSourceReadBbox().\n");
            exit(1);
        }

        const int refDim = boundingSpace.spaceDim;
        VectorX<T> elemMins = VectorX<T>::Constant(refDim, numeric_limits<T>::max());
        VectorX<T> elemMaxs = VectorX<T>::Constant(refDim, numeric_limits<T>::lowest());
        VectorX<T> xyz(geomDim);
        VectorX<T> rot;

        // Compute min/max extents of vertices in readBoundingSpace's rotated basis
        for (int i = 0; i < nconn; i++)
        {
            for (int j = 0; j < geomDim; j++)
            {
                xyz(j) = coords[geomDim * i + j];
            }

            boundingSpace.toRefSpace(xyz, rot);
            elemMins = elemMins.cwiseMin(rot);
            elemMaxs = elemMaxs.cwiseMax(rot);
        }

        // Check if any vertex extents exceed bounding space
        for (int j = 0; j < refDim; j++)
        {
            if (elemMaxs(j) < boundingSpace.refMins(j) ||
                elemMins(j) > boundingSpace.refMaxs(j))
            {
                return false;
            }
        }

        // if the element is more than halfway around the globe from the plane, it's probably on the other side of the sphere and doesn't intersect
        // This is a hack for now, better to just test with original coordinates 
        VectorX<T> planePoint;
        boundingSpace.toPhysicalSpace(rot, planePoint);
        T distanceToPlane = (xyz - planePoint).norm();
        if (distanceToPlane > R_earth / 2)   
        {
            return false;
        }

        return true;
    }

    // Find 1-ring neighbors that share an edge with the input element.
    // We gather adjacent cells through each vertex, then keep only candidates
    // sharing at least two vertices (polygon edge-neighbor criterion).
    vector<int> getElementNeighborsBySharedEdge(EntityHandle element) const
    {
        const EntityHandle* conn = nullptr;
        int nconn = 0;
        ErrorCode rval = mbr->sourceMb->get_connectivity(element, conn, nconn);
        if (rval != MB_SUCCESS || !conn || nconn == 0)
        {
            fmt::print(stderr, "ERROR: Failed to get element connectivity in getElementNeighborsBySharedEdge().\n");
            exit(1);
        }

        unordered_map<int, int> sharedVertexCount;
        for (int i = 0; i < nconn; i++)
        {
            EntityHandle vertex = conn[i];
            Range vertexAdjacentCells;
            rval = mbr->sourceMb->get_adjacencies(&vertex, 1, 2, false, vertexAdjacentCells);
            if (rval != MB_SUCCESS)
            {
                fmt::print(stderr, "ERROR: Failed to get vertex adjacencies in getElementNeighborsBySharedEdge().\n");
                exit(1);
            }

            for (auto it = vertexAdjacentCells.begin(), end = vertexAdjacentCells.end(); it != end; ++it)
            {
                if (*it == element)
                {
                    continue;
                }

                auto idxIt = mbr->sourceEltIdxMap.find(*it);
                if (idxIt != mbr->sourceEltIdxMap.end())
                {
                    sharedVertexCount[idxIt->second]++;
                }
            }
        }

        vector<int> neighbors;
        neighbors.reserve(sharedVertexCount.size());
        for (const auto& kv : sharedVertexCount)
        {
            if (kv.second >= 2)
            {
                neighbors.push_back(kv.first);
            }
        }

        return neighbors;
    }

    // Select source elements that intersect the requested bbox and then add
    // a one-ring halo around boundary elements.
    void selectSourceElements(const mfa::BoundingSpace<T>& readBoundingSpace)
    {
        const int totalElements = static_cast<int>(mbr->sourceElements.size());
        vector<EntityHandle> sourceElementHandles(totalElements);

        int idx = 0;
        for (auto it = mbr->sourceElements.begin(), end = mbr->sourceElements.end(); it != end; ++it, ++idx)
        {
            sourceElementHandles[idx] = *it;
        }

        vector<char> selected(totalElements, 0);
        vector<char> coreSelected(totalElements, 0);
        vector<int> coreIndices;
        coreIndices.reserve(totalElements);

        for (int i = 0; i < totalElements; i++)
        {
            if (elementIntersectsSourceReadBbox(sourceElementHandles[i], readBoundingSpace))
            {
                selected[i] = 1;
                coreSelected[i] = 1;
                coreIndices.push_back(i);
            }
        }

        if (coreIndices.empty())
        {
            fmt::print(stderr, "ERROR: source read bbox does not intersect any source elements.\n");
            exit(1);
        }

        vector<int> boundaryCoreIndices;
        boundaryCoreIndices.reserve(coreIndices.size());
        for (int coreIdx : coreIndices)
        {
            auto neighbors = getElementNeighborsBySharedEdge(sourceElementHandles[coreIdx]);

            bool isBoundary = false;
            for (int nbrIdx : neighbors)
            {
                if (!selected[nbrIdx])
                {
                    isBoundary = true;
                    break;
                }
            }

            if (isBoundary)
            {
                boundaryCoreIndices.push_back(coreIdx);
            }
        }

        vector<char> haloSelected(totalElements, 0);
        for (int coreIdx : boundaryCoreIndices)
        {
            auto neighbors = getElementNeighborsBySharedEdge(sourceElementHandles[coreIdx]);
            for (int nbrIdx : neighbors)
            {
                haloSelected[nbrIdx] = 1;
                selected[nbrIdx] = 1;
            }
        }

        selectedSourceElements.clear();
        sourceElementIsCore.clear();
        int haloCount = 0;
        for (int i = 0; i < totalElements; i++)
        {
            if (selected[i])
            {
                selectedSourceElements.insert(sourceElementHandles[i]);
                sourceElementIsCore.push_back(coreSelected[i]);
                if (!coreSelected[i] && haloSelected[i])
                {
                    haloCount++;
                }
            }
        }

        numSourceElements = static_cast<int>(selectedSourceElements.size());
        if (numSourceElements == 0)
        {
            fmt::print(stderr, "ERROR: Empty selected source element set after bbox/halo selection.\n");
            exit(1);
        }

        if (verbose >= 1)
        {
            fmt::print(stderr, "Selected source elements: {} (core={} halo={})\n",
                numSourceElements, coreIndices.size(), haloCount);
        }
    }

    struct Point2
    {
        T x;
        T y;
    };

    struct Polygon2
    {
        vector<Point2> points;
        T minX;
        T maxX;
        T minY;
        T maxY;
    };

    using BoostPoint2 = boost::geometry::model::d2::point_xy<T>;
    using BoostPolygon2 = boost::geometry::model::polygon<BoostPoint2>;
    using BoostMultiPolygon2 = boost::geometry::model::multi_polygon<BoostPolygon2>;

    static BoostPolygon2 toBoostPolygon(const vector<Point2>& points)
    {
        BoostPolygon2 polygon;
        for (const Point2& point : points)
            boost::geometry::append(polygon.outer(), BoostPoint2(point.x, point.y));
        boost::geometry::correct(polygon);
        return polygon;
    }

    static T signedPolygonArea(const vector<Point2>& polygon)
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

    Polygon2 projectedElement(Interface* mb, EntityHandle element, T coordinateScale) const
    {
        const EntityHandle* connectivity = nullptr;
        int count = 0;
        ErrorCode rval = mb->get_connectivity(element, connectivity, count);
        if (rval != MB_SUCCESS || !connectivity || count < 3)
        {
            fmt::print(stderr, "ERROR: Invalid polygon connectivity for entity {}.\n", element);
            exit(1);
        }

        vector<T> coordinates(count * geomDim);
        rval = mb->get_coords(connectivity, count, coordinates.data());
        if (rval != MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to read polygon coordinates for entity {}.\n", element);
            exit(1);
        }

        Polygon2 polygon;
        polygon.minX = polygon.minY = numeric_limits<T>::max();
        polygon.maxX = polygon.maxY = numeric_limits<T>::lowest();
        polygon.points.reserve(count);
        for (int i = 0; i < count; i++)
        {
            VectorX<T> xyz(geomDim);
            for (int dim = 0; dim < geomDim; dim++)
                xyz(dim) = coordinates[i * geomDim + dim] * coordinateScale;
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

    void classifySourceCoverage()
    {
        const int targetCount = mbr->targetElements.size();
        if (targetCount == 0 || numSourceElements == 0)
        {
            fmt::print(stderr, "ERROR: Cannot classify coverage with empty source or target elements.\n");
            exit(1);
        }

        vector<Polygon2> targetPolygons;
        vector<BoostPolygon2> boostTargetPolygons;
        targetPolygons.reserve(targetCount);
        boostTargetPolygons.reserve(targetCount);
        T gridMinX = numeric_limits<T>::max();
        T gridMaxX = numeric_limits<T>::lowest();
        T gridMinY = numeric_limits<T>::max();
        T gridMaxY = numeric_limits<T>::lowest();
        for (EntityHandle element : mbr->targetElements)
        {
            targetPolygons.push_back(projectedElement(mbr->targetMb, element, R_earth));
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
        vector<vector<int>> bins(gridSize * gridSize);
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
        vector<T> targetProfileMins(targetCount, numeric_limits<T>::max());
        vector<T> targetProfileMaxs(targetCount, numeric_limits<T>::lowest());
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

        sourceCoverageFractions.assign(numSourceElements, 0);
        sourceCoverageClasses.assign(numSourceElements, 0);
        sourcePointEligible.assign(numSourceCells, 0);
        vector<int> candidateStamp(targetCount, -1);
        int fullCount = 0;
        int partialCount = 0;
        int uncoveredCount = 0;
        int haloCount = 0;
        int eligiblePointCount = 0;
        const T fullCoverageTolerance = static_cast<T>(1e-6);

        int sourceIdx = 0;
        for (EntityHandle sourceElement : selectedSourceElements)
        {
            const Polygon2 sourcePolygon = projectedElement(mbr->sourceMb, sourceElement, 1);
            const T sourceArea = std::abs(signedPolygonArea(sourcePolygon.points));
            if (sourceArea <= 0)
            {
                fmt::print(stderr, "ERROR: Degenerate projected MPAS polygon {}.\n", sourceElement);
                exit(1);
            }

            const int minI = binIndex(sourcePolygon.minX, gridMinX, cellWidth);
            const int maxI = binIndex(sourcePolygon.maxX, gridMinX, cellWidth);
            const int minJ = binIndex(sourcePolygon.minY, gridMinY, cellHeight);
            const int maxJ = binIndex(sourcePolygon.maxY, gridMinY, cellHeight);
            vector<int> candidates;
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

            T supportedMinHeight = numeric_limits<T>::lowest();
            T supportedMaxHeight = numeric_limits<T>::max();
            bool hasPositiveOverlap = false;
            BoostMultiPolygon2 coveredRegion;
            const BoostPolygon2 boostSourcePolygon = toBoostPolygon(sourcePolygon.points);
            for (int targetIdx : candidates)
            {
                const Polygon2& targetPolygon = targetPolygons[targetIdx];
                if (targetPolygon.maxX < sourcePolygon.minX || targetPolygon.minX > sourcePolygon.maxX ||
                    targetPolygon.maxY < sourcePolygon.minY || targetPolygon.minY > sourcePolygon.maxY)
                    continue;

                vector<BoostPolygon2> intersections;
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
            sourceCoverageFractions[sourceIdx] = coverageFraction;
            if (!sourceElementIsCore[sourceIdx])
            {
                sourceCoverageClasses[sourceIdx] = 3;
                haloCount++;
            }
            else if (!hasPositiveOverlap || coverageFraction <= fullCoverageTolerance)
            {
                sourceCoverageClasses[sourceIdx] = 0;
                uncoveredCount++;
            }
            else if (coverageFraction < static_cast<T>(1) - fullCoverageTolerance)
            {
                sourceCoverageClasses[sourceIdx] = 1;
                partialCount++;
            }
            else
            {
                sourceCoverageClasses[sourceIdx] = 2;
                fullCount++;
                if (supportedMinHeight <= supportedMaxHeight)
                {
                    for (int level = 0; level < sourceElementLevels[sourceIdx]; level++)
                    {
                        const int pointIdx = sourceCellIDs[sourceIdx][level];
                        const T height = sourceHeights(pointIdx);
                        if (height >= supportedMinHeight && height <= supportedMaxHeight)
                        {
                            sourcePointEligible[pointIdx] = 1;
                            eligiblePointCount++;
                        }
                    }
                }
            }
            sourceIdx++;
        }

        fmt::print(stderr,
                   "MPAS wet-ROMS coverage: full={} partial={} uncovered={} halo={}; eligible active levels={}\n",
                   fullCount, partialCount, uncoveredCount, haloCount, eligiblePointCount);
    }

    // Compute the maximum number of depth levels over all cells in the MPAS mesh
    // Initializes:
    //   * maxLevel
    //   * sourceElementLevels
    void computeSourceLevelInfo()
    {
        if (selectedSourceElements.empty())
        {
            fmt::print(stderr, "ERROR: computeSourceLevelInfo called with empty selectedSourceElements.\n");
            exit(1);
        }

        Tag maxLevelCellTag;
        numSourceElements = selectedSourceElements.size();
        sourceElementLevels.resize(numSourceElements);
        mbr->sourceMb->tag_get_handle("maxLevelCell", maxLevelCellTag);
        mbr->sourceMb->tag_get_data(maxLevelCellTag, selectedSourceElements, sourceElementLevels.data());

        maxLevel = 0;
        for (int i = 0; i < sourceElementLevels.size(); i++)
        {
            if (sourceElementLevels[i] > maxLevel)
                maxLevel = sourceElementLevels[i];
        }
    }

    void computeSourceCellInfo()
    {
        sourceCellIDs.resize(numSourceElements);
        for (int i = 0; i < numSourceElements; i++)
        {
            // Compute and store cell IDs for each depth level of this element
            sourceCellIDs[i].resize(sourceElementLevels[i]);
            for (int level = 0; level < sourceElementLevels[i]; level++)
            {
                int cellID = numSourceCells + level;
                sourceCellIDs[i][level] = cellID;
            }

            // Increment total 3D cell count
            numSourceCells += sourceElementLevels[i];
        }
    }

    template<typename V>
    void addSourceVariable(
        const   diy::Master::ProxyWithLink& cp,
                string filename,
                string varName,
                int    varIdx)
    {
        (void)cp;
        (void)filename;

        Tag dataTag;
        mbr->sourceMb->tag_get_handle(varName.c_str(), dataTag);
        int tagLength = 0;
        mbr->sourceMb->tag_get_length(dataTag, tagLength);

        if (tagLength <= 0)
        {
            fmt::print(stderr, "ERROR: Tag '{}' has invalid length {}.\n", varName, tagLength);
            exit(1);
        }

        vector<T> dataVec(numSourceElements * tagLength);
        mbr->sourceMb->tag_get_data(dataTag, selectedSourceElements, dataVec.data());

        for (int elementID = 0; elementID < numSourceElements; elementID++)
        {
            for (int level = 0; level < sourceElementLevels[elementID]; level++)
            {
                int cellID = sourceCellIDs[elementID][level];       // ID in set of real 3D cells
                if (tagLength == 1)
                {
                    mpas_input->domain(cellID, geomDim + varIdx) = dataVec[elementID];
                }
                else
                {
                    if (level >= tagLength)
                    {
                        fmt::print(stderr, "ERROR: Requested level {} from tag '{}' length {}.\n",
                                   level, varName, tagLength);
                        exit(1);
                    }

                    int fullCellID = elementID * tagLength + level;     // includes masked levels
                    mpas_input->domain(cellID, geomDim + varIdx) = dataVec[fullCellID];
                }
            }
        }

        // // Optionally add data for each boundary face
        // if (addBdryData)
        // {
        //     vector<EntityHandle> parent;
        //     int offset = mbr->sourceElements.size();
        //     for (int j = 0; j < mbr->sourceBdryFaces.size(); j++)
        //     {
        //         // Get tag data associated to parent cell
        //         parent.clear();
        //         mbr->mb->get_adjacencies(&mbr->sourceBdryFaces[j], 1, 3, false, parent);
        //         if (parent.size() != 1)
        //         {
        //             fmt::print(stderr, "ERROR: Expected a single parent element\n");
        //             exit(1);
        //         }

        //         int parentIdx = mbr->sourceEltIdxMap[parent[0]];
        //         mpas_input->domain(offset + j, geomDim + varIdx) = dataVec[parentIdx];
        //     }
        // }
    }

    template <typename V>
    void readSourceData(
            const   diy::Master::ProxyWithLink& cp,
                    string filename)
    {
        // if (!roms_input)
        // {
        //     fmt::print(stderr, "ERROR: Cannot read source data before target data when remapping.\n");
        //     fmt::print(stderr, "       Source data model needs to know the bounding box of the target data ahead of time.\n");
        //     fmt::print(stderr, "       Exiting.\n");
        //     exit(1);
        // }

        if (dom_dim != 3)
        {
            fmt::print(stderr, "ERROR: Set dom_dim to 3 (remove this after debugging).\n");
            exit(1);
        }

        mfa::Bbox<T> sourceReadBbox;

        mbr->loadSourceMesh(filename);
        // mbr->loadSourceBdry();
        mbr->sourceVertexBounds(sourceMins, sourceMaxs);
        if (verbose >= 1)
        {
            // mfa::print_bbox(sourceMins, sourceMaxs, "Source Bounding");
            targetBoundingSpace.print("Target Bounding Space");
        }

        // Fresh source metadata for each load.
        numSourceCells = 0;
        numSourceElements = 0;
        maxLevel = 0;
        sourceCellIDs.clear();
        sourceElementLevels.clear();
        selectedSourceElements.clear();

        // Select the source chunk to read.
        // Sets: numSourceElements, selectedSourceElements
        selectSourceElements(targetBoundingSpace);

        // Loop through selectedSourceElements and compute the bounding box of their centroids in the plane of the target data. 
        MatrixX<T> sourcePointLocations(numSourceElements, geomDim);
        vector<T> ptCoords(geomDim);
        int i = 0;
        for (auto it = selectedSourceElements.begin(), end = selectedSourceElements.end(); it != end; ++it, ++i)
        {
            mbr->sourceMb->get_coords(&(*it), 1, ptCoords.data());
            for (int j = 0; j < geomDim; j++)
            {
                sourcePointLocations(i, j) = ptCoords[j];
            }
        }
        sourceBoundingSpace = mfa::BoundingSpace<T>(planeMap, sourcePointLocations);
        totalBoundingSpace = sourceBoundingSpace.merge(targetBoundingSpace);

        targetBoundingSpace.print("Target Bounding Space");
        sourceBoundingSpace.print("Source Bounding Space");
        totalBoundingSpace.print("Total Bounding Space");

        // Compute metadata about extruded 3d mesh
        computeSourceLevelInfo();
        computeSourceCellInfo();

        // Set up MFA PointSet
        VectorXi modelDims = combineScienceVariables ? VectorXi(2) :
                                                        VectorXi::Ones(varNames.size() + 1);
        modelDims(0) = 3; // points are 3D
        if (combineScienceVariables)
            modelDims(1) = varNames.size();

        // model_dims << geomDim, 1, 1;          // geometry, salinity, temperature
        mpas_input = new mfa::PointSet<T>(dom_dim, modelDims, numSourceCells);

        // Read geometry drivers for extrusion
        Tag sshTag;
        Tag layerThicknessTag;
        mbr->sourceMb->tag_get_handle("timeDaily_avg_ssh", sshTag);
        mbr->sourceMb->tag_get_handle("timeDaily_avg_layerThickness_3d", layerThicknessTag);

        int layerTagLength = 0;
        mbr->sourceMb->tag_get_length(layerThicknessTag, layerTagLength);

        if (layerTagLength <= 0)
        {
            fmt::print(stderr, "ERROR: Invalid layerThickness tag length {}\n", layerTagLength);
            exit(1);
        }

        vector<T> sshData(numSourceElements);
        vector<T> layerThicknessData(numSourceElements * layerTagLength);
        mbr->sourceMb->tag_get_data(sshTag, selectedSourceElements, sshData.data());
        mbr->sourceMb->tag_get_data(layerThicknessTag, selectedSourceElements, layerThicknessData.data());

        // Build one point per active extruded cell. The top cell spans from
        // SSH to -layerThickness[0]; lower cells continue from that bottom.
        int elementID = 0;
        minSourceHeight = numeric_limits<T>::max();
        maxSourceHeight = numeric_limits<T>::lowest();
        sourceHeights.resize(numSourceCells);
        vector<T> sourceLevelHeightSums(maxLevel, 0.0);
        vector<int> sourceLevelHeightCounts(maxLevel, 0);
        vector<V> centroid(geomDim);
        for (auto it = selectedSourceElements.begin(), end = selectedSourceElements.end(); it != end; ++it, ++elementID)
        {
            mbr->sourceMb->get_coords(&(*it), 1, centroid.data());

            VectorX<T> normal = VectorX<T>::Zero(geomDim);
            T norm2 = 0.0;
            for (int k = 0; k < geomDim; k++)
            {
                normal(k) = static_cast<T>(centroid[k]);
                norm2 += normal(k) * normal(k);
            }

            if (norm2 <= 0.0)
            {
                fmt::print(stderr, "ERROR: Degenerate centroid while building source geometry.\n");
                exit(1);
            }
            normal /= sqrt(norm2);

            T cumulativeThickness = 0.0;
            for (int level = 0; level < sourceElementLevels[elementID]; level++)
            {
                if (level >= layerTagLength)
                {
                    fmt::print(stderr, "ERROR: sourceElementLevels exceeds layerThickness tag length.\n");
                    exit(1);
                }

                int cellID = sourceCellIDs[elementID][level];

                T layerThickness = layerThicknessData[elementID * layerTagLength + level];
                const T top = level == 0 ? sshData[elementID] : -cumulativeThickness;
                const T bottom = -(cumulativeThickness + layerThickness);
                T axialCentroid = 0.5 * (top + bottom);

                for (int k = 0; k < geomDim; k++)
                {
                    mpas_input->domain(cellID, k) = static_cast<T>(centroid[k]) + axialCentroid * normal(k);
                }

                // Set point height for later use in calculating parameterizations
                sourceHeights(cellID) = axialCentroid;
                sourceLevelHeightSums[level] += axialCentroid;
                sourceLevelHeightCounts[level]++;

                cumulativeThickness += layerThickness;
            }

            if (sshData[elementID] > maxSourceHeight)
            {
                maxSourceHeight = sshData[elementID];
            }
            if (-1 * cumulativeThickness < minSourceHeight)
            {
                minSourceHeight = -1 * cumulativeThickness;
            }
        }

        sourceLevelMeanHeights.clear();
        sourceLevelMeanHeights.reserve(maxLevel);
        for (int level = 0; level < maxLevel; level++)
        {
            if (sourceLevelHeightCounts[level] > 0)
            {
                sourceLevelMeanHeights.push_back(sourceLevelHeightSums[level] /
                                                 sourceLevelHeightCounts[level]);
            }
        }

        // Add data values to PointSet
        for (int l = 0; l < varNames.size(); l++)
        {
            addSourceVariable<V>(cp, filename, varNames[l], l);
        }
    }

    mfa::AffMap<T> computeTargetPlaneMap(const MatrixX<T>& targetPointLocations)
    {
        VectorX<T> center = targetPointLocations.colwise().mean();
        VectorX<T> n = useDeterministicTargetPlane ? center.normalized() :
                       mfa::estimateSurfaceNormal<T>(targetPointLocations);
        auto [a, b] = mfa::getPlaneVectors<T>(n);

        if (verbose >= 2)
        {
            fmt::print(stderr, "Target Orientation:\n");
            fmt::print(stderr, "  a: {}\n", mfa::print_vec(a));
            fmt::print(stderr, "  b: {}\n", mfa::print_vec(b));
            fmt::print(stderr, "  n: {}\n", mfa::print_vec(n));
        }

        return mfa::AffMap<T>(2, 3, center, {a, b});

        // targetBox = mfa::Bbox<T>({a, b, n}, *roms_input);
        // sourceBox = mfa::Bbox<T>(targetBox.basis, *mpas_input); // Source box gets the same orientation as target
    }

    template <typename V>
    void readTargetData(
            const   diy::Master::ProxyWithLink& cp,
                    string  filename)
    {
        const bool decodeAtVertices = false;

        mbr->loadTargetMesh(filename);
        mbr->targetVertexBounds(targetMins, targetMaxs);

        targetMins *= R_earth;
        targetMaxs *= R_earth;

        if (verbose >= 1) mfa::print_bbox(targetMins, targetMaxs, "Target Bounding");

        int npts = 0;
        vector<V> coord(geomDim);
        int i = 0;
        moab::Range::iterator it, end;
        if (decodeAtVertices)
        {
            npts = mbr->targetVertices.size();
            it = mbr->targetVertices.begin();
            end = mbr->targetVertices.end();
        }
        else
        {
            npts = mbr->targetElements.size();
            it = mbr->targetElements.begin();
            end = mbr->targetElements.end();
        }

        // roms_input = new mfa::PointSet<T>(dom_dim, modelDims, npts);
        MatrixX<T> targetPointLocations(npts, geomDim);
        for ( ; it != end; ++it, ++i)
        {
            mbr->targetMb->get_coords(&(*it), 1, coord.data());
            for (int j = 0; j < geomDim; j++)
            {
                // roms_input->domain(i, j) = static_cast<T>(coord[j]) * R_earth;
                targetPointLocations(i, j) = static_cast<T>(coord[j]) * R_earth;
            }
        }

        planeMap = computeTargetPlaneMap(targetPointLocations);
        targetBoundingSpace = mfa::BoundingSpace<T>(planeMap, targetPointLocations);
    }

    template <typename V>
    void readTargetDataFull()
    {
        int numTargetElements = mbr->targetElements.size();

        VectorXi modelDims = combineScienceVariables ? VectorXi(2) :
                                                        VectorXi::Ones(varNames.size() + 1);
        modelDims(0) = geomDim;
        if (combineScienceVariables)
            modelDims(1) = varNames.size();

        // Read geometry drivers for extrusion
        Tag sshTag;
        Tag bathymetryTag;
        Tag locationTag;
        mbr->targetMb->tag_get_handle("SSH", sshTag);
        mbr->targetMb->tag_get_handle("bathymetry", bathymetryTag);
        mbr->targetMb->tag_get_handle("ROMS3DLocations", locationTag);

        int locationTagLength = 0;
        mbr->targetMb->tag_get_length(locationTag, locationTagLength);

        if (locationTagLength <= 0)
        {
            fmt::print(stderr, "ERROR: Invalid location tag length {}\n", locationTagLength);
            exit(1);
        }

        vector<T> sshData(numTargetElements);
        vector<T> bathymetryData(numTargetElements);
        vector<T> locationData(numTargetElements * locationTagLength);
        mbr->targetMb->tag_get_data(sshTag, mbr->targetElements, sshData.data());
        mbr->targetMb->tag_get_data(bathymetryTag, mbr->targetElements, bathymetryData.data());
        mbr->targetMb->tag_get_data(locationTag, mbr->targetElements, locationData.data());

        int i = 0;
        minTargetHeight = numeric_limits<T>::max();
        maxTargetHeight = numeric_limits<T>::lowest();
        targetHeights.resize(numTargetElements * locationTagLength);
        vector<V> coord(geomDim);
        VectorX<T> centroid(geomDim);
        roms_input = new mfa::PointSet<T>(dom_dim, modelDims, numTargetElements * locationTagLength);
        moab::Range::iterator it = mbr->targetElements.begin();
        moab::Range::iterator end = mbr->targetElements.end();
        for ( ; it != end; ++it, ++i)
        {
            // Get 2D element centroid
            mbr->targetMb->get_coords(&(*it), 1, coord.data());
            for (int k = 0; k < geomDim; k++)
            {
                centroid(k) = coord[k] * R_earth;
            }

            // Extrude in the normal direction by the depth value at each location
            const VectorX<T> normal = centroid.normalized();
            for (int j = 0; j < locationTagLength; j++)
            {
                int idx = i * locationTagLength + j;
                T height = -1 * locationData[idx];
                roms_input->domain.block(idx, 0, 1, geomDim) = (centroid + height * normal).transpose();
                targetHeights(idx) = height;
            }

            // Keep track of min and max z-depths
            // SSH is a height, so it has opposite sign convention from bathymetry and depth.
            // Our convention is that as height decreases, we move toward the ocean floor.
            if (sshData[i] > maxTargetHeight)
            {
                maxTargetHeight = sshData[i];
            }
            if (-1 * bathymetryData[i] < minTargetHeight)
            {
                minTargetHeight = -1 * bathymetryData[i];
            }
        }
    }

    void setParameterizations()
    {
        minHeight = std::min(minSourceHeight, minTargetHeight);
        maxHeight = std::max(maxSourceHeight, maxTargetHeight);

        MatrixX<T> sourceParams(mpas_input->npts, dom_dim);
        MatrixX<T> targetParams(roms_input->npts, dom_dim);

        // parameterization object
        mfa::AffineMap<T> paramMap(totalBoundingSpace);

        // Compute planar parameterization for source data
        // Matri<T> xyz(geomDim);
        MatrixX<T> param2d;
        MatrixX<T> xyz = mpas_input->domain.leftCols(geomDim);
        paramMap.transform(xyz, param2d, true);
        sourceParams.leftCols(2) = param2d;
        
        // Compute axial parameterization for source data
        sourceParams.col(2) = (sourceHeights.array() - minHeight) / (maxHeight - minHeight);

        // Compute planar parameterization for target data
        xyz = roms_input->domain.leftCols(geomDim);
        paramMap.transform(xyz, param2d, true);
        // totalBoundingSpace.toRefSpace(xyz.transpose(), param2d.transpose()); TODO need to normalize to [0,1]
        targetParams.leftCols(2) = param2d; 

        // Compute axial parameterization for target data
        targetParams.col(2) = (targetHeights.array() - minHeight) / (maxHeight - minHeight);


        if (verbose >= 2)
        {
            fmt::print(stderr, "Setting source parameters\n");
        }
        // // Check for out-of-bounds parameter entries (small roundoff can push values
        // // slightly outside [0,1]). Report largest violation and clamp to [0,1].
        // {
        //     T tol = static_cast<T>(1e-8);
        //     T maxDev = static_cast<T>(0);
        //     long maxR = -1, maxC = -1;
        //     long rows = sourceParams.rows();
        //     long cols = sourceParams.cols();
        //     long countViolations = 0;
        //     for (long r = 0; r < rows; ++r)
        //     {
        //         for (long c = 0; c < cols; ++c)
        //         {
        //             T v = sourceParams(r, c);
        //             T dev = static_cast<T>(0);
        //             if (v < static_cast<T>(0)) dev = -v;
        //             else if (v > static_cast<T>(1)) dev = v - static_cast<T>(1);
        //             if (dev > tol) ++countViolations;
        //             if (dev > maxDev)
        //             {
        //                 maxDev = dev;
        //                 maxR = r; maxC = c;
        //             }
        //         }
        //     }
        //     if (countViolations > 0)
        //     {
        //         fmt::print(stderr, "Warning: sourceParams has {} out-of-bounds entries; maxDev={} at ({},{}) val={}\n",
        //                    countViolations, maxDev, maxR, maxC, sourceParams(maxR, maxC));
        //     }
        //     sourceParams = sourceParams.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(1));
        // }
        mpas_input->set_custom_params_unstructured(sourceParams, 1e-6);
        
        if (verbose >= 2)
        {
            fmt::print(stderr, "Setting target parameters\n");
        }
        // {
        //     T tol = static_cast<T>(1e-8);
        //     T maxDev = static_cast<T>(0);
        //     long maxR = -1, maxC = -1;
        //     long rows = targetParams.rows();
        //     long cols = targetParams.cols();
        //     long countViolations = 0;
        //     for (long r = 0; r < rows; ++r)
        //     {
        //         for (long c = 0; c < cols; ++c)
        //         {
        //             T v = targetParams(r, c);
        //             T dev = static_cast<T>(0);
        //             if (v < static_cast<T>(0)) dev = -v;
        //             else if (v > static_cast<T>(1)) dev = v - static_cast<T>(1);
        //             if (dev > tol) ++countViolations;
        //             if (dev > maxDev)
        //             {
        //                 maxDev = dev;
        //                 maxR = r; maxC = c;
        //             }
        //         }
        //     }
        //     if (countViolations > 0)
        //     {
        //         fmt::print(stderr, "Warning: targetParams has {} out-of-bounds entries; maxDev={} at ({},{}) val={}\n",
        //                    countViolations, maxDev, maxR, maxC, targetParams(maxR, maxC));
        //     }
            // targetParams = targetParams.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(1));
        // }
        roms_input->set_custom_params_unstructured(targetParams, 1e-6);

        // computeBbox();
        // roms_input->set_domain_params(bbox);
        // mpas_input->set_domain_params(bbox);
    }


    void computeRemap(
        const diy::Master::ProxyWithLink&   cp,
        mfa::MFAInfo&   info)
    {
        // In 3D, the z-knots are set specifically to the depth levels in the MPAS mesh
        if (dom_dim == 3)
        {
            set3DKnots();
        }

        // Compute parametrization of roms points w.r.t mpas domain
        mpas_approx = new mfa::PointSet<T>(roms_input->params, roms_input->model_dims());

        // Write out collocation matrices for encoding and decoding
        if (dumpMatrices)
        {
            mfa->dumpCollocationMatrixEncode(0, mpas_input);
            mfa->dumpCollocationMatrixDecode(0, mpas_approx);
        }

        // // Create MFA model from source data
        // for (int l = 0; l < varNames.size(); l++)
        // {
        //     mfa->FixedEncodeVar(l, *mpas_input, info.regularization, info.reg1and2, info.weighted);
        // }
        const double encodeStart = MPI_Wtime();
        mfa->FixedEncode(*mpas_input, info.regularization, info.reg1and2, info.weighted);
        fmt::print(stderr, "Forward MFA encode time: {:.3f} s\n", MPI_Wtime() - encodeStart);
        
        // // Evaluate MFA at target mesh locations
        // for (int l = 0; l < varNames.size(); l++)
        // {
        //     mfa->DecodeVar(l, *mpas_approx);
        // }
        mpas_approx->domain.leftCols(geomDim) = roms_input->domain.leftCols(geomDim);
        const double decodeStart = MPI_Wtime();
        for (int varIdx = 0; varIdx < mfa->nvars(); varIdx++)
            mfa->DecodeVar(varIdx, *mpas_approx);
        fmt::print(stderr, "Forward MFA science decode time: {:.3f} s\n", MPI_Wtime() - decodeStart);
        fmt::print(stderr, "Approx npts: {}\n", mpas_approx->npts);
        fmt::print(stderr, "Approx model dims: {}\n", mfa::print_vec(mpas_approx->model_dims()));
        fmt::print(stderr, "Approx var 1 min/max: {}/{}\n", mpas_approx->domain.col(geomDim).minCoeff(), mpas_approx->domain.col(geomDim).maxCoeff());
        fmt::print(stderr, "Approx var 1 has nans: {}\n", mpas_approx->domain.col(geomDim).hasNaN());

        Tag locationTag;
        ErrorCode rval = mbr->targetMb->tag_get_handle("ROMS3DLocations", locationTag);
        if (rval != MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to get ROMS3DLocations tag.\n");
            exit(1);
        }
        int locationTagLength = 0;
        rval = mbr->targetMb->tag_get_length(locationTag, locationTagLength);
        if (rval != MB_SUCCESS || locationTagLength <= 0)
        {
            fmt::print(stderr, "ERROR: Invalid ROMS3DLocations tag length {}.\n", locationTagLength);
            exit(1);
        }

        const size_t expectedTargetPoints = mbr->targetElements.size() * locationTagLength;
        if (static_cast<size_t>(mpas_approx->npts) != expectedTargetPoints)
        {
            fmt::print(stderr, "ERROR: Decoded point count {} does not match {} target elements x {} levels.\n",
                       mpas_approx->npts, mbr->targetElements.size(), locationTagLength);
            exit(1);
        }

        if (writeLegacyRemapTags)
        {
            // Preserve the historical output layout for mfa_remap. remap_loop uses
            // writeForwardRemapH5M(), which writes complete array-valued profiles.
            for (int l = 0; l < varNames.size(); l++)
            {
                Tag remapTag;
                string remapTagName = varNames[l] + "_remap";
                vector<T> remapData(mpas_approx->npts);
                mbr->targetMb->tag_get_handle(remapTagName.c_str(), 1, MB_TYPE_DOUBLE, remapTag, MB_TAG_DENSE | MB_TAG_CREAT);
                for (int i = 0; i < mpas_approx->npts; i++)
                {
                    remapData[i] = mpas_approx->domain(i, geomDim + l);
                }
                mbr->targetMb->tag_set_data(remapTag, mbr->targetElements, remapData.data());
            }
        }
        fmt::print(stderr, "ROMS3DLocations tag length: {}\n", locationTagLength);
        fmt::print(stderr, "mpas_approx npts: {}\n", mpas_approx->npts);
        fmt::print(stderr, "Expected ROMS elements: {}\n", mpas_approx->npts / locationTagLength);
        fmt::print(stderr, "Actual ROMS elements: {}\n", mbr->targetElements.size());

        if (writeLegacyRemapTags)
        {
            int nLevels = 15;
            for (int m = 0; m < nLevels; m++)
            {
                for (int l = 0; l < varNames.size(); l++)
                {
                    Tag remapTag;
                    string remapTag1Name = varNames[l] + "_remap_" + to_string(m);
                    vector<T> remapData(mpas_approx->npts / locationTagLength);
                    mbr->targetMb->tag_get_handle(remapTag1Name.c_str(), 1, MB_TYPE_DOUBLE, remapTag, MB_TAG_DENSE | MB_TAG_CREAT);
                    for (int i = 0; i < mbr->targetElements.size(); i++)
                    {
                        remapData[i] = mpas_approx->domain(i * locationTagLength + m, geomDim + l);
                    }
                    mbr->targetMb->tag_set_data(remapTag, mbr->targetElements, remapData.data());
                }
            }
        }

        // Move pointers around for visualizing in Paraview
        input = mpas_input;
        mpas_input = nullptr;
        approx = mpas_approx;
        mpas_approx = nullptr;
    }

    void print_source_modeling_errors()
    {
        if (errs != nullptr)
        {
            cerr << "Warning: Overwriting existing error field in CBlock::print_source_modeling_errors" << endl;
            delete errs;
        }
        errs = new mfa::PointSet<T>(input->params, input->model_dims());
        input->abs_diff(*approx, *errs);

        // Error summary
        mfa::Stats<T> stats(false); // false: don't log errors explicitly
        stats.init(input);

        for (int i = 0; i < input->npts; i++)
        {
            for (int l = 0; l < varNames.size(); l++)
            stats.update(l, errs->domain(i, geomDim + l));
        }

        fmt::print(stderr, "Source modeling error summary:\n");
        stats.set_style(mfa::PrintStyle::Side);
        for (int l = 0; l < varNames.size(); l++)
        {
            fmt::print(stderr, "Variable '{}':\n", varNames[l]);
            stats.print_var(l);
        }
    }

    void writeForwardRemapH5M(const string& filename)
    {
        if (!approx)
        {
            fmt::print(stderr, "ERROR: Cannot write forward profiles before remapping.\n");
            exit(1);
        }

        Tag locationTag;
        ErrorCode rval = mbr->targetMb->tag_get_handle("ROMS3DLocations", locationTag);
        if (rval != MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to get ROMS3DLocations tag.\n");
            exit(1);
        }

        int locationTagLength = 0;
        rval = mbr->targetMb->tag_get_length(locationTag, locationTagLength);
        const size_t expectedPoints = mbr->targetElements.size() * locationTagLength;
        if (rval != MB_SUCCESS || locationTagLength <= 0 ||
            static_cast<size_t>(approx->npts) != expectedPoints)
        {
            fmt::print(stderr, "ERROR: Cannot pack {} decoded points into {} target profiles of length {}.\n",
                       approx->npts, mbr->targetElements.size(), locationTagLength);
            exit(1);
        }

        for (int varIdx = 0; varIdx < varNames.size(); varIdx++)
        {
            const string tagName = varNames[varIdx] + "_forward_remap";
            const T* profileData = approx->domain.col(geomDim + varIdx).data();
            for (size_t pointIdx = 0; pointIdx < expectedPoints; pointIdx++)
            {
                const T value = profileData[pointIdx];
                if (!std::isfinite(value))
                {
                    fmt::print(stderr, "ERROR: Non-finite value in '{}' at decoded point {}.\n",
                               tagName, pointIdx);
                    exit(1);
                }
            }

            Tag profileTag;
            rval = mbr->targetMb->tag_get_handle(tagName.c_str(), locationTagLength,
                                                  MB_TYPE_DOUBLE, profileTag,
                                                  MB_TAG_DENSE | MB_TAG_CREAT);
            if (rval != MB_SUCCESS)
            {
                fmt::print(stderr, "ERROR: Failed to create profile tag '{}'.\n", tagName);
                exit(1);
            }
            rval = mbr->targetMb->tag_set_data(profileTag, mbr->targetElements,
                                                profileData);
            if (rval != MB_SUCCESS)
            {
                fmt::print(stderr, "ERROR: Failed to populate profile tag '{}'.\n", tagName);
                exit(1);
            }
        }

        rval = mbr->targetMb->write_mesh(filename.c_str(), &mbr->targetFileSet, 1);
        if (rval != MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to write forward remap file '{}'.\n", filename);
            exit(1);
        }
        fmt::print(stderr, "Wrote {} complete forward profiles to '{}'.\n",
                   varNames.size(), filename);
    }

    void writeRoundtripResults(const string& h5mFilename, const string& metricsFilename)
    {
        if (!roundtrip_approx || !input || sourcePointEligible.size() != static_cast<size_t>(input->npts))
        {
            fmt::print(stderr, "ERROR: Round-trip values and eligibility must be ready before writing results.\n");
            exit(1);
        }

        const T nan = numeric_limits<T>::quiet_NaN();
        Tag maxLevelCellTag;
        ErrorCode rval = mbr->sourceMb->tag_get_handle("maxLevelCell", maxLevelCellTag);
        int outputProfileLength = 0;
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->tag_get_length(maxLevelCellTag, outputProfileLength);
        Tag layerThicknessTag;
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->tag_get_handle("timeDaily_avg_layerThickness_3d", layerThicknessTag);
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->tag_get_length(layerThicknessTag, outputProfileLength);
        if (rval != MB_SUCCESS || outputProfileLength < maxLevel)
        {
            fmt::print(stderr, "ERROR: Invalid MPAS output profile length {}.\n", outputProfileLength);
            exit(1);
        }

        Tag coverageFractionTag;
        Tag coverageClassTag;
        rval = mbr->sourceMb->tag_get_handle("roundtrip_coverage_fraction", 1, MB_TYPE_DOUBLE,
                                             coverageFractionTag, MB_TAG_DENSE | MB_TAG_CREAT);
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->tag_get_handle("roundtrip_coverage_class", 1, MB_TYPE_INTEGER,
                                                 coverageClassTag, MB_TAG_DENSE | MB_TAG_CREAT);
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->tag_set_data(coverageFractionTag, selectedSourceElements,
                                               sourceCoverageFractions.data());
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->tag_set_data(coverageClassTag, selectedSourceElements,
                                               sourceCoverageClasses.data());
        if (rval != MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to write MPAS coverage tags.\n");
            exit(1);
        }

        Tag globalIdTag;
        vector<int> sourceGlobalIds(numSourceElements);
        rval = mbr->sourceMb->tag_get_handle("GLOBAL_ID", globalIdTag);
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->tag_get_data(globalIdTag, selectedSourceElements,
                                               sourceGlobalIds.data());
        if (rval != MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to read MPAS GLOBAL_ID values.\n");
            exit(1);
        }

        struct Metric
        {
            long count = 0;
            int maxGlobalId = -1;
            int maxLevel = -1;
            T sumError = 0;
            T sumSquaredError = 0;
            T maxAbsoluteError = 0;
            T originalMin = numeric_limits<T>::max();
            T originalMax = numeric_limits<T>::lowest();
            T roundtripMin = numeric_limits<T>::max();
            T roundtripMax = numeric_limits<T>::lowest();
        };
        vector<Metric> overall(varNames.size());
        vector<vector<Metric>> perLevel(varNames.size(), vector<Metric>(maxLevel));
        auto updateMetric = [](Metric& metric, T original, T roundtrip,
                               int globalId, int level)
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
        };

        vector<vector<T>> roundtripProfiles(varNames.size(),
                                             vector<T>(numSourceElements * outputProfileLength, nan));
        vector<vector<T>> signedErrorProfiles(varNames.size(),
                                              vector<T>(numSourceElements * outputProfileLength, nan));
        vector<vector<T>> absoluteErrorProfiles(varNames.size(),
                                                vector<T>(numSourceElements * outputProfileLength, nan));
        vector<int> eligibilityProfiles(numSourceElements * outputProfileLength, 0);
        vector<int> cellEligibility(numSourceElements, 0);

        int elementIdx = 0;
        for (EntityHandle element : selectedSourceElements)
        {
            for (int level = 0; level < sourceElementLevels[elementIdx]; level++)
            {
                const int pointIdx = sourceCellIDs[elementIdx][level];
                const int profileIdx = elementIdx * outputProfileLength + level;
                eligibilityProfiles[profileIdx] = sourcePointEligible[pointIdx];
                if (sourcePointEligible[pointIdx])
                    cellEligibility[elementIdx] = 1;
                for (int varIdx = 0; varIdx < varNames.size(); varIdx++)
                {
                    const T original = input->domain(pointIdx, geomDim + varIdx);
                    const T roundtrip = roundtrip_approx->domain(pointIdx, geomDim + varIdx);
                    roundtripProfiles[varIdx][profileIdx] = roundtrip;
                    if (sourcePointEligible[pointIdx])
                    {
                        const T error = roundtrip - original;
                        signedErrorProfiles[varIdx][profileIdx] = error;
                        absoluteErrorProfiles[varIdx][profileIdx] = std::abs(error);
                        updateMetric(overall[varIdx], original, roundtrip,
                                     sourceGlobalIds[elementIdx], level);
                        updateMetric(perLevel[varIdx][level], original, roundtrip,
                                     sourceGlobalIds[elementIdx], level);
                    }
                }
            }
            elementIdx++;
        }

        Tag eligibilityTag;
        Tag cellEligibilityTag;
        rval = mbr->sourceMb->tag_get_handle("roundtrip_error_eligible", outputProfileLength,
                                             MB_TYPE_INTEGER, eligibilityTag,
                                             MB_TAG_DENSE | MB_TAG_CREAT);
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->tag_get_handle("roundtrip_error_cell_eligible", 1,
                                                 MB_TYPE_INTEGER, cellEligibilityTag,
                                                 MB_TAG_DENSE | MB_TAG_CREAT);
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->tag_set_data(eligibilityTag, selectedSourceElements,
                                               eligibilityProfiles.data());
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->tag_set_data(cellEligibilityTag, selectedSourceElements,
                                               cellEligibility.data());
        if (rval != MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to write MPAS eligibility tags.\n");
            exit(1);
        }

        for (int varIdx = 0; varIdx < varNames.size(); varIdx++)
        {
            const vector<pair<string, vector<T>*>> outputs = {
                {varNames[varIdx] + "_roundtrip", &roundtripProfiles[varIdx]},
                {varNames[varIdx] + "_roundtrip_signed_error", &signedErrorProfiles[varIdx]},
                {varNames[varIdx] + "_roundtrip_absolute_error", &absoluteErrorProfiles[varIdx]}
            };
            for (const auto& output : outputs)
            {
                Tag tag;
                rval = mbr->sourceMb->tag_get_handle(output.first.c_str(), outputProfileLength,
                                                      MB_TYPE_DOUBLE, tag,
                                                      MB_TAG_DENSE | MB_TAG_CREAT);
                if (rval == MB_SUCCESS)
                    rval = mbr->sourceMb->tag_set_data(tag, selectedSourceElements,
                                                       output.second->data());
                if (rval != MB_SUCCESS)
                {
                    fmt::print(stderr, "ERROR: Failed to write MPAS tag '{}'.\n", output.first);
                    exit(1);
                }
            }
        }

        EntityHandle outputSet;
        rval = mbr->sourceMb->create_meshset(MESHSET_SET, outputSet);
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->add_entities(outputSet, selectedSourceElements);
        if (rval == MB_SUCCESS)
            rval = mbr->sourceMb->write_mesh(h5mFilename.c_str(), &outputSet, 1);
        if (rval != MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to write round-trip MPAS file '{}'.\n", h5mFilename);
            exit(1);
        }

        ofstream metrics(metricsFilename);
        if (!metrics)
        {
            fmt::print(stderr, "ERROR: Failed to open metrics file '{}'.\n", metricsFilename);
            exit(1);
        }
        metrics << "variable,level,count,bias,rmse,max_absolute_error,max_global_id,max_level,"
                   "original_min,original_max,roundtrip_min,roundtrip_max\n";
        metrics << setprecision(17);
        auto writeMetric = [&](const string& variable, const string& level, const Metric& metric)
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
        for (int varIdx = 0; varIdx < varNames.size(); varIdx++)
        {
            writeMetric(varNames[varIdx], "overall", overall[varIdx]);
            for (int level = 0; level < maxLevel; level++)
                writeMetric(varNames[varIdx], to_string(level), perLevel[varIdx][level]);

            const Metric& metric = overall[varIdx];
            if (metric.count > 0)
            {
                fmt::print(stderr,
                           "Round-trip '{}': count={} bias={} RMSE={} max_abs={} at GLOBAL_ID {} level {}\n",
                           varNames[varIdx], metric.count, metric.sumError / metric.count,
                           std::sqrt(metric.sumSquaredError / metric.count), metric.maxAbsoluteError,
                           metric.maxGlobalId, metric.maxLevel);
            }
        }
        fmt::print(stderr, "Wrote round-trip fields to '{}' and metrics to '{}'.\n",
                   h5mFilename, metricsFilename);
    }

    // Write .mb files as a VTK file for debugging
    void writeVTK()
    {
        fmt::print(stderr, "Writing remapped data to vtk...");
        mbr->targetMb->write_mesh("remap_out.vtk", &mbr->targetFileSet, 1);
        // mbr->sourceMb->write_mesh("source_out.vtk", &mbr->sourceFileSet, 1);

        EntityHandle sourceSubset;
        mbr->sourceMb->create_meshset(MESHSET_SET, sourceSubset);
        mbr->sourceMb->add_entities(sourceSubset, selectedSourceElements);

        const int nLevels = 15;
        const vector<string> sourceVarNames = {
            "timeDaily_avg_activeTracers_temperature_3d",
            "timeDaily_avg_activeTracers_salinity_3d"
        };
        for (const string& varName : sourceVarNames)
        {
            Tag sourceTag;
            ErrorCode rval = mbr->sourceMb->tag_get_handle(varName.c_str(), sourceTag);
            if (rval != MB_SUCCESS)
            {
                fmt::print(stderr, "ERROR: Failed to get source tag '{}'.\n", varName);
                exit(1);
            }

            int tagLength = 0;
            rval = mbr->sourceMb->tag_get_length(sourceTag, tagLength);
            if (rval != MB_SUCCESS || tagLength < nLevels)
            {
                fmt::print(stderr, "ERROR: Source tag '{}' has length {}; expected at least {}.\n",
                           varName, tagLength, nLevels);
                exit(1);
            }

            vector<T> sourceData(selectedSourceElements.size() * tagLength);
            rval = mbr->sourceMb->tag_get_data(sourceTag, selectedSourceElements, sourceData.data());
            if (rval != MB_SUCCESS)
            {
                fmt::print(stderr, "ERROR: Failed to read source tag '{}'.\n", varName);
                exit(1);
            }

            vector<T> levelData(selectedSourceElements.size());
            for (int level = 0; level < nLevels; level++)
            {
                for (size_t i = 0; i < selectedSourceElements.size(); i++)
                {
                    levelData[i] = sourceData[i * tagLength + level];
                }

                Tag levelTag;
                string levelTagName = varName + "_" + to_string(level);
                rval = mbr->sourceMb->tag_get_handle(levelTagName.c_str(), 1, MB_TYPE_DOUBLE,
                                                      levelTag, MB_TAG_DENSE | MB_TAG_CREAT);
                if (rval != MB_SUCCESS)
                {
                    fmt::print(stderr, "ERROR: Failed to create source tag '{}'.\n", levelTagName);
                    exit(1);
                }
                rval = mbr->sourceMb->tag_set_data(levelTag, selectedSourceElements, levelData.data());
                if (rval != MB_SUCCESS)
                {
                    fmt::print(stderr, "ERROR: Failed to set source tag '{}'.\n", levelTagName);
                    exit(1);
                }
            }
        }

        mbr->sourceMb->write_mesh("source_subset.vtk", &sourceSubset, 1);

        fmt::print(stderr, "done.\n");
    }

    // Set the vertical knot density from the spacing of representative MPAS levels.
    void set3DKnots()
    {
        if (sourceLevelMeanHeights.size() < 2)
        {
            fmt::print(stderr, "ERROR: Need at least two MPAS vertical levels to set adaptive knots.\n");
            exit(1);
        }

        const int verticalDim = 2;
        const int degree = mfa->var(0).p(verticalDim);
        const int nctrl = mfa->var(0).tmesh.tensor_prods[0].nctrl_pts(verticalDim);
        const int numInteriorKnots = nctrl - degree - 1;
        if (numInteriorKnots < 0)
        {
            fmt::print(stderr, "ERROR: Invalid vertical model: nctrl={} degree={}.\n", nctrl, degree);
            exit(1);
        }

        vector<T> levelParams;
        levelParams.reserve(sourceLevelMeanHeights.size());
        for (T height : sourceLevelMeanHeights)
        {
            T param = (height - minHeight) / (maxHeight - minHeight);
            levelParams.push_back(std::max(static_cast<T>(0), std::min(static_cast<T>(1), param)));
        }
        std::sort(levelParams.begin(), levelParams.end());

        vector<T> zknots(numInteriorKnots);
        for (int i = 0; i < numInteriorKnots; i++)
        {
            T position = static_cast<T>(i + 1) * (levelParams.size() - 1) /
                         static_cast<T>(numInteriorKnots + 1);
            size_t lower = static_cast<size_t>(std::floor(position));
            size_t upper = std::min(lower + 1, levelParams.size() - 1);
            T fraction = position - lower;
            zknots[i] = levelParams[lower] + fraction * (levelParams[upper] - levelParams[lower]);
        }

        vector<vector<T>> allknots(dom_dim);
        for (int dim = 0; dim < dom_dim; dim++)
        {
            allknots[dim] = mfa->var(0).tmesh.all_knots[dim];
        }
        if (!zknots.empty())
        {
            allknots[verticalDim] = mfa->pinKnots(zknots, degree);
        }

        fmt::print(stderr, "MPAS mean vertical level parameters: {}\n", mfa::print_vec(levelParams));
        fmt::print(stderr, "Adaptive vertical interior knots: {}\n", mfa::print_vec(zknots));
        fmt::print(stderr, "Vertical knot vector: {}\n", mfa::print_vec(allknots[verticalDim]));

        mfa->setKnots(allknots);
    }

    void remap(
        const diy::Master::ProxyWithLink&   cp,
        mfa::MFAInfo&   info)
    {
        setParameterizations();
        this->setup_MFA(cp, info);
        computeRemap(cp, info);
    }

    void setReverse3DKnots()
    {
        Tag locationTag;
        ErrorCode rval = mbr->targetMb->tag_get_handle("ROMS3DLocations", locationTag);
        int locationTagLength = 0;
        if (rval == MB_SUCCESS)
            rval = mbr->targetMb->tag_get_length(locationTag, locationTagLength);
        if (rval != MB_SUCCESS || locationTagLength < 2)
        {
            fmt::print(stderr, "ERROR: Need at least two ROMS levels to set reverse vertical knots.\n");
            exit(1);
        }

        const int numTargetElements = mbr->targetElements.size();
        if (targetHeights.size() != numTargetElements * locationTagLength)
        {
            fmt::print(stderr, "ERROR: ROMS height count does not match target profile layout.\n");
            exit(1);
        }

        vector<T> levelParams(locationTagLength, 0);
        for (int level = 0; level < locationTagLength; level++)
        {
            T sum = 0;
            for (int element = 0; element < numTargetElements; element++)
                sum += targetHeights(element * locationTagLength + level);
            const T meanHeight = sum / numTargetElements;
            const T param = (meanHeight - minHeight) / (maxHeight - minHeight);
            levelParams[level] = std::max(static_cast<T>(0), std::min(static_cast<T>(1), param));
        }
        std::sort(levelParams.begin(), levelParams.end());

        const int verticalDim = 2;
        const int degree = reverse_mfa->var(0).p(verticalDim);
        const int nctrl = reverse_mfa->var(0).tmesh.tensor_prods[0].nctrl_pts(verticalDim);
        const int numInteriorKnots = nctrl - degree - 1;
        if (numInteriorKnots < 0)
        {
            fmt::print(stderr, "ERROR: Invalid reverse vertical model: nctrl={} degree={}.\n",
                       nctrl, degree);
            exit(1);
        }

        vector<T> zknots(numInteriorKnots);
        for (int i = 0; i < numInteriorKnots; i++)
        {
            const T position = static_cast<T>(i + 1) * (levelParams.size() - 1) /
                               static_cast<T>(numInteriorKnots + 1);
            const size_t lower = static_cast<size_t>(std::floor(position));
            const size_t upper = std::min(lower + 1, levelParams.size() - 1);
            const T fraction = position - lower;
            zknots[i] = levelParams[lower] + fraction * (levelParams[upper] - levelParams[lower]);
        }

        vector<vector<T>> allknots(dom_dim);
        for (int dim = 0; dim < dom_dim; dim++)
            allknots[dim] = reverse_mfa->var(0).tmesh.all_knots[dim];
        if (!zknots.empty())
            allknots[verticalDim] = reverse_mfa->pinKnots(zknots, degree);

        fmt::print(stderr, "Reverse ROMS vertical interior knots: {}\n", mfa::print_vec(zknots));
        reverse_mfa->setKnots(allknots);
    }

    void reverseRemap(mfa::MFAInfo info)
    {
        if (!input || !approx)
        {
            fmt::print(stderr, "ERROR: Forward remap must complete before reverse remapping.\n");
            exit(1);
        }

        delete reverse_mfa;
        reverse_mfa = new mfa::MFA<T>(info);
        setReverse3DKnots();

        const double encodeStart = MPI_Wtime();
        reverse_mfa->FixedEncode(*approx, info.regularization, info.reg1and2, info.weighted);
        fmt::print(stderr, "Reverse MFA encode time: {:.3f} s\n", MPI_Wtime() - encodeStart);

        delete roundtrip_approx;
        roundtrip_approx = new mfa::PointSet<T>(input->params, input->model_dims());
        roundtrip_approx->domain.leftCols(geomDim) = input->domain.leftCols(geomDim);
        const double decodeStart = MPI_Wtime();
        for (int varIdx = 0; varIdx < reverse_mfa->nvars(); varIdx++)
            reverse_mfa->DecodeVar(varIdx, *roundtrip_approx);
        fmt::print(stderr, "Reverse MFA science decode time: {:.3f} s\n", MPI_Wtime() - decodeStart);

        if (roundtrip_approx->npts != input->npts)
        {
            fmt::print(stderr, "ERROR: Reverse decode produced {} points; expected {} MPAS points.\n",
                       roundtrip_approx->npts, input->npts);
            exit(1);
        }
        for (int varIdx = 0; varIdx < varNames.size(); varIdx++)
        {
            for (int pointIdx = 0; pointIdx < roundtrip_approx->npts; pointIdx++)
            {
                if (!std::isfinite(roundtrip_approx->domain(pointIdx, geomDim + varIdx)))
                {
                    fmt::print(stderr, "ERROR: Reverse decode produced a non-finite value for '{}' at point {}.\n",
                               varNames[varIdx], pointIdx);
                    exit(1);
                }
            }
        }
        fmt::print(stderr, "Reverse decode produced {} MPAS active-level points.\n",
                   roundtrip_approx->npts);
    }

    void activateReverseMFAForSerialization()
    {
        if (!reverse_mfa)
        {
            fmt::print(stderr, "ERROR: Reverse MFA is not available for serialization.\n");
            exit(1);
        }
        std::swap(mfa, reverse_mfa);
        std::swap(input, approx);
        std::swap(approx, roundtrip_approx);
    }


    void print_knots_ctrl(const mfa::MFA_Data<T>& model) const
    {
        VectorXi tot_nctrl_pts_dim = VectorXi::Zero(model.dom_dim);        // number contrl points per dim.
        size_t tot_nctrl_pts = 0;                                        // total number of control points

        for (auto j = 0; j < model.ntensors(); j++)
        {
            tot_nctrl_pts_dim += model.tmesh.tensor_prods[j].nctrl_pts;
            tot_nctrl_pts += model.tmesh.tensor_prods[j].nctrl_pts.prod();
        }
        // print number of control points per dimension only if there is one tensor
        if (model.ntensors() == 1)
            fmt::print(stderr, "# output ctrl pts     = [ {} ]\n", tot_nctrl_pts_dim.transpose());
        fmt::print(stderr, "tot # output ctrl pts = {}\n", tot_nctrl_pts);

        fmt::print(stderr, "# output knots        = [ ");
        for (auto j = 0 ; j < model.tmesh.all_knots.size(); j++)
        {
            fmt::print(stderr, "{} ", model.tmesh.all_knots[j].size());
        }
        fmt::print(stderr, "]\n");
    }

    void print_model(const diy::Master::ProxyWithLink& cp)    // error was computed
    {
        if (!mfa)
        {
            fmt::print(stderr, "gid = {}: No MFA found.\n", cp.gid());
            return;
        }

        fmt::print(stderr, "gid = {}\n", cp.gid());

        // geometry
        fmt::print(stderr, "---------------- geometry model ----------------\n");
        print_knots_ctrl(mfa->geom());
        fmt::print(stderr, "------------------------------------------------\n");

        // science variables
        fmt::print(stderr, "\n----------- science variable models ------------\n");
        for (int i = 0; i < mfa->nvars(); i++)
        {
            fmt::print(stderr, "-------------------- var {} --------------------\n", i);
            print_knots_ctrl(mfa->var(i));
            fmt::print(stderr, "------------------------------------------------\n");
        }
    }
};

#endif // _MFA_CBLOCK
