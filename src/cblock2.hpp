//--------------------------------------------------------------
// A custom block for climate remapping
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_CBLOCK
#define _MFA_CBLOCK

#include    <random>
#include    <stdio.h>
#include    <cmath>
#include    <limits>
#include    <unordered_map>
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
        fmt::print(stderr, "MBReader: Starting to load source mesh file {}\n", filename_);
        rval = sourceMb->load_file(sourceFilename.c_str(), &sourceFileSet); MB_CHK_ERR_RET(rval);
        fmt::print(stderr, "MBReader: Finished loading source mesh file {}\n", filename_);

        // Get vertices
        fmt::print(stderr, "MBReader: Getting vertices for source mesh file {}\n", filename_);
        rval = sourceMb->get_entities_by_type(sourceFileSet, MBVERTEX, sourceVertices); MB_CHK_ERR_RET(rval);
        fmt::print(stderr, "MBReader: Finished getting vertices for source mesh file {}\n", filename_);

        // Get vertex coordinates
        sourceCoords.resize(sourceVertices.size()*3);
        rval = sourceMb->get_coords(sourceVertices, sourceCoords.data()); MB_CHK_ERR_RET(rval);

        // Get mesh element rage
        rval = sourceMb->get_entities_by_dimension(sourceFileSet, cellDim, sourceElements); MB_CHK_ERR_RET(rval);

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

        // Get mesh element rage
        rval = targetMb->get_entities_by_dimension(targetFileSet, cellDim, targetElements); MB_CHK_ERR_RET(rval);

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

    mfa::PointSet<T>*   mpas_input;
    mfa::PointSet<T>*   mpas_approx;
    mfa::PointSet<T>*   mpas_error;
    mfa::PointSet<T>*   roms_input;

    MBReader<T>* mbr;

    mfa::Bbox<T> sourceBox;
    mfa::Bbox<T> targetBox;
    mfa::Bbox<T> bbox;

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
        roms_input(nullptr)
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

    void computeBbox()
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
        if (!bbox.doesContain(*mpas_input, 2))
        {
            fmt::print(stderr, "ERROR: Bounding box does not contain all of source data\n");
            exit(1);
        }
        if (!bbox.doesContain(*roms_input, 2))
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
            const mfa::Bbox<T>&  sourceReadBbox) const
    {
        const EntityHandle* conn = nullptr;
        int nconn = 0;
        ErrorCode rval = mbr->sourceMb->get_connectivity(element, conn, nconn);
        if (rval != MB_SUCCESS || !conn || nconn == 0)
        {
            fmt::print(stderr, "ERROR: Failed to get element connectivity in elementIntersectsSourceReadBbox().\n");
            exit(1);
        }

        vector<T> coords(geomDim * nconn);
        rval = mbr->sourceMb->get_coords(conn, nconn, coords.data());
        if (rval != MB_SUCCESS)
        {
            fmt::print(stderr, "ERROR: Failed to get element coordinates in elementIntersectsSourceReadBbox().\n");
            exit(1);
        }

        VectorX<T> elemMins = VectorX<T>::Constant(geomDim, numeric_limits<T>::max());
        VectorX<T> elemMaxs = VectorX<T>::Constant(geomDim, numeric_limits<T>::lowest());
        VectorX<T> xyz(geomDim);
        VectorX<T> rot;

        for (int i = 0; i < nconn; i++)
        {
            for (int j = 0; j < geomDim; j++)
            {
                xyz(j) = coords[geomDim * i + j];
            }

            sourceReadBbox.toRotatedSpace(xyz, rot);
            elemMins = elemMins.cwiseMin(rot);
            elemMaxs = elemMaxs.cwiseMax(rot);
        }

        for (int j = 0; j < geomDim; j++)
        {
            if (elemMaxs(j) < sourceReadBbox.rotatedMins(j) ||
                elemMins(j) > sourceReadBbox.rotatedMaxs(j))
            {
                return false;
            }
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
            rval = mbr->sourceMb->get_adjacencies(&vertex, 1, mbr->cellDim, false, vertexAdjacentCells);
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
    void selectSourceElements(const mfa::Bbox<T>& sourceReadBbox)
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
            if (elementIntersectsSourceReadBbox(sourceElementHandles[i], sourceReadBbox))
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
        int haloCount = 0;
        for (int i = 0; i < totalElements; i++)
        {
            if (selected[i])
            {
                selectedSourceElements.insert(sourceElementHandles[i]);
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
                    string filename,
            const   mfa::Bbox<T>& sourceReadBbox)
    {
        // if (!roms_input)
        // {
        //     fmt::print(stderr, "ERROR: Cannot read source data before target data when remapping.\n");
        //     fmt::print(stderr, "       Source data model needs to know the bounding box of the target data ahead of time.\n");
        //     fmt::print(stderr, "       Exiting.\n");
        //     exit(1);
        // }

        mbr->loadSourceMesh(filename);
        // mbr->loadSourceBdry();
        mbr->sourceVertexBounds(sourceMins, sourceMaxs);
        if (verbose >= 1)
        {
            mfa::print_bbox(sourceMins, sourceMaxs, "Source Bounding");
            sourceReadBbox.print("Source Read Bbox");
        }

        // Fresh source metadata for each load.
        numSourceCells = 0;
        numSourceElements = 0;
        maxLevel = 0;
        sourceCellIDs.clear();
        sourceElementLevels.clear();
        selectedSourceElements.clear();

        // Select the source chunk to read.
        selectSourceElements(sourceReadBbox);

        // Compute metadata about extruded 3d mesh
        computeSourceLevelInfo();
        computeSourceCellInfo();

        // Set up MFA PointSet
        VectorXi modelDims = VectorXi::Ones(varNames.size() + 1); // one model per remapped variable, plus geometry
        modelDims(0) = geomDim;

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

        // Build one 3D point per active extruded cell using axial heights:
        // top = ssh - cumulative_thickness; bottom = top - layerThickness.
        int elementID = 0;
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

                // TODO: adjust top layer
                T layerThickness = layerThicknessData[elementID * layerTagLength + level];
                T top = sshData[elementID] - cumulativeThickness;
                T bottom = top - layerThickness;
                T axialCentroid = static_cast<T>(0.5) * (top + bottom);

                for (int k = 0; k < geomDim; k++)
                {
                    mpas_input->domain(cellID, k) = static_cast<T>(centroid[k]) + axialCentroid * normal(k);
                }

                cumulativeThickness += layerThickness;
            }
        }

        // Add data values to PointSet
        for (int l = 0; l < varNames.size(); l++)
        {
            addSourceVariable<V>(cp, filename, varNames[l], l);
        }
    }

    template <typename V>
    void readTargetData(
            const   diy::Master::ProxyWithLink& cp,
                    string  filename)
    {
        const bool decodeAtVertices = false;

        // Dimensionality of each variable (assumed to be scalar)
        VectorXi modelDims = VectorXi::Ones(varNames.size() + 1); // one model per remapped variable, plus geometry
        modelDims(0) = geomDim;

        mbr->loadTargetMesh(filename);
        mbr->targetVertexBounds(targetMins, targetMaxs);
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

        roms_input = new mfa::PointSet<T>(dom_dim, modelDims, npts);
        for ( ; it != end; ++it, ++i)
        {
            mbr->targetMb->get_coords(&(*it), 1, coord.data());
            for (int j = 0; j < geomDim; j++)
            {
                roms_input->domain(i, j) = coord[j];
            }
        }
    }

    void setParameterizations()
    {
        computeBbox();
        roms_input->set_domain_params(bbox);
        mpas_input->set_domain_params(bbox);
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

        // Create MFA model from source data
        for (int l = 0; l < varNames.size(); l++)
        {
            mfa->FixedEncodeVar(l, *mpas_input, info.regularization, info.reg1and2, info.weighted);
        }
        
        // Evaluate MFA at target mesh locations
        for (int l = 0; l < varNames.size(); l++)
        {
            mfa->DecodeVar(l, *mpas_approx);
        }

        // Add remapped values as new tags
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

        // Move pointers around for visualizing in Paraview
        input = mpas_input;
        mpas_input = nullptr;
        approx = mpas_approx;
        mpas_approx = nullptr;
    }

    // Write .mb files as a VTK file for debugging
    void writeVTK()
    {
        fmt::print(stderr, "Writing remapped data to vtk...");
        mbr->targetMb->write_mesh("remap_out.vtk", &mbr->targetFileSet, 1);
        mbr->sourceMb->write_mesh("source_out.vtk", &mbr->sourceFileSet, 1);
        fmt::print(stderr, "done.\n");
    }

    // Set custom knots based on depth levels in 3D
    void set3DKnots()
    {
        // All depth levels
        vector<T> depth1 = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170.197, 180.761, 191.821, 203.499, 215.923, 229.233, 243.584, 259.156, 276.152, 294.815, 315.424, 338.312, 363.875, 392.58, 424.989, 461.767, 503.707, 551.749, 606.997, 670.729, 744.398, 829.607, 928.043, 1041.37, 1171.04, 1318.09, 1482.9, 1664.99, 1863.01, 2074.87, 2298.04, 2529.9, 2768.1, 3010.67, 3256.14};

        // Removes some depth levels near ocean floor
        vector<T> depth2 = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170.197, 180.761, 191.821, 203.499, 215.923, 229.233, 243.584, 259.156, 276.152, 294.815, 315.424, 338.312, 363.875, 392.58, 424.989, 461.767, 503.707, 551.749, 606.997, 670.729, 744.398, 829.607, 928.043, 1171.04, 1482.9, 1863.01, 2298.04, 2768.1, 3256.14};

        vector<T> refBottomDepth = depth2;

        auto dom_mins = mpas_input->mins();
        auto dom_maxs = mpas_input->maxs();
        vector<T> zknots(refBottomDepth.size());
        for (int i = 0; i < refBottomDepth.size(); i++)
        {
            T zk = (-1*refBottomDepth[i] - dom_mins(2)) / (dom_maxs(2) - dom_mins(2));
            zknots[refBottomDepth.size() - 1 - i] = zk;
        }

        vector<vector<T>> allknots(3);
        allknots[0] = mfa->var(0).tmesh.all_knots[0];   // leave unchanged
        allknots[1] = mfa->var(1).tmesh.all_knots[1];   // leave unchanged
        allknots[2] = mfa->pinKnots(zknots, mfa->var(0).p(0));

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
