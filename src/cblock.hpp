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

    Interface* mb;
    ParallelComm* pc;
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
        mb = new Core;
        pc = new ParallelComm(mb, local);
    }

    ~MBReader()
    {
        delete mb;
        delete pc;
    }

    void loadSourceMesh(string filename_)
    {
        cout << "MBReader: Reading remap source file " << filename_ << endl;
        sourceFilename = filename_;

        mb->create_meshset(MESHSET_SET, sourceFileSet);
        rval = mb->load_file(sourceFilename.c_str(), &sourceFileSet); MB_CHK_ERR_RET(rval);

        // Get vertices
        rval = mb->get_entities_by_type(sourceFileSet, MBVERTEX, sourceVertices); MB_CHK_ERR_RET(rval);

        // Get vertex coordinates
        sourceCoords.resize(sourceVertices.size()*3);
        rval = mb->get_coords(sourceVertices, sourceCoords.data()); MB_CHK_ERR_RET(rval);

        // Get mesh element rage
        rval = mb->get_entities_by_dimension(sourceFileSet, cellDim, sourceElements); MB_CHK_ERR_RET(rval);

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
        cout << "MBReader: Reading remap target file " << filename_ << endl;
        targetFilename = filename_;

        //
        mb->create_meshset(MESHSET_SET, targetFileSet);
        rval = mb->load_file(targetFilename.c_str(), &targetFileSet); MB_CHK_ERR_RET(rval);

        // Get vertices
        rval = mb->get_entities_by_type(targetFileSet, MBVERTEX, targetVertices); MB_CHK_ERR_RET(rval);

        // Get vertex coordinates
        targetCoords.resize(targetVertices.size()*3);
        rval = mb->get_coords(targetVertices, targetCoords.data()); MB_CHK_ERR_RET(rval);

        // Get mesh element rage
        rval = mb->get_entities_by_dimension(targetFileSet, cellDim, targetElements); MB_CHK_ERR_RET(rval);

        // Create map of element handles to range indices
        int idx = 0;
        for (auto eIt = targetElements.begin(), end = targetElements.end(); eIt != end; ++eIt)
        {
            targetEltIdxMap[*eIt] = idx;
            idx++;
        }
    }

    // NOTE: In 3D, this only selects polygonal boundary faces, since this is the structure of the expected
    //       MPAS data. However, 3D mesh boundary could be MBPOLYGON, MBQUAD, or MBTRI.
    void loadSourceBdry()
    {
        if (!sourceFileSet)
        {
            cerr << "MBReader Error: Cannot load source boundary, source file has not been read.\nExiting." << endl;
            exit(1);
        }
        
        Range faces;
        vector<EntityHandle> neighbors;

        // Get set of all faces. For a 2D mesh, the bdry is edges. In 3D it could be several cell types.
        // Right now we only expect polygons but this should be made more generic.
        if (cellDim == 2)
        {
            rval = mb->get_entities_by_type(sourceFileSet, MBEDGE, faces); MB_CHK_ERR_RET(rval);
        }
        else if (cellDim == 3)
        {
            rval = mb->get_entities_by_type(sourceFileSet, MBPOLYGON, faces); MB_CHK_ERR_RET(rval);
        }
        else
        {
            cerr << "ERROR: Unsupported cellDim in MBReader::loadSourceBdry()\nExiting" << endl;
            exit(1);
        }

        // Search through faces and only keep those with one parent cell
        for (auto it = faces.begin(), end = faces.end(); it != end; ++it)
        {
            neighbors.clear();
            mb->get_adjacencies( &(*it) , 1, 3, false, neighbors);
            if (neighbors.size() == 1)
            {
                sourceBdryFaces.push_back(*it);
            }
            else if (neighbors.size() == 0)
            {
                cerr << "ERROR: Found face with no parent cells" << endl;
                exit(1);
            }
        }
    }

    void sourceVertexBounds(VectorX<V>& mins, VectorX<V>& maxs)
    {
        if (sourceVertices.size() == 0)
        {
            cerr << "Warning: Attempted to compute sourceVertexBounds but no vertices found." << endl;
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
            cerr << "Warning: Attempted to compute targetVertexBounds but no vertices found." << endl;
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
struct CBlock : public BlockBase<T>
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

    mfa::PointSet<T>*   mpas_input;
    mfa::PointSet<T>*   mpas_approx;
    mfa::PointSet<T>*   mpas_error;
    mfa::PointSet<T>*   roms_input;

    MBReader<T>* mbr;

    VectorX<T> sourceMins;
    VectorX<T> sourceMaxs;
    VectorX<T> targetMins;
    VectorX<T> targetMaxs;
    VectorX<T> bboxMins;
    VectorX<T> bboxMaxs;

    // zero-initialize pointers during default construction
    CBlock() : 
        Base(),
        geomDim(3),
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

    virtual ~CBlock()
    {
        delete mpas_input;
        delete mpas_approx;
        delete mpas_error;
        delete roms_input;
        delete mbr;
    }

    static
        void* create()              { return mfa::create<CBlock>(); }

    static
        void destroy(void* b)       { mfa::destroy<CBlock>(b); }

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
        mfa::add<CBlock, T>(gid, core, bounds, domain, link, master, dom_dim, pt_dim, ghost_factor);
    }

    static
        void save(const void* b_, diy::BinaryBuffer& bb)    { mfa::save<CBlock, T>(b_, bb); }
    static
        void load(void* b_, diy::BinaryBuffer& bb)          { mfa::load<CBlock, T>(b_, bb); }

    void initMOAB(MPI_Comm local, int dim)
    {
        mbr = new MBReader<T>(local, dim);
    }

    void computeBbox()
    {
        bboxMins = VectorX<T>::Zero(geomDim);
        bboxMaxs = VectorX<T>::Zero(geomDim);
        for (int i = 0; i < geomDim; i++)
        {
            bboxMins(i) = std::min(sourceMins(i), targetMins(i));
            bboxMaxs(i) = std::max(sourceMaxs(i), targetMaxs(i));
        }

        mfa::print_bbox(bboxMins, bboxMaxs, "Full Bounding Box");
    }

    template<typename V>
    void add_source_data(
        const   diy::Master::ProxyWithLink& cp,
                string filename,
                string varName,
                int    varIdx,
                bool   addBdryData = true)
    {
        Tag dataTag;
        mbr->mb->tag_get_handle(varName.c_str(), dataTag);
        vector<T> dataVec(mbr->sourceElements.size());
        mbr->mb->tag_get_data(dataTag, mbr->sourceElements, dataVec.data());

        for (int i = 0; i < dataVec.size(); i++)
        {
            mpas_input->domain(i, geomDim + varIdx) = dataVec[i];
        }

        // Optionally add data for each boundary face
        if (addBdryData)
        {
            vector<EntityHandle> parent;
            int offset = mbr->sourceElements.size();
            for (int j = 0; j < mbr->sourceBdryFaces.size(); j++)
            {
                // Get tag data associated to parent cell
                parent.clear();
                mbr->mb->get_adjacencies(&mbr->sourceBdryFaces[j], 1, 3, false, parent);
                if (parent.size() != 1)
                {
                    cerr << "ERROR: Expected a single parent element" << endl;
                    exit(1);
                }

                int parentIdx = mbr->sourceEltIdxMap[parent[0]];
                mpas_input->domain(offset + j, geomDim + varIdx) = dataVec[parentIdx];
            }
        }
    }

    template <typename V>
    void read_mpas_data(
            const   diy::Master::ProxyWithLink& cp,
                    string filename,
                    mfa::MFAInfo& mfa_info,
                    bool addBdryData = true)
    {
        mbr->loadSourceMesh(filename);
        mbr->loadSourceBdry();
        mbr->sourceVertexBounds(sourceMins, sourceMaxs);

        // Set up MFA PointSet
        VectorXi model_dims(3);
        model_dims << geomDim, 1, 1;          // geometry, bathymetry, salinity, temperature
        mpas_input = new mfa::PointSet<T>(dom_dim, model_dims, mbr->sourceElements.size() + mbr->sourceBdryFaces.size());

        // Add cell centroid coordinates to PointSet
        int i = 0;
        vector<V> centroid(geomDim);
        for (auto it = mbr->sourceElements.begin(), end = mbr->sourceElements.end(); it != end; ++it, ++i)
        {
            mbr->mb->get_coords(&(*it), 1, centroid.data());
            for (int k = 0; k < geomDim; k++)
            {
                mpas_input->domain(i, k) = centroid[k];
            }          
        }

        // Add boundary data
        if (addBdryData)
        {
            vector<EntityHandle> parent;
            int offset = mbr->sourceElements.size();
            for (int j = 0; j < mbr->sourceBdryFaces.size(); j++)
            {
                mbr->mb->get_coords(&mbr->sourceBdryFaces[j], 1, centroid.data());
                for (int k = 0; k < geomDim; k++)
                {
                    mpas_input->domain(offset + j, k) = centroid[k];
                }
            }
        }

        // Add data values to PointSet
        add_source_data<V>(cp, filename, "Salinity3d", 0, addBdryData);
        add_source_data<V>(cp, filename, "Temperature3d", 1, addBdryData);

        // Compute total bounding box around mpas and roms
        computeBbox();

        // Set parametrization
        mpas_input->set_domain_params(bboxMins, bboxMaxs);

        // Set up MFA from user-specified options
        this->setup_MFA(cp, mfa_info);

        // Find block bounds for coordinates and values
        bounds_mins = mpas_input->domain.colwise().minCoeff();
        bounds_maxs = mpas_input->domain.colwise().maxCoeff();
        core_mins   = bounds_mins.head(dom_dim);
        core_maxs   = bounds_maxs.head(dom_dim);

        // debug
        mfa::print_bbox(bboxMins, bboxMaxs, "MPAS Custom");
        mfa::print_bbox(bounds_mins, bounds_maxs, "MPAS Bounds");
    }

    template <typename V>
    void read_roms_data(
            const   diy::Master::ProxyWithLink& cp,
                    string  filename)
    {
        const bool decodeAtVertices = false;

        VectorXi model_dims(1);
        model_dims(0) = geomDim;      // geometric coordinates only

        mbr->loadTargetMesh(filename);
        mbr->targetVertexBounds(targetMins, targetMaxs);

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

        roms_input = new mfa::PointSet<T>(dom_dim, model_dims, npts);
        for ( ; it != end; ++it, ++i)
        {
            mbr->mb->get_coords(&(*it), 1, coord.data());
            for (int j = 0; j < geomDim; j++)
            {
                roms_input->domain(i, j) = coord[j];
            }
        }

        // Compute input parametrization
        roms_input->set_domain_params();

        // Find block bounds for coordinates and values
        bounds_mins = roms_input->domain.colwise().minCoeff();
        bounds_maxs = roms_input->domain.colwise().maxCoeff();
        core_mins   = bounds_mins.head(dom_dim);
        core_maxs   = bounds_maxs.head(dom_dim);

        // debug
        mfa::print_bbox(core_mins, core_maxs, "ROMS Core");
        mfa::print_bbox(bounds_mins, bounds_maxs, "ROMS Bounds");
    }

    void convertParams( const mfa::PointSet<T>*  source,
                        const mfa::PointSet<T>*  target,
                              mfa::PointSet<T>*& convert)
    {
        VectorX<T> tMins = target->mins();
        VectorX<T> tMaxs = target->maxs();
        VectorX<T> sMins = source->mins();
        VectorX<T> sMaxs = source->maxs();
        VectorX<T> sDiff = sMaxs - sMins;

        // Compute parametrization of target points w.r.t. source domain
        shared_ptr<mfa::Param<T>> new_param = make_shared<mfa::Param<T>>(dom_dim);
        new_param->param_list.resize(target->npts, dom_dim);
        for (size_t k = 0; k < dom_dim; k++)
        {
            new_param->param_list.col(k) = (target->domain.col(k).array() - sMins(k)) * (1/sDiff(k));
        }

        // Set parametrization object for the PointSet we will decode into
        if (convert)
        {
            cerr << "Overwriting existing pointset in convertParams()" << endl;
            delete convert;
            convert = nullptr;
        }
        convert = new mfa::PointSet<T>(new_param, source->model_dims());
    }

    void remap(
        const diy::Master::ProxyWithLink&   cp,
        mfa::MFAInfo&   info,
        bool            dumpMatrices = false)
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
        mfa->FixedEncode(*mpas_input, info.regularization, info.reg1and2, info.weighted);

        // Compute parametrization of roms points w.r.t mpas domain
        convertParams(mpas_input, roms_input, mpas_approx);
        // VectorX<T> roms_mins = roms_input->mins();
        // VectorX<T> roms_maxs = roms_input->maxs();
        // VectorX<T> mpas_mins = mpas_input->mins();
        // VectorX<T> mpas_maxs = mpas_input->maxs();
        // VectorX<T> mpas_diff = mpas_maxs - mpas_mins;

        // // Compute parametrization of ROMS points in terms of MPAS domain
        // shared_ptr<mfa::Param<T>> new_param = make_shared<mfa::Param<T>>(dom_dim);
        // new_param->param_list.resize(roms_input->npts, dom_dim);
        // for (size_t k = 0; k < dom_dim; k++)
        // {
        //     new_param->param_list.col(k) = (roms_input->domain.col(k).array() - mpas_mins(k)) * (1/mpas_diff(k));
        // }

        // // Set parametrization object for the PointSet we will decode into
        // mpas_approx = new mfa::PointSet<T>(new_param, mpas_input->model_dims());

        // Write out collocation matrices for encoding and decoding
        if (dumpMatrices)
        {
            mfa->dumpCollocationMatrixEncode(0, mpas_input);
            mfa->dumpCollocationMatrixDecode(0, mpas_approx);
        }

        // Evaluate MFA
        mfa->Decode(*mpas_approx, false);

        Tag sal_remap_tag;
        Tag temp_remap_tag;
        mbr->mb->tag_get_handle("salinity_remap", 1, MB_TYPE_DOUBLE, sal_remap_tag, MB_TAG_DENSE | MB_TAG_CREAT);
        mbr->mb->tag_get_handle("temperature_remap", 1, MB_TYPE_DOUBLE, temp_remap_tag, MB_TAG_DENSE | MB_TAG_CREAT);
        vector<T> sal_remap_data(mpas_approx->npts);
        vector<T> temp_remap_data(mpas_approx->npts);
        for (int i = 0; i < mpas_approx->npts; i++)
        {
            sal_remap_data[i] = mpas_approx->domain(i, 3);
            temp_remap_data[i] = mpas_approx->domain(i, 4);
        }
        mbr->mb->tag_set_data(sal_remap_tag, mbr->targetElements, sal_remap_data.data());
        mbr->mb->tag_set_data(temp_remap_tag, mbr->targetElements, temp_remap_data.data());

        mbr->mb->write_mesh("remap_out.vtk", &mbr->targetFileSet, 1);

        // Move pointers around for visualizing in Paraview
        input = mpas_input;
        mpas_input = nullptr;
        approx = mpas_approx;
        mpas_approx = nullptr;
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
            cerr << "# output ctrl pts     = [ " << tot_nctrl_pts_dim.transpose() << " ]" << endl;
        cerr << "tot # output ctrl pts = " << tot_nctrl_pts << endl;

        cerr << "# output knots        = [ ";
        for (auto j = 0 ; j < model.tmesh.all_knots.size(); j++)
        {
            cerr << model.tmesh.all_knots[j].size() << " ";
        }
        cerr << "]" << endl;
    }

    void print_model(const diy::Master::ProxyWithLink& cp)    // error was computed
    {
        if (!mfa)
        {
            fmt::print("gid = {}: No MFA found.\n", cp.gid());
            return;
        }

        fmt::print("gid = {}\n", cp.gid());

        // geometry
        fmt::print("---------------- geometry model ----------------\n");
        print_knots_ctrl(mfa->geom());
        fmt::print("------------------------------------------------\n");

        // science variables
        fmt::print("\n----------- science variable models ------------\n");
        for (int i = 0; i < mfa->nvars(); i++)
        {
            fmt::print("-------------------- var {} --------------------\n", i);
            print_knots_ctrl(mfa->var(i));
            fmt::print("------------------------------------------------\n");
            // ray_stats.print_var(i);
            // fmt::print("------------------------------------------------\n");
        }
        
        // ray_stats.print_max();
        // fmt::print("------------------------------------------------\n");
        // fmt::print("# input points        = {}\n", input->npts);
        // fmt::print("compression ratio     = {:.2f}\n", this->compute_compression());
    }
};

#endif // _MFA_CBLOCK