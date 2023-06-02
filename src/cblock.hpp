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
#include    <mfa/types.hpp>
#include    <mfa/mfa.hpp>
#include    <mfa/block_base.hpp>
#include    <diy/master.hpp>
#include    <diy/reduce-operations.hpp>
#include    <diy/decomposition.hpp>
#include    <diy/assigner.hpp>
#include    <diy/io/block.hpp>
#include    <diy/io/bov.hpp>
#include    <diy/pick.hpp>
#include    <Eigen/Dense>
#include    <highfive/H5DataSet.hpp>
#include    <highfive/H5DataSpace.hpp>
#include    <highfive/H5File.hpp>

#include    "domain_args.hpp"

using namespace std;

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

    mfa::PointSet<T>*   mpas_input;
    mfa::PointSet<T>*   mpas_approx;
    mfa::PointSet<T>*   mpas_error;
    mfa::PointSet<T>*   roms_input;

    // zero-initialize pointers during default construction
    CBlock() : 
        Base(),
        mpas_input(nullptr),
        mpas_approx(nullptr),
        mpas_error(nullptr),
        roms_input(nullptr) { }

    virtual ~CBlock()
    {
        delete mpas_input;
        delete mpas_approx;
        delete mpas_error;
        delete roms_input;
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


    // TODO: error handling
    template <typename V>
    void read_hdf5_dataset_1d(  string filename,
                                string dsetname,
                                vector<V>& buffer,
                                bool verbose = false)
    {
        using namespace HighFive;
        try
        {
            File datafile(filename.c_str(), File::ReadWrite);
            DataSet dataset = datafile.getDataSet(dsetname.c_str());

            vector<size_t> dims = dataset.getDimensions();
            if (dims.size() != 1)
            {
                fmt::print("ERROR: Dimension mismatch in read_hdf5_dataset_1d. Got: {}. Expected: {}.\n", dims.size(), 1);
                fmt::print("       filename: {}\n", filename);
                fmt::print("       dataset:  {}\n", dsetname);
                fmt::print("Exiting.\n");
                exit(1);
            }
            dataset.read(buffer);

            if (verbose)
            {
                fmt::print("Read dataset {} from file {}\n", dsetname, filename);
                fmt::print("  Data size: {}\n", dims[0]);
            }
        }
        catch (Exception& err)
        {
            // catch and print any HDF5 error
            std::cerr << err.what() << std::endl;
        }

        return;
    }

    // TODO: error handling
    template <typename V>
    void read_hdf5_dataset_2d(  string filename,
                                string dsetname,
                                vector<vector<V>>& buffer,
                                bool verbose = false)
    {
        using namespace HighFive;
        try
        {
            File datafile(filename.c_str(), File::ReadWrite);
            DataSet dataset = datafile.getDataSet(dsetname.c_str());

            vector<size_t> dims = dataset.getDimensions();
            if (dims.size() != 2)
            {
                fmt::print("ERROR: Dimension mismatch in read_hdf5_dataset_2d. Got: {}. Expected: {}.\n", dims.size(), 2);
                fmt::print("       filename: {}\n", filename);
                fmt::print("       dataset:  {}\n", dsetname);
                fmt::print("Exiting.\n");
                exit(1);
            }
            dataset.read(buffer);

            if (verbose)
            {
                fmt::print("Read dataset {} from file {}\n", dsetname, filename);
                fmt::print("  Data size: {} x {}\n", dims[0], dims[1]);
            }
        }
        catch (Exception& err)
        {
            // catch and print any HDF5 error
            std::cerr << err.what() << std::endl;
        }

        return;
    }

    template <typename V>
    void read_mpas_data(
            const   diy::Master::ProxyWithLink& cp,
                    MFAInfo&    mfa_info,
                    DomainArgs& args)
    {        
        // Set up MFA and associated data members
        const int nvars = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);

        // Read HDF5 data into STL vectors
        vector<vector<V>>           coords;
        vector<vector<vector<int>>> conn_mats;
        vector<vector<V>>           depth_vecs;
        int                         ncells = 0;

        const int ndatasets = 3;
        string filename = args.infile;
        string coords_name = "tstt/nodes/coordinates";
        vector<string> connect_names;
        vector<string> depth_names;
        conn_mats.resize(ndatasets);
        depth_vecs.resize(ndatasets);
        connect_names.push_back("tstt/elements/Polygon5/connectivity");
        connect_names.push_back("tstt/elements/Polygon6/connectivity");
        connect_names.push_back("tstt/elements/Polygon7/connectivity");
        depth_names.push_back("tstt/elements/Polygon5/tags/bottomDepth");
        depth_names.push_back("tstt/elements/Polygon6/tags/bottomDepth");
        depth_names.push_back("tstt/elements/Polygon7/tags/bottomDepth");

        read_hdf5_dataset_2d<double>(filename, coords_name, coords, true);
        for (int n = 0; n < ndatasets; n++)
        {
            
            read_hdf5_dataset_1d<double>(filename, depth_names[n], depth_vecs[n], true);
            read_hdf5_dataset_2d<int>(filename, connect_names[n], conn_mats[n], true);
            ncells += conn_mats[n].size();
        }

        input = new mfa::PointSet<T>(dom_dim, mfa_info.model_dims(), ncells);
        
        int ofst = 0;
        VectorX<T> centroid(3);
        for (int n = 0; n < ndatasets; n++) // for each class of polygon
        {
            int cellsize = conn_mats[n][0].size();
            for (int i = 0; i < conn_mats[n].size(); i++)   // for each cell in this class
            {
                // compute cell centroid
                centroid.setZero();
                for (int j = 0; j < cellsize; j++)  
                {
                    int vid = conn_mats[n][i][j] - 1;
                    centroid(0) += coords[vid][0];
                    centroid(1) += coords[vid][1];
                    centroid(2) += coords[vid][2];
                }
                centroid = (1.0/cellsize) * centroid;

                // Fill centroid coordinates and depth associated to the cell
                input->domain(ofst+i, 0) = centroid(0);
                input->domain(ofst+i, 1) = centroid(1);
                input->domain(ofst+i, 2) = centroid(2);
                input->domain(ofst+i, 3) = depth_vecs[n][i];
            }

            // Move the offset for the next class of polygons
            ofst += conn_mats[n].size();
        }

        input->set_domain_params();
        this->setup_MFA(cp, mfa_info);

        // Find block bounds for coordinates and values
        bounds_mins = input->domain.colwise().minCoeff();
        bounds_maxs = input->domain.colwise().maxCoeff();
        core_mins   = bounds_mins.head(dom_dim);
        core_maxs   = bounds_maxs.head(dom_dim);

        // debug
        mfa::print_bbox(core_mins, core_maxs, "Core");
        mfa::print_bbox(bounds_mins, bounds_maxs, "Bounds");
    }


    template <typename V>               // floating point type used in HDF5 file (assumed to be only one)
    void read_roms_data(
            const   diy::Master::ProxyWithLink& cp,
                    MFAInfo&    mfa_info,
                    DomainArgs& args)
    {        
        // Set up MFA and associated data members
        const int nvars = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);

        // Read HDF5 data into STL vectors
        vector<vector<V>>   coords;
        vector<vector<int>> conn_mat;
        vector<V>           depth_vec;
        size_t              npts = 0;           // total number of points
        int                 cellsize = 0;
        int                 ncells = 0;
        V                   c_depth = 0;

        // roms
        string filename = args.infile;
        string coords_name = "tstt/nodes/coordinates";
        string cell_connect_name = "tstt/elements/Quad4/connectivity";
        string node_depth_name = "tstt/nodes/tags/bathymetry";

        read_hdf5_dataset_2d<double>(filename, coords_name, coords, true);
        read_hdf5_dataset_1d<double>(filename, node_depth_name, depth_vec, true);
        read_hdf5_dataset_2d<int>(filename, cell_connect_name, conn_mat, true);

        ncells = conn_mat.size();
        cellsize = conn_mat[0].size();

        // Initialize input PointSet from buffers
        input = new mfa::PointSet<T>(dom_dim, mfa_info.model_dims(), ncells);
        VectorX<T> centroid(3);
        for (size_t i = 0; i < input->npts; i++)
        {
            centroid.setZero();
            c_depth = 0;
            for (int j = 0; j < cellsize; j++)
            {
                int vid = conn_mat[i][j] - 1;
                centroid(0) += coords[vid][0];
                centroid(1) += coords[vid][1];
                centroid(2) += coords[vid][2];
                c_depth += depth_vec[vid];
            }
            centroid = (1.0/cellsize) * centroid;
            c_depth = (1.0/cellsize) * c_depth;

            input->domain(i, 0) = centroid[0];
            input->domain(i, 1) = centroid[1];
            input->domain(i, 2) = centroid[2];
            input->domain(i, 3) = c_depth;
        }
        input->set_domain_params();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // Find block bounds for coordinates and values
        bounds_mins = input->domain.colwise().minCoeff();
        bounds_maxs = input->domain.colwise().maxCoeff();
        core_mins   = bounds_mins.head(dom_dim);
        core_maxs   = bounds_maxs.head(dom_dim);

        // debug
        mfa::print_bbox(core_mins, core_maxs, "Core");
        mfa::print_bbox(bounds_mins, bounds_maxs, "Bounds");
    }


    // Compute error metrics between a pointset and an analytical function
    // evaluated at the points in the pointset
    void analytical_error_pointset(
        const diy::Master::ProxyWithLink&   cp,
        mfa::PointSet<T>*                   ps,
        string                              fun,
        vector<T>&                          L1, 
        vector<T>&                          L2,
        vector<T>&                          Linf,
        DomainArgs&                         args,
        const std::function<void(const VectorX<T>&, VectorX<T>&, DomainArgs&, int)>& f = {}) const
    {
        int nvars = ps->nvars();
        if (L1.size() != nvars || L2.size() != nvars || Linf.size() != nvars)
        {
            cerr << "ERROR: Error metric vector sizes do not match in analytical_error_pointset().\nAborting" << endl;
            exit(1);
        }

        // Compute the analytical error at each point
        T l1err = 0, l2err = 0, linferr = 0;
        VectorX<T> dom_pt(ps->geom_dim());

        for (int k = 0; k < ps->nvars(); k++)
        {
            VectorX<T> true_pt(ps->var_dim(k));
            VectorX<T> test_pt(ps->var_dim(k));
            VectorX<T> residual(ps->var_dim(k));

            for (auto pt_it = ps->begin(), pt_end = ps->end(); pt_it != pt_end; ++pt_it)
            {
                pt_it.geom_coords(dom_pt); // extract the geometry coordinates

                // Get exact value. If 'f' is non-NULL, ignore 'fun'
                if (f)
                    f(dom_pt, true_pt, args, k);
                else
                    evaluate_function(fun, dom_pt, true_pt, args, k);

                // Get approximate value
                pt_it.var_coords(k, test_pt);

                // NOTE: For now, we are using the norm of the residual for all error statistics.
                //       Is this the most appropriate way to measure errors norms of a vector field?
                //       May want to consider revisiting this.
                //
                // Compute errors for this point. When the science variable is vector-valued, we 
                // distinguish between the L1, L2, and Linfty distances. L1 distance is 
                // used for 'sum_errs', L2 for 'sum_sq_errs,' and Linfty for 'max_err.'
                // Thus, 'max_err' reports the maximum difference in any vector
                // component, taken over all of the points in the Pointset.
                //
                // n.b. When the science variable is scalar valued, L2 error and Linfty error are the same. 
                residual = (true_pt - test_pt).cwiseAbs();
                // l1err   = residual.sum();
                l2err   = residual.norm();          // L2 difference between vectors 
                // linferr = residual.maxCoeff();      // Maximum difference in components

                // Update error statistics
                L1[k]   += l2err;
                L2[k]   += l2err * l2err;
                if (l2err > Linf[k]) Linf[k] = l2err;
            }

            L1[k] = L1[k] / ps->npts;
            L2[k] = sqrt(L2[k] / ps->npts);
        }
    }


    // Simplified function signature when we don't need to keep the PointSets
    void analytical_error_field(
        const diy::Master::ProxyWithLink&   cp,
        vector<int>&                        grid,               // size of regular grid
        string                              fun,                // analytical function name
        vector<T>&                          L1,                 // (output) L-1 norm
        vector<T>&                          L2,                 // (output) L-2 norm
        vector<T>&                          Linf,               // (output) L-infinity norm
        DomainArgs&                         args,               // input args
        const std::function<void(const VectorX<T>&, VectorX<T>&, DomainArgs&, int)>& f = {},
        vector<T>                           subset_mins = vector<T>(),
        vector<T>                           subset_maxs = vector<T>() ) // (optional) subset of the domain to consider for errors
    {
        mfa::PointSet<T>* unused = nullptr;
        analytical_error_field(cp, grid, fun, L1, L2, Linf, args, unused, unused, unused, f, subset_mins, subset_maxs);
    }

    // Compute error field on a regularly spaced grid of points. The size of the grid
    // is given by args.ndom_pts. Error metrics are saved in L1, L2, Linf. The fields 
    // of the exact, approximate, and residual data are save to PointSets.
    void analytical_error_field(
        const diy::Master::ProxyWithLink&   cp,
        vector<int>&                        grid,               // size of regular grid
        string                              fun,                // analytical function name
        vector<T>&                          L1,                 // (output) L-1 norm
        vector<T>&                          L2,                 // (output) L-2 norm
        vector<T>&                          Linf,               // (output) L-infinity norm
        DomainArgs&                         args,               // input args
        mfa::PointSet<T>*&                  exact_pts,          // PointSet to contain analytical signal
        mfa::PointSet<T>*&                  approx_pts,         // PointSet to contain approximation
        mfa::PointSet<T>*&                  error_pts,          // PointSet to contain errors
        const std::function<void(const VectorX<T>&, VectorX<T>&, DomainArgs&, int)>& f = {},
        vector<T>                           subset_mins = vector<T>(),
        vector<T>                           subset_maxs = vector<T>() ) // (optional) subset of the domain to consider for errors
    {
        int nvars = mfa->nvars();
        if (L1.size() != nvars || L2.size() != nvars || Linf.size() != nvars)
        {
            cerr << "ERROR: Error metric vector sizes do not match in analytical_error_field().\nAborting" << endl;
            exit(1);
        }

        // Check if we accumulated errors over subset of domain only and report
        bool do_subset = false;
        bool in_box = true;
        if (subset_mins.size() != 0)
        {
            do_subset = true;

            if (cp.gid() == 0)
            {
                cout << "Accumulating errors over subset of domain" << endl;
                cout << "  subset mins: " << mfa::print_vec(subset_mins) << endl;
                cout << "  subset maxs: " << mfa::print_vec(subset_maxs) << endl;
            }

            if (subset_mins.size() != subset_maxs.size())
            {
                cerr << "ERROR: Dimensions of subset_mins and subset_maxs do not match" << endl;
                exit(1);
            }
            if (subset_mins.size() != dom_dim)
            {
                cerr << "ERROR: subset dimension does not match dom_dim" << endl;
                exit(1);
            }
        }

        // Size of grid on which to test error
        VectorXi test_pts(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            test_pts(i) = grid[i];
        }

        // Free any existing memory at PointSet pointers
        if (exact_pts) cerr << "Warning: Overwriting \'exact_pts\' pointset in analytical_error_field()" << endl;
        if (approx_pts) cerr << "Warning: Overwriting \'approx_pts\' pointset in analytical_error_field()" << endl;
        if (error_pts) cerr << "Warning: Overwriting \'error_pts\' pointset in analytical_error_field()" << endl;
        delete exact_pts;
        delete approx_pts;
        delete error_pts;

        // Set up PointSets with grid parametrizations
        exact_pts = new mfa::PointSet<T>(mfa->dom_dim, mfa->model_dims(), test_pts.prod(), test_pts);
        approx_pts= new mfa::PointSet<T>(mfa->dom_dim, mfa->model_dims(), test_pts.prod(), test_pts);
        error_pts = new mfa::PointSet<T>(mfa->dom_dim, mfa->model_dims(), test_pts.prod(), test_pts);
        approx_pts->set_grid_params();

        // Decode on above-specified grid
        mfa->Decode(*approx_pts, false);

        // Copy geometric point coordinates into error and exact PointSets
        exact_pts->domain.leftCols(exact_pts->geom_dim()) = approx_pts->domain.leftCols(approx_pts->geom_dim());
        error_pts->domain.leftCols(error_pts->geom_dim()) = approx_pts->domain.leftCols(approx_pts->geom_dim());

        // Compute the analytical error at each point and accrue errors
        T l1err = 0, l2err = 0, linferr = 0;
        VectorX<T> dom_pt(approx_pts->geom_dim());

        for (int k = 0; k < nvars; k++)
        {
            VectorX<T> true_pt(approx_pts->var_dim(k));
            VectorX<T> test_pt(approx_pts->var_dim(k));
            VectorX<T> residual(approx_pts->var_dim(k));
            int num_pts_in_box = 0;

            for (auto pt_it = approx_pts->begin(), pt_end = approx_pts->end(); pt_it != pt_end; ++pt_it)
            {
                pt_it.geom_coords(dom_pt); // extract the geometry coordinates

                // Get exact value. If 'f' is non-NULL, ignore 'fun'
                if (f)
                    f(dom_pt, true_pt, args, k);
                else
                    evaluate_function(fun, dom_pt, true_pt, args, k);
                    
                // Get approximate value
                pt_it.var_coords(k, test_pt);

                // Update error field
                residual = (true_pt - test_pt).cwiseAbs();
                for (int j = 0; j < error_pts->var_dim(k); j++)
                {
                    error_pts->domain(pt_it.idx(), error_pts->var_min(k) + j) = residual(j);
                    exact_pts->domain(pt_it.idx(), exact_pts->var_min(k) + j) = true_pt(j);
                }

                // Accrue error only in subset
                in_box = true;
                if (do_subset) 
                {
                    for (int i = 0; i < dom_dim; i++)
                        in_box = in_box && (dom_pt(i) >= subset_mins[i]) && (dom_pt(i) <= subset_maxs[i]);
                }

                if (in_box)
                {
                    // NOTE: For now, we are using the norm of the residual for all error statistics.
                    //       Is this the most appropriate way to measure errors norms of a vector field?
                    //       May want to consider revisiting this.
                    //
                    // l1err   = residual.sum();           // L1 difference between vectors
                    l2err   = residual.norm();          // L2 difference between vectors 
                    // linferr = residual.maxCoeff();      // Maximum difference in components

                    L1[k]   += l2err;
                    L2[k]   += l2err * l2err;
                    if (l2err > Linf[k]) Linf[k] = l2err;

                    num_pts_in_box++;
                }
            }

            L1[k] = L1[k] / num_pts_in_box;
            L2[k] = sqrt(L2[k] / num_pts_in_box);
        }
    }
};

#endif // _MFA_CBLOCK