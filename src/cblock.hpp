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

#include    "moab/Core.hpp"

#include    "domain_args.hpp"

using namespace std;
using namespace moab;

template <typename V>
struct MOABReader
{
    int max_id;
    string filename;
    vector<vector<V>> coords;
    vector<vector<int>> elements;
    map<string, int> start_ids;
    map<string, int> end_ids; // last id of an element type (inclusive)
    map<string, int> cell_counts;
    map<string, int> faces_per_cell_;

    // face_cell_map[CellType][i] is a vector containing all cells of type CellType that contain face i
    // all indices are global
    map<string, map<int, vector<int>>> face_cell_map; 

    // Each string key is a cell type name (or "node")
    // Each value is vector of datasets, so 
    // data["Quad4"]["Salinity"] would be the vector storing the Salinity data read from file
    map<string, map<string, vector<V>>> data;

    MOABReader(string filename_) :
        filename(filename_)
    {
        max_id = get_max_id(filename);
        elements.resize(max_id);

        read_vertices();

        // Get MOAB instance
        Interface* mb = new( std::nothrow ) Core;
    }

    template <typename DT>
    static DT read_hdf5_group_attr(   string filename,
                        string group_name,
                        string attrname,
                        bool verbose = false)
    {
        using namespace HighFive;

        DT attr_value[1];
        try
        {
            File datafile(filename.c_str(), File::ReadWrite);
            Group group = datafile.getGroup(group_name.c_str());
            Attribute attr = group.getAttribute(attrname.c_str());
            attr.read<DT>(attr_value);
        }
        catch (Exception& err)
        {
            // catch and print any HDF5 error
            fmt::print("{}\n", err.what());
        }

        fmt::print("Debug attribute {} is {}\n", attrname, attr_value[0]);

        return attr_value[0];
    }

    template <typename DT>
    static DT read_hdf5_dataset_attr(   string filename,
                        string dataset_name,
                        string attrname,
                        bool verbose = false)
    {
        using namespace HighFive;

        DT attr_value[1];
        try
        {
            File datafile(filename.c_str(), File::ReadWrite);
            DataSet dset = datafile.getDataSet(dataset_name.c_str());
            Attribute attr = dset.getAttribute(attrname.c_str());
            attr.read<DT>(attr_value);
        }
        catch (Exception& err)
        {
            // catch and print any HDF5 error
            fmt::print("{}\n", err.what());
        }

        fmt::print("Debug attribute {} is {}\n", attrname, attr_value[0]);

        return attr_value[0];
    }

    template <typename DT>
    static void read_hdf5_dataset_1d(  string filename,
                                string dsetname,
                                vector<DT>& buffer,
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
            fmt::print("{}\n", err.what());
        }

        return;
    }

    template <typename DT>
    static void read_hdf5_dataset_2d(  string filename,
                                string dsetname,
                                vector<vector<DT>>& buffer,
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
            fmt::print("{}\n", err.what());
        }

        return;
    }

    vector<string> get_element_names()
    {
        vector<string> names;

        for (const auto& it : start_ids)
        {
            names.push_back(it.first);
        }

        return names;
    }

    string get_cell_type(int i)
    {
        for (auto name : get_element_names())
        {
            if (i >= start_ids[name] && i <= end_ids[name])
            {
                return name;
            }
        }

        cerr << "ERROR: Could not find element type for index " << i << endl;
        exit(1);

        return "";
    }

    int get_max_id(string filename)
    {
        string max_id_attr_name = "max_id";
        int id = read_hdf5_group_attr<int>(filename, "tstt", max_id_attr_name, true);

        return id;
    }

    void read_vertices()
    {
        string coords_name = "tstt/nodes/coordinates";
        read_hdf5_dataset_2d<V>(filename, coords_name, coords, true);

        int start_id = read_hdf5_dataset_attr<int>(filename, coords_name, "start_id", true);
        start_ids["nodes"] = start_id;
        end_ids["nodes"] = start_id + coords.size() - 1;
        cell_counts["nodes"] = coords.size();
        faces_per_cell_["nodes"] = 0;
        data["nodes"] = map<string, vector<V>>();
    }

    void read_cells(string cell_name)
    {
        vector<vector<int>> cell_conn;
        string cell_conn_name = "tstt/elements/" + cell_name + "/connectivity";

        int start_id = read_cell_attr<int>(cell_name, "start_id");
        read_hdf5_dataset_2d<int>(filename, cell_conn_name, cell_conn, true);

        // Copy data into global elements vector
        for (int i = 0; i < cell_conn.size(); i++)
        {
            int cell_id = i+start_id;
            elements[cell_id] = cell_conn[i];
        }

        start_ids[cell_name] = start_id;
        end_ids[cell_name] = start_id + cell_conn.size() - 1;
        cell_counts[cell_name] = cell_conn.size();
        faces_per_cell_[cell_name] = cell_conn[0].size();
        data[cell_name] = map<string, vector<V>>();
    }

    void build_face_cell_map()
    {
        vector<string> enames = get_element_names();
        for (auto face_name : enames)
        {
            face_cell_map[face_name] = map<int, vector<int>>();
        }

        // For each cell type
        for (auto cell_name : enames)
        {
            // For each cell of this type
            for (int i = 0; i < num_cells(cell_name); i++)
            {
                int start_id = start_ids[cell_name];
                int cell_id = start_id + i;

                // For each face of this cell
                for (auto face_id : elements[cell_id])
                {
                    // Get name of face type
                    string face_name = get_cell_type(face_id);

                    // Check if any cells have been added to the list for this face
                    if (face_cell_map[face_name].count(face_id) == 0)
                    {
                        face_cell_map[face_name][face_id] = vector<int>();
                    }

                    // Add cell to list
                    face_cell_map[face_name][face_id].push_back(cell_id);
                }
            }
        }


        //  Don't think this sanity check makes sense as written 
        //
        // // Sanity check
        // set<string> no_parents; // list of cell types with no parent cells
        // for (auto face_name : enames)
        // {
        //     for (int i = 0; i < num_cells(face_name); i++)
        //     {
        //         // Check if there are any parents for this face
        //         if (face_cell_map[face_name].count(i) == 0)
        //         {
        //             // If this is the first cell, mark the cell type as having no parents
        //             // This will hapen for full-dimensional cells
        //             if (i == 0)
        //             {
        //                 no_parents.insert(face_name);
        //             } 
        //             else
        //             {
        //                 // if this is not the first cell, but this face type has not previously
        //                 // been marked as having no parents, then something is wrong.
        //                 // This means there are some faces with parents (namely, the first), and 
        //                 // some without. This should not happen.
        //                 if (no_parents.count(face_name) == 0)
        //                 {
        //                     cerr << face_name << " " << i << " does not have parent cells, but other cells of this type do" << endl;
        //                     cerr << "This should not happen. Exiting" << endl;
        //                     exit(1);
        //                 }
        //             }

        //             continue; // skip to next face_id
        //         }
        //     }
        // }

        // cout << "Cell types with no parent cells are: ";
        // for (auto name : no_parents)
        // {
        //     cout << name << " ";
        // }
        // cout << endl;
    }

    template<typename AT>
    AT read_cell_attr(string cell_name, string attr_name)
    {
        // string full_attr_name = "/tstt/elements/" + cell_name + "/connectivity/" + attr_name;
        return read_hdf5_dataset_attr<AT>(filename, "/tstt/elements/" + cell_name + "/connectivity/", attr_name);
    }

    void read_cell_data(string cell_name, string data_name)
    {
        // read cell data into buffer
        vector<V> data_vec;
        string tag_name = "tstt/elements/" + cell_name + "/tags/" + data_name;
        read_hdf5_dataset_1d<V>(filename, tag_name, data_vec, true);

        // If 'data' does not have a map for given cell_name yet, default construct one
        if (data.count(cell_name) == 0)
        {
            data[cell_name] = map<string, vector<V>>();
        }

        // Inefficient deep copy
        data[cell_name][data_name] = data_vec;
    }

    int num_verts()
    {
        return coords.size();
    }

    int num_cells(string cell_name)
    {
        if (cell_counts.count(cell_name) == 0)
        {
            cerr << "ERROR: Lookup error for key \'" << cell_name << "\' in MOABConnectivity::num_cells()" << endl;
            exit(1);
        }
        return cell_counts[cell_name];
    }

    int faces_per_cell(string cell_name)
    {
        if (faces_per_cell_.count(cell_name) == 0)
        {
            cerr << "ERROR: Lookup error for key \'" << cell_name << "\' in MOABConnectivity::faces_per_cell()" << endl;
            exit(1);
        }
        return faces_per_cell_[cell_name];
    }

    // Return the element ids for the parent cells containing face i
    // Return the global index of the parent cells
    // input i is the local index of the faces. So i is in [0, num_faces]
    const vector<int>& get_parent_cells(string face_name, int i)
    {
        int start_id = start_ids[face_name];
        return face_cell_map[face_name][start_id + i];
    }

    // Look up faces associated with the ith cell of type 'cell_name'
    const vector<int>& get_faces(string cell_name, int i)
    {
        int start_id = start_ids[cell_name];
        return elements[start_id + i];
    }

    // Look up faces associated with the ith overall cell (global indexing)
    const vector<int>& get_faces(int i)
    {
        return elements[i];
    }

    // Look up vector of coordinates for the ith node
    const vector<V>& get_coords(int i)
    {
        int start_id = start_ids["nodes"];  // this is usually == 1
        return coords[i-start_id];
    }

    const vector<V>& get_data(string cell_name, string data_name)
    {
        // int data_id = dataset_ids[data_name];
        return data[cell_name][data_name];
    }

    // return data value for data_name, using the global element index i
    V get_element_data(int i, string data_name)
    {
        string cell_name = get_cell_type(i);
        int cell_id = i - start_ids[cell_name];
        return data[cell_name][data_name][cell_id];
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

    template <typename V>
    void read_mpas_data_3d(
            const   diy::Master::ProxyWithLink& cp,
                    string filename,
                    MFAInfo& mfa_info,
                    const VectorX<T>& roms_mins = VectorX<T>(),
                    const VectorX<T>& roms_maxs = VectorX<T>())
    {
        VectorXi model_dims(3);
        model_dims << 3, 1, 1;          // geometry, bathymetry, salinity, temperature

        VectorX<T> mins = VectorX<T>::Constant(3, 100000000000000);
        VectorX<T> maxs = VectorX<T>::Constant(3, -100000000000000);

        MOABReader<V> mr(filename);
        mr.read_cells("Quad4");
        mr.read_cells("Polygon6");
        mr.read_cells("Polyhedron8");
        mr.read_cell_data("Polyhedron8", "Salinity3d");
        mr.read_cell_data("Polyhedron8", "Temperature3d");
        mr.build_face_cell_map();

        // Create lists of surface faces and the 3D cells that contain them
        int p6_start = mr.start_ids["Polygon6"];
        vector<int> surface_face_ids;
        vector<int> surface_parent_ids;
        for (int i = 0; i < mr.num_cells("Polygon6"); i++)
        {
            vector<int> parents = mr.get_parent_cells("Polygon6", i);
            int nparents = parents.size();
            if (nparents == 1)
            {
                // Add global id of polygon cell
                surface_face_ids.push_back(i + p6_start);

                // Add global id of parent cell
                surface_parent_ids.push_back(parents[0]);
            }
            else if (nparents = 0)
            {
                cerr << "surface " << i << " has no parents" << endl;
            }
            else if (nparents > 2)
            {
                cerr << "surface " << i << " has " << nparents << " parents" << endl;
            }
        }

        // Create MFA PointSet to hold all data
        int num_surface_faces = surface_face_ids.size();
        int num_polyhedra = mr.num_cells("Polyhedron8");
        mpas_input = new mfa::PointSet<T>(dom_dim, model_dims, num_surface_faces + num_polyhedra);

        // Data structures for reading cell data
        vector<T> centroid(3, 0);
        vector<T> vertex_coords;
        vector<int> faces;
        vector<int> verts;
        set<int> cell_vertices;

        // Add the surface facets to the PointSet
        for (int i = 0; i < num_surface_faces; i++)
        {
            centroid.clear();
            centroid.assign(3, 0);
            cell_vertices.clear();

            // Get vertices of face
            int face_id = surface_face_ids[i];
            verts = mr.get_faces(face_id);
            for (auto v : verts)
            {
                cell_vertices.insert(v);
            }

            // Compute face centroid
            int num_vertices = cell_vertices.size();
            if (num_vertices != 6) {cerr << "BAD VERTICES" << endl; exit(1);}   // we expect the faces to be hexagonal
            for (auto v : cell_vertices)
            {
                vertex_coords = mr.get_coords(v);
                
                // Compute bounding box to cover all MPAS vertices, not just their centroids
                // The depth levels that define the knot distribution are taken from the MPAS
                // data file, so we need to make sure that each of these depth levels is inside
                // our bounding box. If the bounding box was set from centroids only, then the
                // lowest or highest depths might lie outside the box.
                if (vertex_coords[0] < mins[0]) mins[0] = vertex_coords[0];
                if (vertex_coords[0] > maxs[0]) maxs[0] = vertex_coords[0];
                if (vertex_coords[1] < mins[1]) mins[1] = vertex_coords[1];
                if (vertex_coords[1] > maxs[1]) maxs[1] = vertex_coords[1];
                if (vertex_coords[2] < mins[2]) mins[2] = vertex_coords[2];
                if (vertex_coords[2] > maxs[2]) maxs[2] = vertex_coords[2];

                // Compute centroid
                centroid[0] += vertex_coords[0] / num_vertices;
                centroid[1] += vertex_coords[1] / num_vertices;
                centroid[2] += vertex_coords[2] / num_vertices;
            }

            // Add to PointSet with variable data
            mpas_input->domain(i, 0) = centroid[0];
            mpas_input->domain(i, 1) = centroid[1];
            mpas_input->domain(i, 2) = centroid[2];
            mpas_input->domain(i, 3) = mr.get_element_data(surface_parent_ids[i], "Salinity3d");
            mpas_input->domain(i, 4) = mr.get_element_data(surface_parent_ids[i], "Temperature3d");
        }

        // Add the volumetric cells to the PointSet
        for (int i = 0; i < mr.num_cells("Polyhedron8"); i++)
        {
            centroid.clear();
            centroid.assign(3, 0); // Fill with all zeros
            cell_vertices.clear();

            // Get vertices of cell
            // Use a Set to hold vertices because they will be duplicated as 
            // we loop of each face of the cell. (Set de-duplicates for us)
            faces = mr.get_faces("Polyhedron8", i);
            for (auto face_id : faces)
            {
                verts = mr.get_faces(face_id);
                for (auto v : verts)
                {
                    cell_vertices.insert(v);
                }
            }

            // Compute cell centroid
            int num_vertices = cell_vertices.size();
            for (auto v : cell_vertices)
            {
                vertex_coords = mr.get_coords(v);

                // Compute bounding box to cover all MPAS vertices, not just their centroids
                // The depth levels that define the knot distribution are taken from the MPAS
                // data file, so we need to make sure that each of these depth levels is inside
                // our bounding box. If the bounding box was set from centroids only, then the
                // lowest or highest depths might lie outside the box.
                if (vertex_coords[0] < mins[0]) mins[0] = vertex_coords[0];
                if (vertex_coords[0] > maxs[0]) maxs[0] = vertex_coords[0];
                if (vertex_coords[1] < mins[1]) mins[1] = vertex_coords[1];
                if (vertex_coords[1] > maxs[1]) maxs[1] = vertex_coords[1];
                if (vertex_coords[2] < mins[2]) mins[2] = vertex_coords[2];
                if (vertex_coords[2] > maxs[2]) maxs[2] = vertex_coords[2];
                centroid[0] += vertex_coords[0] / num_vertices;
                centroid[1] += vertex_coords[1] / num_vertices;
                centroid[2] += vertex_coords[2] / num_vertices;
            }

            // Add to PointSet with variable data
            mpas_input->domain(num_surface_faces + i, 0) = centroid[0];
            mpas_input->domain(num_surface_faces + i, 1) = centroid[1];
            mpas_input->domain(num_surface_faces + i, 2) = centroid[2];
            mpas_input->domain(num_surface_faces + i, 3) = mr.get_data("Polyhedron8", "Salinity3d")[i];
            mpas_input->domain(num_surface_faces + i, 4) = mr.get_data("Polyhedron8", "Temperature3d")[i];
        }

        // Update bounding box if roms bounds were passed
        if (roms_mins.size() == 3 && roms_maxs.size() == 3)
        {
            bounds_mins = mpas_input->domain.colwise().minCoeff();
            bounds_maxs = mpas_input->domain.colwise().maxCoeff();
            core_mins   = bounds_mins.head(dom_dim);
            core_maxs   = bounds_maxs.head(dom_dim);

            if (roms_mins(0) < mins(0))
            {
                cout << "Updating xmin" << endl;
                mins(0) = roms_mins(0);
            }
            if (roms_maxs(0) > maxs(0))
            {
                cout << "Updating xmax" << endl;
                maxs(0) = roms_maxs(0);
            }
            if (roms_mins(1) < mins(1))
            {
                cout << "Updating ymin" << endl;
                mins(1) = roms_mins(1);
            }
            if (roms_maxs(1) > maxs(1))
            {
                cout << "Updating ymax" << endl;
                maxs(1) = roms_maxs(1);
            }
            if (roms_mins(2) < mins(2))
            {
                cout << "Updating zmin" << endl;
                mins(2) = roms_mins(2);
            }
            if (roms_maxs(2) > maxs(2))
            {
                cout << "Updating zmax" << endl;
                maxs(2) = roms_maxs(2);
            }
        }

        // Set parametrization
        mpas_input->set_domain_params(mins, maxs);

        // Set up MFA from user-specified options
        this->setup_MFA(cp, mfa_info);

        // Find block bounds for coordinates and values
        bounds_mins = mpas_input->domain.colwise().minCoeff();
        bounds_maxs = mpas_input->domain.colwise().maxCoeff();
        core_mins   = bounds_mins.head(dom_dim);
        core_maxs   = bounds_maxs.head(dom_dim);

        // debug
        mfa::print_bbox(mins, maxs, "MPAS Custom");
        mfa::print_bbox(core_mins, core_maxs, "MPAS Core");
        mfa::print_bbox(bounds_mins, bounds_maxs, "MPAS Bounds");
    }

    template <typename V>
    void read_roms_data_3d(
            const   diy::Master::ProxyWithLink& cp,
                    string filename)
    {
        VectorXi model_dims(1);
        model_dims << 3;          // geometry, bathymetry, salinity, temperature

        // Only reading vertex locations
        MOABReader<V> mr(filename);

        // Create MFA PointSet and add vertices
        roms_input = new mfa::PointSet<T>(dom_dim, model_dims, mr.num_verts());
        for (int i = 0; i < roms_input->npts; i++)
        {
            roms_input->domain(i, 0) = mr.coords[i][0];
            roms_input->domain(i, 1) = mr.coords[i][1];
            roms_input->domain(i, 2) = mr.coords[i][2];
        }

        // Compute parametrization and set up MFA
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

    template <typename V>
    void read_mpas_data(
            const   diy::Master::ProxyWithLink& cp,
                    string      mpasfile,
                    MFAInfo&    mfa_info)
    {
        VectorXi model_dims(4);
        model_dims << 3, 1, 1, 1;          // geometry, bathymetry, salinity, temperature

        // Read HDF5 data into STL vectors
        vector<vector<V>>           coords;
        vector<vector<vector<int>>> conn_mats;
        vector<vector<V>>           depth_vecs, salinity_vecs, temp_vecs;
        int                         ncells = 0;

        const int ndatasets = 3;
        string filename = mpasfile;
        string coords_name = "tstt/nodes/coordinates";
        vector<string> connect_names;
        vector<string> var_names;
        conn_mats.resize(ndatasets);
        depth_vecs.resize(ndatasets);
        salinity_vecs.resize(ndatasets);
        temp_vecs.resize(ndatasets);
        connect_names.push_back("tstt/elements/Polygon5/connectivity");
        connect_names.push_back("tstt/elements/Polygon6/connectivity");
        connect_names.push_back("tstt/elements/Polygon7/connectivity");
        var_names.push_back("tstt/elements/Polygon5/tags/");
        var_names.push_back("tstt/elements/Polygon6/tags/");
        var_names.push_back("tstt/elements/Polygon7/tags/");

        // Read vertex coordinates
        MOABReader<V>::read_hdf5_dataset_2d(filename, coords_name, coords, true);

        // For each kind of polygonal cell, read each variable and cell connectivity
        for (int n = 0; n < ndatasets; n++)
        {
            // read each variable into buffers
            MOABReader<V>::read_hdf5_dataset_1d(filename, var_names[n]+"bottomDepth", depth_vecs[n], true);
            MOABReader<V>::read_hdf5_dataset_1d(filename, var_names[n]+"salinity", salinity_vecs[n], true);
            MOABReader<V>::read_hdf5_dataset_1d(filename, var_names[n]+"temperature", temp_vecs[n], true);

            // read cell connectivity
            MOABReader<V>::read_hdf5_dataset_2d(filename, connect_names[n], conn_mats[n], true);
            ncells += conn_mats[n].size();
        }

        // Create MFA PointSet to hold all data
        mpas_input = new mfa::PointSet<T>(dom_dim, model_dims, ncells);
        
        // Fill pointset
        int ofst = 0;
        VectorX<T> centroid(3);
        for (int n = 0; n < ndatasets; n++) // for each class of polygon
        {
            int cellsize = conn_mats[n][0].size();
            for (int i = 0; i < conn_mats[n].size(); i++)   // for each cell in this class
            {
                // compute cell centroid
                centroid.setZero();
                for (int j = 0; j < cellsize; j++)      // for each vertex in cell
                {
                    int vid = conn_mats[n][i][j] - 1;
                    centroid(0) += coords[vid][0];
                    centroid(1) += coords[vid][1];
                    centroid(2) += coords[vid][2];
                }
                centroid = (1.0/cellsize) * centroid;

                // Fill centroid coordinates and depth associated to the cell
                mpas_input->domain(ofst+i, 0) = centroid(0);
                mpas_input->domain(ofst+i, 1) = centroid(1);
                mpas_input->domain(ofst+i, 2) = centroid(2);
                mpas_input->domain(ofst+i, 3) = depth_vecs[n][i];
                mpas_input->domain(ofst+i, 4) = salinity_vecs[n][i];
                mpas_input->domain(ofst+i, 5) = temp_vecs[n][i];                
            }

            // Move the offset for the next class of polygons
            ofst += conn_mats[n].size();
        }

        // Compute parametrization and set up MFA
        mpas_input->set_domain_params();
        this->setup_MFA(cp, mfa_info);

        // Find block bounds for coordinates and values
        bounds_mins = mpas_input->domain.colwise().minCoeff();
        bounds_maxs = mpas_input->domain.colwise().maxCoeff();
        core_mins   = bounds_mins.head(dom_dim);
        core_maxs   = bounds_maxs.head(dom_dim);

        // debug
        mfa::print_bbox(core_mins, core_maxs, "Core");
        mfa::print_bbox(bounds_mins, bounds_maxs, "Bounds");
    }


    template <typename V>               // floating point type used in HDF5 file (assumed to be only one)
    void read_roms_data(
            const   diy::Master::ProxyWithLink& cp,
                    string      romsfile)
    {        
        VectorXi model_dims(2);
        model_dims << 3, 1;          // reading geometry and bathymetry only

        // Read HDF5 data into STL vectors
        vector<vector<V>>   coords;
        vector<vector<int>> conn_mat;
        vector<V>           depth_vec;
        size_t              npts = 0;           // total number of points

        // roms
        string filename = romsfile;
        string coords_name = "tstt/nodes/coordinates";
        string cell_connect_name = "tstt/elements/Quad4/connectivity";
        string node_depth_name = "tstt/nodes/tags/bathymetry";

        MOABReader<V>::read_hdf5_dataset_2d(filename, coords_name, coords, true);
        MOABReader<V>::read_hdf5_dataset_1d(filename, node_depth_name, depth_vec, true);
        MOABReader<V>::read_hdf5_dataset_2d(filename, cell_connect_name, conn_mat, true);

        npts = coords.size();

        // Initialize input PointSet from buffers
        roms_input = new mfa::PointSet<T>(dom_dim, model_dims, npts);
        for (size_t i = 0; i < roms_input->npts; i++)
        {
            roms_input->domain(i, 0) = coords[i][0];
            roms_input->domain(i, 1) = coords[i][1];
            roms_input->domain(i, 2) = coords[i][2];
            roms_input->domain(i, 3) = depth_vec[i];
        }
        roms_input->set_domain_params();

        // Find block bounds for coordinates and values
        bounds_mins = roms_input->domain.colwise().minCoeff();
        bounds_maxs = roms_input->domain.colwise().maxCoeff();
        core_mins   = bounds_mins.head(dom_dim);
        core_maxs   = bounds_maxs.head(dom_dim);

        // debug
        mfa::print_bbox(core_mins, core_maxs, "Core");
        mfa::print_bbox(bounds_mins, bounds_maxs, "Bounds");
    }

    void remap(
        const diy::Master::ProxyWithLink&   cp,
        MFAInfo&    info)
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

        VectorX<T> roms_mins = roms_input->mins();
        VectorX<T> roms_maxs = roms_input->maxs();
        VectorX<T> mpas_mins = mpas_input->mins();
        VectorX<T> mpas_maxs = mpas_input->maxs();
        VectorX<T> mpas_diff = mpas_maxs - mpas_mins;

        // Compute parametrization of ROMS points in terms of MPAS domain
        shared_ptr<mfa::Param<T>> new_param = make_shared<mfa::Param<T>>(dom_dim);
        new_param->param_list.resize(roms_input->npts, dom_dim);
        for (size_t k = 0; k < dom_dim; k++)
        {
            new_param->param_list.col(k) = (roms_input->domain.col(k).array() - mpas_mins(k)) * (1/mpas_diff(k));
        }

        // Set parametrization object for the PointSet we will decode into
        mpas_approx = new mfa::PointSet<T>(new_param, mpas_input->model_dims());

        // Evaluate MFA
        mfa->Decode(*mpas_approx, false);

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