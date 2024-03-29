//--------------------------------------------------------------
// writes all vtk files for initial, approximated, and control points
//
// optionally generates test data for analytical functions and writes to vtk
//
// output precision is float irrespective whether input is float or double
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include    "mfa/mfa.hpp"
#include    <iostream>
#include    <stdio.h>

#include    <diy/master.hpp>
#include    <diy/io/block.hpp>

#include    "opts.h"
#include    "writer.hpp"
#include    "cblock.hpp"

using B = CBlock<real_t>;

// 3d point or vector
struct vec3d
{
    float x, y, z;
    float mag() { return sqrt(x*x + y*y + z*z); }
};

// TODO: Only scalar-valued and 3D vector-valued variables are supported (because of the VTK writer)
// If a variable has a different output dimension, the writer will skip that variable and continue.
template<typename T>
void write_pointset_vtk(mfa::PointSet<T>* ps, char* filename, int sci_var = -1)
{
    if (ps == nullptr)
    {
        cout << "Did not write " << filename << " due to uninitialized pointset" << endl;
        return;
    }
    if (ps->npts == 0)
    {
        cout << "Did not write " << filename << " due to empty pointset" << endl;
        return;
    }

    int dom_dim = ps->dom_dim;
    int pt_dim  = ps->pt_dim;
    int geom_dim = ps->geom_dim();
    int nvars = ps->nvars();
    bool include_var = true;        // Include the specified science variable in the geometry coordinates
    int var_col = ps->model_dims().head(sci_var + 1).sum(); // column of the variable to be visualized

    // Sanity checks and modify 'include_var' if settings conflict
    if (geom_dim < 1 || geom_dim > 3)
    {
        cerr << "Did not write " << filename << " due to improper dimension in pointset" << endl;
        return;
    }
    if (sci_var < 0)
    {
        include_var = false;
    }
    else if (ps->var_dim(sci_var) != 1 && geom_dim < 3)
    {
        cerr << "For " << filename << ", specified science variable (#" << sci_var << ") is not a scalar. Output will be planar." << endl;
        include_var = false;
    }

    vector<int> npts_dim;  // only used if data is structured
    if (ps->is_structured())
    {
        for (size_t k = 0; k < 3; k++)
        {
            if (k < dom_dim) 
                npts_dim.push_back(ps->ndom_pts(k));
            else
                npts_dim.push_back(1);
        }
    }
    
    float** pt_data = new float*[nvars];
    for (size_t k = 0; k < nvars; k++)
    {
        pt_data[k]  = new float[ps->npts * ps->var_dim(k)];
    }

    vec3d           pt;
    vector<vec3d>   pt_coords;
    for (int j = 0; j < ps->npts; j++)
    {
        // Add geometric coordinates
        if (geom_dim == 1)
        {
            pt.x = ps->domain(j, 0);
            pt.y = include_var ? ps->domain(j, var_col) : 0.0;
            pt.z = 0.0;
        }
        else if (geom_dim == 2)
        {
            pt.x = ps->domain(j, 0);
            pt.y = ps->domain(j, 1);
            pt.z = include_var ? ps->domain(j, var_col) : 0.0;
        }
        else
        {
            pt.x = ps->domain(j, 0);
            pt.y = ps->domain(j, 1);
            pt.z = ps->domain(j, 2);
        }
        pt_coords.push_back(pt);

        // Add science variable data
        int offset_idx = 0;
        int all_vars_dim = pt_dim - geom_dim;
        for (int k = 0; k < nvars; k++)
        {
            int vd = ps->var_dim(k);
            for (int l = 0; l < vd; l++)
            {
                pt_data[k][j*vd + l] = ps->domain(j, geom_dim + offset_idx);
                offset_idx++;
            }
        }    
    }

    // science variable settings
    int* vardims        = new int[nvars];
    char** varnames     = new char*[nvars];
    int* centerings     = new int[nvars];
    for (int k = 0; k < nvars; k++)
    {
        vardims[k]      = ps->var_dim(k);
        varnames[k]     = new char[256];
        centerings[k]   = 1;
        sprintf(varnames[k], "var%d", k);
    }

    // write raw original points
    if (ps->is_structured())
    {
        write_curvilinear_mesh(
            /* const char *filename */                  filename,
            /* int useBinary */                         0,
            /* int *dims */                             &npts_dim[0],
            /* float *pts */                            &(pt_coords[0].x),
            /* int nvars */                             nvars,
            /* int *vardim */                           vardims,
            /* int *centering */                        centerings,
            /* const char * const *varnames */          varnames,
            /* float **vars */                          pt_data);
    }
    else
    {
        write_point_mesh(
        /* const char *filename */                      filename,
        /* int useBinary */                             0,
        /* int npts */                                  pt_coords.size(),
        /* float *pts */                                &(pt_coords[0].x),
        /* int nvars */                                 nvars,
        /* int *vardim */                               vardims,
        /* const char * const *varnames */              varnames,
        /* float **vars */                              pt_data);
    }  

    delete[] vardims;
    for (int i = 0; i < nvars; i++)
        delete[] varnames[i];
    delete[] varnames;
    delete[] centerings;
    for (int j = 0; j < nvars; j++)
    {
        delete[] pt_data[j];
    }
    delete[] pt_data;
}

// make combinations of min, max corner vertices in index and real space
void CellVertices(
        int             ndom_dims,                      // number of domain dimensions
        vec3d&          min,                            // min corner
        vec3d&          max,                            // max corner
        vector<vec3d>&  tensor_pts)                     // (output) vertices
{
    vec3d p;

    p.x = min.x;
    p.y = min.y;
    p.z = min.z;
    tensor_pts.push_back(p);

    p.x = max.x;
    tensor_pts.push_back(p);

    if (ndom_dims > 1)
    {
        p.y = max.y;
        tensor_pts.push_back(p);

        p.x = min.x;
        tensor_pts.push_back(p);

        if (ndom_dims > 2)
        {
            p.x = min.x;
            p.y = min.y;
            p.z = max.z;
            tensor_pts.push_back(p);

            p.x = max.x;
            tensor_pts.push_back(p);

            p.y = max.y;
            tensor_pts.push_back(p);

            p.x = min.x;
            tensor_pts.push_back(p);
        }
    }
}

// prep tmesh tensor extents
void PrepTmeshTensorExtents(
        int             nvars,                              // number of variables
        int             ndom_dims,                          // number of domain dimensions
        vector<vec3d>&  tensor_pts_real,                    // (output) points in real space
        vector<vec3d>&  tensor_pts_index,                   // (output) points in index space
        B*  block)                              // curent block
{
    for (auto j = 0; j < nvars; j++)
    {
        vec3d min_real, max_real, min_index, max_index;         // extents in real and index space

        const mfa::Tmesh<real_t>& tmesh = block->mfa->var(j).tmesh;

        // form extents in index and real space
        for (auto k = 0; k < tmesh.tensor_prods.size(); k++)
        {
            min_index.x = tmesh.tensor_prods[k].knot_mins[0];
            min_real.x  = block->core_mins[0] + tmesh.all_knots[0][tmesh.tensor_prods[k].knot_mins[0]] *
                (block->core_maxs[0] - block->core_mins[0]);
            if (ndom_dims > 1)
            {
                min_index.y = tmesh.tensor_prods[k].knot_mins[1];
                min_real.y  = block->core_mins[1] + tmesh.all_knots[1][tmesh.tensor_prods[k].knot_mins[1]] *
                    (block->core_maxs[1] - block->core_mins[1]);
            }
            else
            {
                min_index.y = 0.0;
                min_real.y  = 0.0;
            }
            if (ndom_dims > 2)
            {
                min_index.z = tmesh.tensor_prods[k].knot_mins[2];
                min_real.z  = block->core_mins[2] + tmesh.all_knots[2][tmesh.tensor_prods[k].knot_mins[2]] *
                    (block->core_maxs[2] - block->core_mins[2]);
            }
            else
            {
                min_index.z = 0.0;
                min_real.z  = 0.0;
            }

            max_index.x = tmesh.tensor_prods[k].knot_maxs[0];
            max_real.x  = block->core_mins[0] + tmesh.all_knots[0][tmesh.tensor_prods[k].knot_maxs[0]] *
                (block->core_maxs[0] - block->core_mins[0]);
            if (ndom_dims > 1)
            {
                max_index.y = tmesh.tensor_prods[k].knot_maxs[1];
                max_real.y  = block->core_mins[1] + tmesh.all_knots[1][tmesh.tensor_prods[k].knot_maxs[1]] *
                    (block->core_maxs[1] - block->core_mins[1]);
            }
            else
            {
                max_index.y = 0.0;
                max_real.y  = 0.0;
            }
            if (ndom_dims > 2)
            {
                max_index.z = tmesh.tensor_prods[k].knot_maxs[2];
                max_real.z  = block->core_mins[2] + tmesh.all_knots[2][tmesh.tensor_prods[k].knot_maxs[2]] *
                    (block->core_maxs[2] - block->core_mins[2]);
            }
            else
            {
                max_index.z = 0.0;
                max_real.z  = 0.0;
            }

            // make vertex points for cells
            CellVertices(ndom_dims, min_index, max_index, tensor_pts_index);
            CellVertices(ndom_dims, min_real, max_real, tensor_pts_real);
        }   // tensor products
    }   // nvars
}

// package rendering data
void PrepRenderingData(
        int&                        nvars,
        vector<vec3d>&              geom_ctrl_pts,
        vector< vector <vec3d> >&   vars_ctrl_pts,
        float**&                    vars_ctrl_data,
        vector<vec3d>&              tensor_pts_real,
        vector<vec3d>&              tensor_pts_index,
        vector<int>&                ntensor_pts,
        B*                          block,
        int                         sci_var,                // science variable to render geometrically for 1d and 2d domains
        int&                        pt_dim)                 // (output) dimensionality of point
{
    vec3d p;

    // number of geometry dimensions and science variables
    int dom_dim     = block->mfa->dom_dim;
    nvars           = block->mfa->nvars();                    // number of science variables
    pt_dim          = block->mfa->pt_dim;                     // dimensionality of point

    // --- geometry control points ---

    // compute vectors of individual control point coordinates for the tensor product
    vector<vector<float>> ctrl_pts_coords(dom_dim);
    for (auto t = 0; t < block->mfa->geom().tmesh.tensor_prods.size(); t++)                      // tensor products
    {
        const TensorProduct<real_t>& tc = block->mfa->geom().tmesh.tensor_prods[t];
        mfa::VolIterator vol_iter(tc.nctrl_pts);
        for (auto k = 0; k < dom_dim; k++)                                                            // domain dimensions
        {
            int skip = 0;
            // starting knot in sequence for computing control point coordinate
            KnotIdx knot_min = tc.knot_mins[k];
            if (knot_min)
            {
                // skip knots at a deeper level than the tensor
                for (auto l = 0; l < block->mfa->geom().p(k); l++)
                {
                    while (block->mfa->geom().tmesh.all_knot_levels[k][knot_min - l - skip] > tc.level)
                        skip++;
                }
                knot_min -= (block->mfa->geom().p(k) - 1 + skip);
            }

            for (auto j = 0; j < tc.nctrl_pts(k); j++)                      // control points
            {
                float tsum  = 0.0;
                int skip1   = skip;                                         // number of knots at a deeper level that should be skipped
                // skip knots at a deeper level than the tensor
                for (int l = 1; l < block->mfa->geom().p(k) + 1; l++)
                {
                    // skip knots at a deeper level than the tensor
                    while (block->mfa->geom().tmesh.all_knot_levels[k][knot_min + j + l + skip1] > tc.level)
                        skip1++;
                    tsum += block->mfa->geom().tmesh.all_knots[k][knot_min + j + l + skip1];
                }
                tsum /= float(block->mfa->geom().p(k));
                ctrl_pts_coords[k].push_back(block->core_mins(k) + tsum * (block->core_maxs(k) - block->core_mins(k)));

                // debug
//                 fprintf(stderr, "t=%d k=%d j=%d tsum=%.3lf ctrl_pts_coord=%.3lf\n", t, k, j, tsum, ctrl_pts_coords[k].back());
            }   // control points
        }   // domain dimensions
    }   // tensor products

    // form the tensor product of control points from the vectors of individual coordinates
    VectorXi ofst = VectorXi::Zero(3);                              // offset of indices for current tensor
    for (auto t = 0; t < block->mfa->geom().tmesh.tensor_prods.size(); t++)                      // tensor products
    {
        const TensorProduct<real_t>& tc   = block->mfa->geom().tmesh.tensor_prods[t];
        mfa::VolIterator vol_iter(tc.nctrl_pts);
        VectorXi ijk(dom_dim);
        while (!vol_iter.done())                                    // control points
        {
            vol_iter.idx_ijk(vol_iter.cur_iter(), ijk);

            if (tc.weights(vol_iter.cur_iter()) == MFA_NAW)
            {
                vol_iter.incr_iter();
                continue;
            }

            // first 3 dims stored as mesh geometry
            p.x = ctrl_pts_coords[0][ofst(0) + ijk(0)];
            if (dom_dim < 2)
                p.y = 0.0;
            else
                p.y = ctrl_pts_coords[1][ofst(1) + ijk(1)];
            if (dom_dim < 3)
                p.z = 0.0;
            else
                p.z = ctrl_pts_coords[2][ofst(2) + ijk(2)];
            geom_ctrl_pts.push_back(p);

            // debug
//             fprintf(stderr, "t = %d geom_ctrl_pt = [%.3lf %.3lf]\n", t, geom_ctrl_pts.back().x, geom_ctrl_pts.back().y);

            vol_iter.incr_iter();
        }       // control points
        ofst.head(dom_dim) += tc.nctrl_pts;
    }       // tensor products

    // --- science variable control points ---

    vars_ctrl_pts.resize(nvars);
    vars_ctrl_data = new float*[nvars];
    for (size_t i = 0; i < nvars; i++)                              // science variables
    {
        size_t nctrl_pts = 0;
        for (auto t = 0; t < block->mfa->var(i).tmesh.tensor_prods.size(); t++)                   // tensor products
        {
            size_t prod = 1;
            for (auto k = 0; k < dom_dim; k++)                                                        // domain dimensions
                prod *= block->mfa->var(i).tmesh.tensor_prods[t].nctrl_pts(k);
            nctrl_pts += prod;
        }
        vars_ctrl_data[i] = new float[nctrl_pts];

        // compute vectors of individual control point coordinates for the tensor product
        vector<vector<float>> ctrl_pts_coords(dom_dim);
        for (auto t = 0; t < block->mfa->var(i).tmesh.tensor_prods.size(); t++)                   // tensor products
        {
            const TensorProduct<real_t>& tc = block->mfa->var(i).tmesh.tensor_prods[t];
            mfa::VolIterator vol_iter(tc.nctrl_pts);
            for (auto k = 0; k < dom_dim; k++)                                                        // domain dimensions
            {
                int skip = 0;
                // starting knot in sequence for computing control point coordinate
                KnotIdx knot_min = tc.knot_mins[k];
                if (knot_min)
                {
                    // skip knots at a deeper level than the tensor
                    for (auto l = 0; l < block->mfa->var(i).p(k); l++)
                    {
                        while (block->mfa->var(i).tmesh.all_knot_levels[k][knot_min - l - skip] > tc.level)
                            skip++;
                    }
                    knot_min -= (block->mfa->var(i).p(k) - 1 + skip);
                }

                int skip1   = skip;                                 // number of knots at a deeper level that should be skipped
                for (auto j = 0; j < tc.nctrl_pts(k); j++)              // control points
                {
                    float tsum  = 0.0;
                    for (auto l = 1; l < block->mfa->var(i).p(k) + 1; l++)
                    {
                        // skip knots at a deeper level than the tensor
                        while (block->mfa->var(i).tmesh.all_knot_levels[k][knot_min + j + l + skip1] > tc.level)
                            skip1++;
                        tsum += block->mfa->var(i).tmesh.all_knots[k][knot_min + j + l + skip1];
                    }
                    tsum /= float(block->mfa->var(i).p(k));
                    ctrl_pts_coords[k].push_back(block->core_mins(k) + tsum * (block->core_maxs(k) - block->core_mins(k)));

                    // debug
//                     fprintf(stderr, "t=%d k=%d j=%d tsum=%.3lf ctrl_pts_coord=%.3lf\n", t, k, j, tsum, ctrl_pts_coords[k].back());
                }   // control points
            }   // domain dimensions
        }   // tensor products

        // form the tensor product of control points from the vectors of individual coordinates
        VectorXi ofst = VectorXi::Zero(3);                              // offset of indices for current tensor
        for (auto t = 0; t < block->mfa->var(i).tmesh.tensor_prods.size(); t++)                  // tensor products
        {
            const TensorProduct<real_t>& tc   = block->mfa->var(i).tmesh.tensor_prods[t];
            mfa::VolIterator vol_iter(tc.nctrl_pts);
            VectorXi ijk(dom_dim);
            while (!vol_iter.done())                                        // control points
            {
                vol_iter.idx_ijk(vol_iter.cur_iter(), ijk);

                if (tc.weights(vol_iter.cur_iter()) == MFA_NAW)
                {
                    vol_iter.incr_iter();
                    continue;
                }

                // first 3 dims stored as mesh geometry
                // control point position and optionally science variable, if the total fits in 3d
                p.x = ctrl_pts_coords[0][ofst(0) + ijk(0)];
                if (dom_dim < 2)
                {
                    p.y = tc.ctrl_pts(vol_iter.cur_iter(), 0);
                    p.z = 0.0;
                }
                else
                {
                    p.y = ctrl_pts_coords[1][ofst(1) + ijk(1)];
                    if (dom_dim < 3)
                        p.z = tc.ctrl_pts(vol_iter.cur_iter(), 0);
                    else
                        p.z = ctrl_pts_coords[2][ofst(2) + ijk(2)];
                }
                vars_ctrl_pts[i].push_back(p);

                // science variable also stored as data
                vars_ctrl_data[i][vars_ctrl_pts[i].size() - 1] = tc.ctrl_pts(vol_iter.cur_iter(), 0);

                // debug
//                 fprintf(stderr, "t=%d ctrl_pt= [%.3lf %.3lf %.3lf]\n", t, vars_ctrl_pts[i].back().x, vars_ctrl_pts[i].back().y, vars_ctrl_data[i][vars_ctrl_pts[i].size() - 1]);

                vol_iter.incr_iter();
            }   // control points
            ofst.head(dom_dim) += tc.nctrl_pts;
        }   // tensor products
    }   // science variables

    // tmesh tensor extents
    ntensor_pts.resize(3);
    for (auto i = 0; i < 3; i++)
        ntensor_pts[i] = (i >= dom_dim ? 1 : 2);
    PrepTmeshTensorExtents(nvars, dom_dim, tensor_pts_real, tensor_pts_index, block);
}

// write vtk files for initial, approximated, control points
void write_vtk_files(
        B* b,
        const          diy::Master::ProxyWithLink& cp,
        int            sci_var,                     // science variable to render geometrically for 1d and 2d domains
        int&           dom_dim,                     // (output) domain dimensionality
        int&           pt_dim)                      // (output) point dimensionality
{
    int                         nvars;              // number of science variables (excluding geometry)
    vector<vec3d>               geom_ctrl_pts;      // control points (<= 3d) in geometry
    vector < vector <vec3d> >   vars_ctrl_pts;      // control points (<= 3d) in science variables
    float**                     vars_ctrl_data;     // control point data values (4d)
    vector<vec3d>               tensor_pts_real;    // tmesh tensor product extents in real space
    vector<vec3d>               tensor_pts_index;   // tmesh tensor product extents in index space
    vector<int>                 ntensor_pts;        // number of tensor extent points in each dim

    // Write PointSets
    char input_filename[256];
    char approx_filename[256];
    char errs_filename[256];
    char blend_filename[256];
    sprintf(input_filename, "initial_points_gid_%d.vtk", cp.gid());
    sprintf(approx_filename, "approx_points_gid_%d.vtk", cp.gid());
    sprintf(errs_filename, "error_gid_%d.vtk", cp.gid());
    sprintf(blend_filename, "blend_gid_%d.vtk", cp.gid());
    write_pointset_vtk(b->input, input_filename, sci_var);
    write_pointset_vtk(b->approx, approx_filename, sci_var);
    write_pointset_vtk(b->errs, errs_filename, sci_var);
    write_pointset_vtk(b->blend, blend_filename, sci_var);

    if (b->mfa != nullptr)
    {
        // package rendering data
        PrepRenderingData(nvars,
                        geom_ctrl_pts,
                        vars_ctrl_pts,
                        vars_ctrl_data,
                        tensor_pts_real,
                        tensor_pts_index,
                        ntensor_pts,
                        b,
                        sci_var,
                        pt_dim);

        // pad dimensions up to 3
        dom_dim = b->dom_dim;

        // science variable settings
        int vardim          = 1;
        int centering       = 1;
        int* vardims        = new int[nvars];
        char** varnames     = new char*[nvars];
        int* centerings     = new int[nvars];
        float* vars;
        for (int i = 0; i < nvars; i++)
        {
            vardims[i]      = 1;                                // TODO; treating each variable as a scalar (for now)
            varnames[i]     = new char[256];
            centerings[i]   = 1;
            sprintf(varnames[i], "var%d", i);
        }

        // write geometry control points
        char filename[256];
        sprintf(filename, "geom_control_points_gid_%d.vtk", cp.gid());
        if (geom_ctrl_pts.size())
            write_point_mesh(
                /* const char *filename */                      filename,
                /* int useBinary */                             0,
                /* int npts */                                  geom_ctrl_pts.size(),
                /* float *pts */                                &(geom_ctrl_pts[0].x),
                /* int nvars */                                 0,
                /* int *vardim */                               NULL,
                /* const char * const *varnames */              NULL,
                /* float **vars */                              NULL);

        // write science variables control points
        for (auto i = 0; i < nvars; i++)
        {
            sprintf(filename, "var%d_control_points_gid_%d.vtk", i, cp.gid());
            if (vars_ctrl_pts[i].size())
                write_point_mesh(
                /* const char *filename */                      filename,
                /* int useBinary */                             0,
                /* int npts */                                  vars_ctrl_pts[i].size(),
                /* float *pts */                                &(vars_ctrl_pts[i][0].x),
                /* int nvars */                                 nvars,
                /* int *vardim */                               vardims,
                /* const char * const *varnames */              varnames,
                /* float **vars */                              vars_ctrl_data);
        }

        // write tensor product extents
        int pts_per_cell = pow(2, dom_dim);
        int ncells = tensor_pts_real.size() / pts_per_cell;
        vector<int> cell_types(ncells);
        for (auto i = 0; i < cell_types.size(); i++)
        {
            if (dom_dim == 1)
                cell_types[i] = VISIT_LINE;
            else if (dom_dim == 2)
                cell_types[i] = VISIT_QUAD;
            else
                cell_types[i] = VISIT_HEXAHEDRON;
        }
        vector<float> tensor_data(tensor_pts_real.size(), 1.0); // tensor data set to fake value
        vector<int> conn(tensor_pts_real.size());               // connectivity
        for (auto i = 0; i < conn.size(); i++)
            conn[i] = i;
        vars = &tensor_data[0];
        sprintf(filename, "tensor_real_gid_%d.vtk", cp.gid());
        const char* name_tensor ="tensor0";

        // in real space
        write_unstructured_mesh(
                /* const char *filename */                      filename,
                /* int useBinary */                             0,
                /* int npts */                                  tensor_pts_real.size(),
                /* float *pts */                                &(tensor_pts_real[0].x),
                /* int ncells */                                ncells,
                /* int *celltypes */                            &cell_types[0],
                /* int *conn */                                 &conn[0],
                /* int nvars */                                 1,
                /* int *vardim */                               &vardim,
                /* int *centering */                            &centering,
                /* const char * const *varnames */              &name_tensor,
                /* float **vars */                              &vars);

        // in index space
        sprintf(filename, "tensor_index_gid_%d.vtk", cp.gid());
        write_unstructured_mesh(
                /* const char *filename */                      filename,
                /* int useBinary */                             0,
                /* int npts */                                  tensor_pts_index.size(),
                /* float *pts */                                &(tensor_pts_index[0].x),
                /* int ncells */                                ncells,
                /* int *celltypes */                            &cell_types[0],
                /* int *conn */                                 &conn[0],
                /* int nvars */                                 1,
                /* int *vardim */                               &vardim,
                /* int *centering */                            &centering,
                /* const char * const *varnames */              &name_tensor,
                /* float **vars */                              &vars);

        delete[] vardims;
        for (int i = 0; i < nvars; i++)
            delete[] varnames[i];
        delete[] varnames;
        delete[] centerings;
        for (int j = 0; j < nvars; j++)
        {
            delete[] vars_ctrl_data[j];
        }
        delete[] vars_ctrl_data;
    }
}

int main(int argc, char ** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);       // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;                 // equivalent of MPI_COMM_WORLD

    string                      input  = "sine";        // input dataset
    int                         ntest  = 0;             // number of input test points in each dim for analytical error tests
    string                      infile = "approx.mfa";  // diy input file
    bool                        help;                   // show help
    int                         dom_dim, pt_dim;        // domain and point dimensionality, respectively
    int                         sci_var = 0;            // science variable to render geometrically for 1d and 2d domains

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('f', "infile",      infile,     " diy input file name");
    ops >> opts::Option('a', "ntest",       ntest,      " number of test points in each dimension of domain (for analytical error calculation)");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('v', "var",         sci_var,    " science variable to render geometrically for 1d and 2d domains");
    ops >> opts::Option('h', "help",        help,       " show help");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr << "infile = " << infile << " test_points = "    << ntest <<        endl;
    if (ntest)
        cerr << "input = "          << input     << endl;
#ifdef MFA_TBB
    cerr << "threading: TBB" << endl;
#endif
#ifdef MFA_KOKKOS
    cerr << "threading: Kokkos" << endl;
#endif
#ifdef MFA_SYCL
    cerr << "threading: SYCL" << endl;
#endif
#ifdef MFA_SERIAL
    cerr << "threading: serial" << endl;
#endif
    fprintf(stderr, "-------------------------------------\n\n");

    // initialize DIY
    diy::FileStorage storage("./DIY.XXXXXX");     // used for blocks to be moved out of core
    diy::Master      master(world,
            1,
            -1,
            &B::create,
            &B::destroy);
    diy::ContiguousAssigner   assigner(world.size(), -1); // number of blocks set by read_blocks()

    diy::io::read_blocks(infile.c_str(), world, assigner, master, &B::load);
    std::cout << master.size() << " blocks read from file "<< infile << "\n\n";

    // write vtk files for initial and approximated points
    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
            { write_vtk_files(b, cp, sci_var, dom_dim, pt_dim); });
}
