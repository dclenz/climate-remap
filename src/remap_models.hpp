//--------------------------------------------------------------
// Forward and reverse MFA model operations for climate remapping
//--------------------------------------------------------------
#ifndef MFA_REMAP_REMAP_MODELS_HPP
#define MFA_REMAP_REMAP_MODELS_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <mfa/mfa.hpp>

// Creates a knot vector that follows the distribution of vertical levels in the ocean mesh.
// The knots are roughly distributed across the depth levels, so the distribution of knots is
// consistent with the distribution of levels, even when the number of knots is not equal to
// the number of levels.
// Does not handle repeated end knots.
template <typename T>
std::vector<T> interpolateInteriorKnots(const std::vector<T>& levelParams,
                                        int numInteriorKnots)
{
    std::vector<T> knots(numInteriorKnots);
    for (int i = 0; i < numInteriorKnots; i++)
    {
        T position = static_cast<T>(i + 1) * (levelParams.size() - 1) /
                     static_cast<T>(numInteriorKnots + 1);
        size_t lower = static_cast<size_t>(std::floor(position));
        size_t upper = std::min(lower + 1, levelParams.size() - 1);
        T fraction = position - lower;
        knots[i] = levelParams[lower] + fraction * (levelParams[upper] - levelParams[lower]);
    }
    return knots;
}

template <typename T>
class RemapModels
{
  public:
    // Sets the final set of knot vectors for the forward MFA. Computes the nonuniform vertical knot
    // vector, adds pinned knots, and then updates the forward MFA.
    void applyForwardVerticalKnots(mfa::MFA<T>& forwardMFA,
                                   const std::vector<T>& sourceLevelMeanHeights,
                                   T minHeight, T maxHeight, int domDim) const
    {
        if (sourceLevelMeanHeights.size() < 2)
        {
            fmt::print(stderr, "ERROR: Need at least two MPAS vertical levels to set adaptive knots.\n");
            exit(1);
        }

        const int verticalDim = 2;
        const int degree = forwardMFA.var(0).p(verticalDim);
        const int nctrl = forwardMFA.var(0).tmesh.tensor_prods[0].nctrl_pts(verticalDim);
        const int numInteriorKnots = nctrl - degree - 1;
        if (numInteriorKnots < 0)
        {
            fmt::print(stderr, "ERROR: Invalid vertical model: nctrl={} degree={}.\n", nctrl, degree);
            exit(1);
        }

        // Prepare list of depth levels in the source mesh (normalized to [0,1])
        std::vector<T> levelParams;
        levelParams.reserve(sourceLevelMeanHeights.size());
        for (T height : sourceLevelMeanHeights)
        {
            T param = (height - minHeight) / (maxHeight - minHeight);
            levelParams.push_back(std::max(static_cast<T>(0), std::min(static_cast<T>(1), param)));
        }
        std::sort(levelParams.begin(), levelParams.end());

        // Compute interior knots
        std::vector<T> zknots = interpolateInteriorKnots(levelParams, numInteriorKnots);

        // Update the forward MFA with the new vertical knot vector
        std::vector<std::vector<T>> allknots(domDim);
        for (int dim = 0; dim < domDim; dim++)
            allknots[dim] = forwardMFA.var(0).tmesh.all_knots[dim];
        allknots[verticalDim] = forwardMFA.pinKnots(zknots, degree);
        forwardMFA.setKnots(allknots);

        // Debug info
        fmt::print(stderr, "MPAS mean vertical level parameters: {}\n", mfa::print_vec(levelParams));
        fmt::print(stderr, "Adaptive vertical interior knots: {}\n", mfa::print_vec(zknots));
        fmt::print(stderr, "Vertical knot vector: {}\n", mfa::print_vec(allknots[verticalDim]));
    }

    std::unique_ptr<mfa::PointSet<T>> applyForward(
        mfa::MFA<T>& forwardMFA, mfa::PointSet<T>& source, mfa::PointSet<T>& target,
        const mfa::MFAInfo& info, int domDim, int geomDim, bool dumpMatrices,
        const std::vector<T>& sourceLevelMeanHeights, T minHeight, T maxHeight) const
    {
        if (domDim == 3)
            applyForwardVerticalKnots(forwardMFA, sourceLevelMeanHeights, minHeight, maxHeight, domDim);

        auto forwardApprox = std::make_unique<mfa::PointSet<T>>(target.params, target.model_dims());
        if (dumpMatrices)
        {
            forwardMFA.dumpCollocationMatrixEncode(0, &source);
            forwardMFA.dumpCollocationMatrixDecode(0, forwardApprox.get());
        }

        const double encodeStart = MPI_Wtime();
        forwardMFA.FixedEncode(source, info.regularization, info.reg1and2, info.weighted);
        fmt::print(stderr, "Forward MFA encode time: {:.3f} s\n", MPI_Wtime() - encodeStart);

        forwardApprox->domain.leftCols(geomDim) = target.domain.leftCols(geomDim);
        const double decodeStart = MPI_Wtime();
        for (int varIdx = 0; varIdx < forwardMFA.nvars(); varIdx++)
            forwardMFA.DecodeVar(varIdx, *forwardApprox);
        fmt::print(stderr, "Forward MFA science decode time: {:.3f} s\n", MPI_Wtime() - decodeStart);
        fmt::print(stderr, "Approx npts: {}\n", forwardApprox->npts);
        fmt::print(stderr, "Approx model dims: {}\n", mfa::print_vec(forwardApprox->model_dims()));
        fmt::print(stderr, "Approx var 1 min/max: {}/{}\n", forwardApprox->domain.col(geomDim).minCoeff(), forwardApprox->domain.col(geomDim).maxCoeff());
        fmt::print(stderr, "Approx var 1 has nans: {}\n", forwardApprox->domain.col(geomDim).hasNaN());
        return forwardApprox;
    }

    void applyReverse(const mfa::MFAInfo& info, const mfa::PointSet<T>& forwardApprox,
                      const mfa::PointSet<T>& source, const VectorX<T>& targetHeights,
                      int numTargetElements, int profileLength, T minHeight, T maxHeight,
                      int domDim, int geomDim, const std::vector<std::string>& varNames)
    {
        reverse_mfa = std::make_unique<mfa::MFA<T>>(info);
        applyReverseVerticalKnots(targetHeights, numTargetElements, profileLength,
                                  minHeight, maxHeight, domDim);

        const double encodeStart = MPI_Wtime();
        reverse_mfa->FixedEncode(forwardApprox, info.regularization, info.reg1and2, info.weighted);
        fmt::print(stderr, "Reverse MFA encode time: {:.3f} s\n", MPI_Wtime() - encodeStart);

        roundtrip_approx = std::make_unique<mfa::PointSet<T>>(source.params, source.model_dims());
        roundtrip_approx->domain.leftCols(geomDim) = source.domain.leftCols(geomDim);
        const double decodeStart = MPI_Wtime();
        for (int varIdx = 0; varIdx < reverse_mfa->nvars(); varIdx++)
            reverse_mfa->DecodeVar(varIdx, *roundtrip_approx);
        fmt::print(stderr, "Reverse MFA science decode time: {:.3f} s\n", MPI_Wtime() - decodeStart);

        if (roundtrip_approx->npts != source.npts)
        {
            fmt::print(stderr, "ERROR: Reverse decode produced {} points; expected {} MPAS points.\n",
                       roundtrip_approx->npts, source.npts);
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

    mfa::MFA<T>* reverseMFA() const { return reverse_mfa.get(); }
    mfa::PointSet<T>* roundtripApproximation() const { return roundtrip_approx.get(); }

    std::unique_ptr<mfa::MFA<T>> exchangeReverseMFA(std::unique_ptr<mfa::MFA<T>> model)
    {
        reverse_mfa.swap(model);
        return model;
    }

    std::unique_ptr<mfa::PointSet<T>> exchangeRoundtripApproximation(
        std::unique_ptr<mfa::PointSet<T>> approximation)
    {
        roundtrip_approx.swap(approximation);
        return approximation;
    }

    static void printModel(const mfa::MFA<T>* model, int gid)
    {
        if (!model)
        {
            fmt::print(stderr, "gid = {}: No MFA found.\n", gid);
            return;
        }

        fmt::print(stderr, "gid = {}\n", gid);
        fmt::print(stderr, "---------------- geometry model ----------------\n");
        printKnotsAndControlPoints(model->geom());
        fmt::print(stderr, "------------------------------------------------\n");
        fmt::print(stderr, "\n----------- science variable models ------------\n");
        for (int i = 0; i < model->nvars(); i++)
        {
            fmt::print(stderr, "-------------------- var {} --------------------\n", i);
            printKnotsAndControlPoints(model->var(i));
            fmt::print(stderr, "------------------------------------------------\n");
        }
    }

  private:
    void applyReverseVerticalKnots(const VectorX<T>& targetHeights, int numTargetElements,
                                   int profileLength, T minHeight, T maxHeight, int domDim)
    {
        if (profileLength < 2)
        {
            fmt::print(stderr, "ERROR: Need at least two ROMS levels to set reverse vertical knots.\n");
            exit(1);
        }
        if (targetHeights.size() != numTargetElements * profileLength)
        {
            fmt::print(stderr, "ERROR: ROMS height count does not match target profile layout.\n");
            exit(1);
        }

        std::vector<T> levelParams(profileLength, 0);
        for (int level = 0; level < profileLength; level++)
        {
            T sum = 0;
            for (int element = 0; element < numTargetElements; element++)
                sum += targetHeights(element * profileLength + level);
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

        std::vector<T> zknots = interpolateInteriorKnots(levelParams, numInteriorKnots);
        std::vector<std::vector<T>> allknots(domDim);
        for (int dim = 0; dim < domDim; dim++)
            allknots[dim] = reverse_mfa->var(0).tmesh.all_knots[dim];
        if (!zknots.empty())
            allknots[verticalDim] = reverse_mfa->pinKnots(zknots, degree);

        fmt::print(stderr, "Reverse ROMS vertical interior knots: {}\n", mfa::print_vec(zknots));
        reverse_mfa->setKnots(allknots);
    }

    static void printKnotsAndControlPoints(const mfa::MFA_Data<T>& model)
    {
        VectorXi totalControlPointsByDimension = VectorXi::Zero(model.dom_dim);
        size_t totalControlPoints = 0;
        for (auto tensor = 0; tensor < model.ntensors(); tensor++)
        {
            totalControlPointsByDimension += model.tmesh.tensor_prods[tensor].nctrl_pts;
            totalControlPoints += model.tmesh.tensor_prods[tensor].nctrl_pts.prod();
        }
        if (model.ntensors() == 1)
            fmt::print(stderr, "# output ctrl pts     = [ {} ]\n", totalControlPointsByDimension.transpose());
        fmt::print(stderr, "tot # output ctrl pts = {}\n", totalControlPoints);

        fmt::print(stderr, "# output knots        = [ ");
        for (auto dim = 0; dim < model.tmesh.all_knots.size(); dim++)
            fmt::print(stderr, "{} ", model.tmesh.all_knots[dim].size());
        fmt::print(stderr, "]\n");
    }

    std::unique_ptr<mfa::MFA<T>> reverse_mfa;
    std::unique_ptr<mfa::PointSet<T>> roundtrip_approx;
};

#endif // MFA_REMAP_REMAP_MODELS_HPP
