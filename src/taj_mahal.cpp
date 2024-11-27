#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <eigen3/Eigen/Dense>
#include <boost/math/distributions/chi_squared.hpp>
#include <vector>
#include <cmath>

namespace py = pybind11;
using namespace Eigen;


/**
 * @brief Calculates the Mahalanobis distance of one point
 *
 * @param x The point for which distance is calculated
 * @param mu The mean vector of the dataset (column-wise mean)
 * @param S_inv the inverse covariance matrix
 *
 * @return The Mahalanobis distance of the point
 */
double mahalanobis_distance(const VectorXd& x, const VectorXd& mu, const MatrixXd& S_inv) {
    VectorXd diff = x - mu;
    return std::sqrt(diff.transpose() * S_inv * diff);
}

/**
 * @brief Calculates all mahalanobis distances for a dataset
 *
 * @param points A Numpy matrix representing a dataset
 *
 * @return A 1D Numpy array containing all the distances
 */
py::array mahalanobis_distances(py::array_t<double> points) {
    py::buffer_info points_info = points.request();

    size_t num_points = points_info.shape[0];
    size_t num_features = points_info.shape[1];

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> points_matrix(
        static_cast<double*>(points_info.ptr), num_points, num_features);

    VectorXd mu = points_matrix.colwise().mean();

    MatrixXd centered = points_matrix.rowwise() - mu.transpose();
    MatrixXd cov_matrix = (centered.transpose() * centered) / double(num_points - 1);

    MatrixXd S_inv;

    if(cov_matrix.determinant() == 0) {
        S_inv = cov_matrix.completeOrthogonalDecomposition().pseudoInverse();
    } else {
        S_inv = cov_matrix.inverse();
    }

    std::vector<double> distances;

    for (size_t i = 0; i < num_points; ++i) {
        Map<VectorXd> point(static_cast<double*>(points_info.ptr) + i * num_features, num_features);
        double dist = mahalanobis_distance(point, mu, S_inv);
        distances.push_back(dist);
    }

    return py::array(distances.size(), distances.data());
}

/**
 * @brief Detects the outliers in the dataset using the Mahalanobis distance
 *
 * @param points A Numpy matrix representing a dataset
 * @param alpha The significance level of the threshold; default 0.01
 * @param indices Whether or not to return the indices instead of the values; default false
 *
 * @return Array of outliers in the dataset
 */
py::array detect_outliers(py::array_t<double> points, double alpha = 0.01, bool indices = false) {
    py::buffer_info points_info = points.request();
    size_t num_points = points_info.shape[0];
    size_t num_features = points_info.shape[1];

    py::array distances = mahalanobis_distances(points);
    py::buffer_info dist_info = distances.request();
    auto* distances_ptr = static_cast<double*>(dist_info.ptr);

    boost::math::chi_squared chi2_dist(num_features);
    double threshold = std::sqrt(boost::math::quantile(chi2_dist, 1 - alpha));

    std::vector<size_t> outliers_idx;
    for (size_t i = 0; i < num_points; i++) {
        if (distances_ptr[i] > threshold) outliers_idx.push_back(i);
    }

    if (indices) {
        py::array_t<size_t> indices_array(outliers_idx.size());
        py::buffer_info indices_info = indices_array.request();
        auto* indices_ptr = static_cast<size_t*>(indices_info.ptr);
        std::copy(outliers_idx.begin(), outliers_idx.end(), indices_ptr);
        return indices_array;
    }

    size_t num_outliers = outliers_idx.size();
    py::array_t<double> outliers({num_outliers, num_features});
    py::buffer_info outliers_info = outliers.request();
    auto* outliers_ptr = static_cast<double*>(outliers_info.ptr);

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> points_matrix(
        static_cast<double*>(points_info.ptr), num_points, num_features);

    for (size_t i = 0; i < num_outliers; i++) {
        size_t idx = outliers_idx[i];
        for (size_t j = 0; j < num_features; j++) {
            outliers_ptr[i * num_features + j] = points_matrix(idx, j);
        }
    }

    return outliers;
}

PYBIND11_MODULE(taj_mahal, m) {
    m.def("distances", &mahalanobis_distances, "Calculates Mahalanobis distances for a numerical dataset",
        py::arg("points"));
    m.def("outliers", &detect_outliers, "Returns outlier values or indices",
        py::arg("points"),
        py::arg("alpha") = 0.01,
        py::arg("indices") = false);
}
