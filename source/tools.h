
#ifndef TOOLS
#define TOOLS

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iostream>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <chrono>
#include <ctime>
#include <math.h>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/error_estimator.h>

using namespace dealii;

namespace tools
{

  double coth(double x)
  {
    return (1. + std::exp(-2. * x)) / (1. - std::exp(-2. * x));
  }

  Tensor<1, 2> curl_from_grad(Tensor<1, 2> grad_X)
  {
    Tensor<1, 2> curl;
    curl[0] = grad_X[1];
    curl[1] = -grad_X[0];
    return curl;
  }

  Tensor<1, 3> curl_from_grad(Tensor<1, 3> grad_X)
  {
    Tensor<1, 3> curl;
    curl[0] = grad_X[1];
    curl[1] = -grad_X[0];
    return curl;
  }

  inline double kd(int i, int j) //kronecker delta
  {
    return (i == j ? 1.0 : 0.0);
  }

  inline double kd(int i, int j, int k) //kronecker delta
  {
    return ((i == j && i == k) ? 1.0 : 0.0);
  }
}

#endif //TOOLS