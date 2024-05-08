
#ifndef OUTPUT
#define OUTPUT

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

#include <deal.II/base/table_indices.h>

#include "problem_conditions.h"
#include "tools.h"

using namespace dealii;
using namespace tools;
using namespace pb_cond;
namespace output
{

  double corrector=1.;

  template <int dim>
  class GradientPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    GradientPostprocessor()
        : DataPostprocessorVector<dim>("grad_A", update_gradients) {}
    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);
        for (unsigned int d = 0; d < dim; ++d)
          computed_quantities[p][d] = input_data.solution_gradients[p][d];
      }
    }
  };

  template <int dim>
  class BPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    BPostprocessor(double b0_in)
        : DataPostprocessorVector<dim>("B", update_gradients),
          magnetic_field_loading(b0_in) {}

    double magnetic_field_loading;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);
        computed_quantities[p][0] = input_data.solution_gradients[p][1];
        computed_quantities[p][1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];
      }
    }
  };

  template <int dim>
  class BcylindricalPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    BcylindricalPostprocessor(double b0_in)
        : DataPostprocessorVector<dim>("B_cylind", update_gradients |
                                                       update_quadrature_points),
          magnetic_field_loading(b0_in) {}

    double magnetic_field_loading;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);

        Tensor<1, 2> magnetic_field;

        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        double theta;
        if (abs(input_data.evaluation_points[p][0]*corrector) > 1.e-9)
        {
          theta = std::atan(input_data.evaluation_points[p][1]*corrector /
                            abs(input_data.evaluation_points[p][0]*corrector));
          if (input_data.evaluation_points[p][0]*corrector < 0.)
          {
            theta = M_PI - theta;
          }
          double cos = std::cos(theta);
          double sin = std::sin(theta);

          computed_quantities[p][0] =
              cos * magnetic_field[0] + sin * magnetic_field[1];
          computed_quantities[p][1] =
              -sin * magnetic_field[0] + cos * magnetic_field[1];
        }
        else
        {
          if (input_data.evaluation_points[p][1]*corrector > 0)
          {
            computed_quantities[p][0] = magnetic_field[1];
            computed_quantities[p][1] = -magnetic_field[0];
          }
          else
          {
            computed_quantities[p][0] = -magnetic_field[1];
            computed_quantities[p][1] = magnetic_field[0];
          }
        }
      }
    }
  };

  template <int dim>
  class ForcePostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    ForcePostprocessor(double b0_in, double current_density_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorVector<dim>(
              "f", update_gradients | update_quadrature_points | update_hessians),
          magnetic_field_loading(b0_in), current_density(current_density_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    double current_density;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);

        Tensor<2, 3> Id_tensor;
        Id_tensor[0][0] = 1.;
        Id_tensor[1][1] = 1.;
        Id_tensor[2][2] = 1.;

        // we compute sigma_m and then takes the divergence

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build local magnetisation
        Tensor<1, 3> magnetisation_vector =
            magnetisation(input_data.evaluation_points[p][0]*corrector,
                          input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);

        // build current vector
        Tensor<1, 3> current_vector;
        current_vector[2] =
            j(input_data.evaluation_points[p][0]*corrector,
              input_data.evaluation_points[p][1]*corrector, current_density,geom,size,size2,exterior_size);

        Tensor<1, 3> force;

        // curl_from_hessian_of_A(magnetic_field,input_data.solution_hessians) is
        // curl(curl(A)+B_irot)=curl(curl(A)), as A=Az, curl(curl(A))[0]=0 ,
        // curl(curl(A))[1]=0, curl(curl(A))[2]=-Az,xx-Az,yy = -\Delta Az
        // div_from_hessian_of_A(magnetic_field,input_data.solution_hessians) is
        // div(curl(A)+B_irot)=div(B_sole),

        // builgin laplacien vector of A ()
        Tensor<1, 3> laplacien_vector_A;
        laplacien_vector_A[2] = input_data.solution_hessians[p][0][0] +
                                input_data.solution_hessians[p][1][1];

        // we also need div m:
        // reminder, div(scalar * vector)=scalar*div(vector) + grad(scalar).vector
        // m = - 2 phi,J1 * b
        // div m = -2 phi,J1 * div b - 2 grad(phi,J1) . b =  -2 grad(phi,J1) . b
        // as div(b)=0 grad(phi,J1)=phi,J1J1 * grad(J1) grad(J1) = 2 grad(b).b we
        // need grad(b) div m =-2 phi,J1J1 * 2 (grad(b).b). b
        Tensor<2, 3> grad_b;
        grad_b[0][0] = input_data.solution_hessians[p][1][0];
        grad_b[0][1] = input_data.solution_hessians[p][1][1];
        grad_b[1][0] = -input_data.solution_hessians[p][0][0];
        grad_b[1][1] = -input_data.solution_hessians[p][0][1];

        double chi_local = local_magnetic_suceptibility(
            input_data.evaluation_points[p][0]*corrector,
            input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);
        double J1 = magnetic_field * magnetic_field;

        force =
            1. / mu0 / (1. + chi_local) *
                cross_product_3d(-laplacien_vector_A, magnetic_field) 
                +
            magnetisation_vector * grad_b 
            +
            4. *
                phi_J1_J1(input_data.evaluation_points[p][0]*corrector,
                          input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size) *
                (magnetic_field * grad_b) *
                (outer_product(magnetic_field, magnetic_field) - J1 * Id_tensor);

        computed_quantities[p][0] = force[0];
        computed_quantities[p][1] = force[1];
      }
    }
  };

  template <int dim>
  class ForcelambdaPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    ForcelambdaPostprocessor(double b0_in, double current_density_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorVector<dim>(
              "f_with_lambda", update_gradients | update_quadrature_points | update_hessians),
          magnetic_field_loading(b0_in), current_density(current_density_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    double current_density;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);

        Tensor<2, 3> Id_tensor;
        Id_tensor[0][0] = 1.;
        Id_tensor[1][1] = 1.;
        Id_tensor[2][2] = 1.;

        // we compute sigma_m and then takes the divergence

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build local magnetisation
        Tensor<1, 3> magnetisation_vector =
            magnetisation(input_data.evaluation_points[p][0]*corrector,
                          input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);

        // build current vector
        Tensor<1, 3> current_vector;
        current_vector[2] =
            j(input_data.evaluation_points[p][0]*corrector,
              input_data.evaluation_points[p][1]*corrector, current_density,geom,size,size2,exterior_size);

        Tensor<1, 3> force;

        // curl_from_hessian_of_A(magnetic_field,input_data.solution_hessians) is
        // curl(curl(A)+B_irot)=curl(curl(A)), as A=Az, curl(curl(A))[0]=0 ,
        // curl(curl(A))[1]=0, curl(curl(A))[2]=-Az,xx-Az,yy = -\Delta Az
        // div_from_hessian_of_A(magnetic_field,input_data.solution_hessians) is
        // div(curl(A)+B_irot)=div(B_sole),

        // builgin laplacien vector of A ()
        Tensor<1, 3> laplacien_vector_A;
        laplacien_vector_A[2] = input_data.solution_hessians[p][0][0] +
                                input_data.solution_hessians[p][1][1];

        // we also need div m:
        // reminder, div(scalar * vector)=scalar*div(vector) + grad(scalar).vector
        // m = - 2 phi,J1 * b
        // div m = -2 phi,J1 * div b - 2 grad(phi,J1) . b =  -2 grad(phi,J1) . b
        // as div(b)=0 grad(phi,J1)=phi,J1J1 * grad(J1) grad(J1) = 2 grad(b).b we
        // need grad(b) div m =-2 phi,J1J1 * 2 (grad(b).b). b
        Tensor<2, 3> grad_b;
        grad_b[0][0] = input_data.solution_hessians[p][1][0];
        grad_b[0][1] = input_data.solution_hessians[p][1][1];
        grad_b[1][0] = -input_data.solution_hessians[p][0][0];
        grad_b[1][1] = -input_data.solution_hessians[p][0][1];

        Tensor<2, 3> grad_b_T;
        grad_b_T[0][0]=grad_b[0][0];
        grad_b_T[0][1]=grad_b[1][0];
        grad_b_T[1][0]=grad_b[0][1];
        grad_b_T[1][1]=grad_b[1][1];

        double chi_local = local_magnetic_suceptibility(
            input_data.evaluation_points[p][0]*corrector,
            input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);
        double J1 = magnetic_field * magnetic_field;

        //mef grad_b_T et grad_b
        double lambda_local=0.*-2.e4;
        if(J1>1.){
          lambda_local=lambda_local/J1;
        }

        force =
            1. / mu0 / (1. + chi_local) *
                cross_product_3d(-laplacien_vector_A, magnetic_field) 
            -chi_local/(1.+chi_local)/mu0*magnetic_field*(grad_b_T)
            +lambda_local/(1.+chi_local)/mu0*magnetic_field*(grad_b);

        computed_quantities[p][0] = force[0];
        computed_quantities[p][1] = force[1];
      }
    }
  };


  template <int dim>
  class ForceintPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    ForceintPostprocessor(double b0_in, double current_density_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorVector<dim>(
              "fint", update_gradients | update_quadrature_points | update_hessians),
          magnetic_field_loading(b0_in), current_density(current_density_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    double current_density;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);

        Tensor<2, 3> Id_tensor;
        Id_tensor[0][0] = 1.;
        Id_tensor[1][1] = 1.;
        Id_tensor[2][2] = 1.;

        // we compute sigma_m and then takes the divergence

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build local magnetisation
        Tensor<1, 3> magnetisation_vector =
            magnetisation(input_data.evaluation_points[p][0]*corrector,
                          input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);

        // build current vector
        Tensor<1, 3> current_vector;
        current_vector[2] =
            j(input_data.evaluation_points[p][0]*corrector,
              input_data.evaluation_points[p][1]*corrector, current_density,geom,size,size2,exterior_size);

        Tensor<1, 3> force;

        // curl_from_hessian_of_A(magnetic_field,input_data.solution_hessians) is
        // curl(curl(A)+B_irot)=curl(curl(A)), as A=Az, curl(curl(A))[0]=0 ,
        // curl(curl(A))[1]=0, curl(curl(A))[2]=-Az,xx-Az,yy = -\Delta Az
        // div_from_hessian_of_A(magnetic_field,input_data.solution_hessians) is
        // div(curl(A)+B_irot)=div(B_sole),

        // builgin laplacien vector of A ()
        Tensor<1, 3> laplacien_vector_A;
        laplacien_vector_A[2] = input_data.solution_hessians[p][0][0] +
                                input_data.solution_hessians[p][1][1];

        // we also need div m:
        // reminder, div(scalar * vector)=scalar*div(vector) + grad(scalar).vector
        // m = - 2 phi,J1 * b
        // div m = -2 phi,J1 * div b - 2 grad(phi,J1) . b =  -2 grad(phi,J1) . b
        // as div(b)=0 grad(phi,J1)=phi,J1J1 * grad(J1) grad(J1) = 2 grad(b).b we
        // need grad(b) div m =-2 phi,J1J1 * 2 (grad(b).b). b
        Tensor<2, 3> grad_b;
        grad_b[0][0] = input_data.solution_hessians[p][1][0];
        grad_b[0][1] = input_data.solution_hessians[p][1][1];
        grad_b[1][0] = -input_data.solution_hessians[p][0][0];
        grad_b[1][1] = -input_data.solution_hessians[p][0][1];

        double chi_local = local_magnetic_suceptibility(
            input_data.evaluation_points[p][0]*corrector,
            input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);
        double J1 = magnetic_field * magnetic_field;

        force =
            1. / mu0 / (1. + chi_local) *
                cross_product_3d(-laplacien_vector_A, magnetic_field) 
                +
            magnetisation_vector * grad_b 
            +
            4. *
                phi_J1_J1(input_data.evaluation_points[p][0]*corrector,
                          input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size) *
                (magnetic_field * grad_b) *
                (outer_product(magnetic_field, magnetic_field) - J1 * Id_tensor);

        double test_interior=0.;
        double x_coord = input_data.evaluation_points[p][0];
        double y_coord = input_data.evaluation_points[p][1];
        if(is_in(geom, x_coord,  y_coord, size,size2, exterior_size)){
          test_interior=1.;
        }

        computed_quantities[p][0] = force[0]*test_interior;
        computed_quantities[p][1] = force[1]*test_interior;
      }
    }
  };

  template <int dim>
  class NonlinearForcePostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    NonlinearForcePostprocessor(double b0_in, double current_density_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorVector<dim>(
              "non_linear_f",
              update_gradients | update_quadrature_points | update_hessians),
          magnetic_field_loading(b0_in), current_density(current_density_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    double current_density;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {

      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        AssertDimension(computed_quantities[p].size(), dim);

        Tensor<2, 3> Id_tensor;
        Id_tensor[0][0] = 1.;
        Id_tensor[1][1] = 1.;
        Id_tensor[2][2] = 1.;

        // we compute sigma_m and then takes the divergence

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build current vector
        Tensor<1, 3> current_vector;
        current_vector[2] =
            j(input_data.evaluation_points[p][0]*corrector,
              input_data.evaluation_points[p][1]*corrector, current_density,geom,size,size2,exterior_size);

        Tensor<1, 3> force;

        // curl_from_hessian_of_A(magnetic_field,input_data.solution_hessians) is
        // curl(curl(A)+B_irot)=curl(curl(A)), as A=Az, curl(curl(A))[0]=0 ,
        // curl(curl(A))[1]=0, curl(curl(A))[2]=-Az,xx-Az,yy = -\Delta Az
        // div_from_hessian_of_A(magnetic_field,input_data.solution_hessians) is
        // div(curl(A)+B_irot)=div(B_sole),

        // builgin laplacien vector of A ()
        Tensor<1, 3> laplacien_vector_A;
        laplacien_vector_A[2] = input_data.solution_hessians[p][0][0] +
                                input_data.solution_hessians[p][1][1];

        // we also need div m:
        // reminder, div(scalar * vector)=scalar*div(vector) + grad(scalar).vector
        // m = - 2 phi,J1 * b
        // div m = -2 phi,J1 * div b - 2 grad(phi,J1) . b =  -2 grad(phi,J1) . b
        // as div(b)=0 grad(phi,J1)=phi,J1J1 * grad(J1) grad(J1) = 2 grad(b).b we
        // need grad(b) div m =-2 phi,J1J1 * 2 (grad(b).b). b
        Tensor<2, 3> grad_b;
        grad_b[0][0] = input_data.solution_hessians[p][1][0];
        grad_b[0][1] = input_data.solution_hessians[p][1][1];
        grad_b[1][0] = -input_data.solution_hessians[p][0][0];
        grad_b[1][1] = -input_data.solution_hessians[p][0][1];

        double J1 = magnetic_field * magnetic_field;

        force = 4. *
                phi_J1_J1(input_data.evaluation_points[p][0]*corrector,
                          input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size) *
                (magnetic_field * grad_b) *
                (outer_product(magnetic_field, magnetic_field) - J1 * Id_tensor);

        computed_quantities[p][0] = force[0];
        computed_quantities[p][1] = force[1];
      }
    }
  };

  template <int dim>
  class Curlh_inplaneForcePostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    Curlh_inplaneForcePostprocessor(double b0_in, double current_density_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorVector<dim>(
              "Curlh_inplane",
              update_gradients | update_quadrature_points | update_hessians),
          magnetic_field_loading(b0_in), current_density(current_density_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    double current_density;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build local magnetisation
        Tensor<1, 3> magnetisation_vector =
            magnetisation(input_data.evaluation_points[p][0]*corrector,
                          input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);

        // build current vector
        Tensor<1, 3> current_vector;
        current_vector[2] =
            j(input_data.evaluation_points[p][0]*corrector,
              input_data.evaluation_points[p][1]*corrector, current_density,geom,size,size2,exterior_size);

        Tensor<1, 3> curl_m;

        // buildin laplacien vector of A ()
        Tensor<1, 3> laplacien_vector_A;
        laplacien_vector_A[2] = input_data.solution_hessians[p][0][0] +
                                input_data.solution_hessians[p][1][1];

        // we also need div m:
        // reminder, div(scalar * vector)=scalar*div(vector) + grad(scalar).vector
        // m = - 2 phi,J1 * b
        // div m = -2 phi,J1 * div b - 2 grad(phi,J1) . b =  -2 grad(phi,J1) . b
        // as div(b)=0 grad(phi,J1)=phi,J1J1 * grad(J1) grad(J1) = 2 grad(b).b we
        // need grad(b) div m =-2 phi,J1J1 * 2 (grad(b).b). b
        Tensor<2, 3> grad_b;
        grad_b[0][0] = input_data.solution_hessians[p][1][0];
        grad_b[0][1] = input_data.solution_hessians[p][1][1];
        grad_b[1][0] = -input_data.solution_hessians[p][0][0];
        grad_b[1][1] = -input_data.solution_hessians[p][0][1];

        double chi_local = local_magnetic_suceptibility(
            input_data.evaluation_points[p][0]*corrector,
            input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);
        double mu_local = mu0 * (1 + chi);
        double J1 = magnetic_field * magnetic_field;

        curl_m = chi_local / mu_local * (-laplacien_vector_A) -
                 4. *
                     phi_J1_J1(input_data.evaluation_points[p][0]*corrector,
                               input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size) *
                     cross_product_3d(magnetic_field * grad_b, magnetic_field);

        computed_quantities[p][0] = curl_m[0];
        computed_quantities[p][1] = curl_m[1];
      }
    }
  };

  template <int dim>
  class Curlh_zForcePostprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Curlh_zForcePostprocessor(double b0_in, double current_density_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorScalar<dim>("Curlh_z", update_gradients |
                                                      update_quadrature_points |
                                                      update_hessians),
          magnetic_field_loading(b0_in), current_density(current_density_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    double current_density;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {

      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), 1);

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build current vector
        Tensor<1, 3> current_vector;
        current_vector[2] =
            j(input_data.evaluation_points[p][0]*corrector,
              input_data.evaluation_points[p][1]*corrector, current_density,geom,size,size2,exterior_size);

        Tensor<1, 3> curl_m;

        // curl_from_hessian_of_A(magnetic_field,input_data.solution_hessians) is
        // curl(curl(A)+B_irot)=curl(curl(A)), as A=Az, curl(curl(A))[0]=0 ,
        // curl(curl(A))[1]=0, curl(curl(A))[2]=-Az,xx-Az,yy = -\Delta Az
        // div_from_hessian_of_A(magnetic_field,input_data.solution_hessians) is
        // div(curl(A)+B_irot)=div(B_sole),

        // builgin laplacien vector of A ()
        Tensor<1, 3> laplacien_vector_A;
        laplacien_vector_A[2] = input_data.solution_hessians[p][0][0] +
                                input_data.solution_hessians[p][1][1];

        // we also need div m:
        // reminder, div(scalar * vector)=scalar*div(vector) + grad(scalar).vector
        // m = - 2 phi,J1 * b
        // div m = -2 phi,J1 * div b - 2 grad(phi,J1) . b =  -2 grad(phi,J1) . b
        // as div(b)=0 grad(phi,J1)=phi,J1J1 * grad(J1) grad(J1) = 2 grad(b).b we
        // need grad(b) div m =-2 phi,J1J1 * 2 (grad(b).b). b
        Tensor<2, 3> grad_b;
        grad_b[0][0] = input_data.solution_hessians[p][1][0];
        grad_b[0][1] = input_data.solution_hessians[p][1][1];
        grad_b[1][0] = -input_data.solution_hessians[p][0][0];
        grad_b[1][1] = -input_data.solution_hessians[p][0][1];

        double chi_local = local_magnetic_suceptibility(
            input_data.evaluation_points[p][0]*corrector,
            input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);
        double mu_local = mu0 * (1 + chi_local);
        double J1 = magnetic_field * magnetic_field;

        curl_m = chi_local / mu_local * (-laplacien_vector_A) -
                 4. *
                     phi_J1_J1(input_data.evaluation_points[p][0]*corrector,
                               input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size) *
                     cross_product_3d(magnetic_field * grad_b, magnetic_field);

        double curl_h = 1 / mu0 * (-laplacien_vector_A[2]) - curl_m[2];

        computed_quantities[p](0) = curl_h;
      }
    }
  };

  template <int dim>
  class Curlh_zbisForcePostprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Curlh_zbisForcePostprocessor(double b0_in, double current_density_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorScalar<dim>(
              "Curlh_z_bis",
              update_gradients | update_quadrature_points | update_hessians),
          magnetic_field_loading(b0_in), current_density(current_density_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    double current_density;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {

      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        AssertDimension(computed_quantities[p].size(), 1);

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build current vector
        Tensor<1, 3> current_vector;
        current_vector[2] =
            j(input_data.evaluation_points[p][0]*corrector,
              input_data.evaluation_points[p][1]*corrector, current_density,geom,size,size2,exterior_size);

        // builgin laplacien vector of A ()
        Tensor<1, 3> laplacien_vector_A;
        laplacien_vector_A[2] = input_data.solution_hessians[p][0][0] +
                                input_data.solution_hessians[p][1][1];

        Tensor<2, 3> grad_b;
        grad_b[0][0] = input_data.solution_hessians[p][1][0];
        grad_b[0][1] = input_data.solution_hessians[p][1][1];
        grad_b[1][0] = -input_data.solution_hessians[p][0][0];
        grad_b[1][1] = -input_data.solution_hessians[p][0][1];

        double J1 = magnetic_field * magnetic_field;
        double phi_J1_local = phi_J1(input_data.evaluation_points[p][0]*corrector,
                                     input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size);
        double phi_J1_J1_local =
            phi_J1_J1(input_data.evaluation_points[p][0]*corrector,
                      input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size);

        Tensor<2, 3> grad_h;

        grad_h = grad_b * (1. / mu0 + 2. * phi_J1_local) +
                 4. * phi_J1_J1_local *
                     outer_product(magnetic_field, magnetic_field * grad_b);

        computed_quantities[p](0) = grad_h[1][0] - grad_h[0][1];
      }
    }
  };

  template <int dim>
  class MagnetisationPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    MagnetisationPostprocessor(double b0_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorVector<dim>(
              "m", update_gradients | update_quadrature_points | update_hessians),
          magnetic_field_loading(b0_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {

      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        AssertDimension(computed_quantities[p].size(), dim);

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build local magnetisation
        Tensor<1, 3> magnetisation_vector =
            magnetisation(input_data.evaluation_points[p][0]*corrector,
                          input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);
        
        double adim=get_minf();

        computed_quantities[p][0] = magnetisation_vector[0]/adim;
        computed_quantities[p][1] = magnetisation_vector[1]/adim;
      }
    }
  };

  template <int dim>
  class MagneticstrengthPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    MagneticstrengthPostprocessor(double b0_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorVector<dim>(
              "h", update_gradients | update_quadrature_points | update_hessians),
          magnetic_field_loading(b0_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build local magnetisation
        Tensor<1, 3> magnetisation_vector =
            magnetisation(input_data.evaluation_points[p][0]*corrector,
                          input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);

        Tensor<1, 3> magnetic_field_strength =
            1. / mu0 * magnetic_field - magnetisation_vector;

        computed_quantities[p][0] = magnetic_field_strength[0];
        computed_quantities[p][1] = magnetic_field_strength[1];
      }
    }
  };

  template <int dim>
  class MagneticstrengthcylindPostprocessor
      : public DataPostprocessorVector<dim>
  {
  public:
    MagneticstrengthcylindPostprocessor(double b0_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorVector<dim>(
              "h_cylindrical",
              update_gradients | update_quadrature_points | update_hessians),
          magnetic_field_loading(b0_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build local magnetisation
        Tensor<1, 3> magnetisation_vector =
            magnetisation(input_data.evaluation_points[p][0]*corrector,
                          input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);

        Tensor<1, 3> magnetic_field_strength =
            1. / mu0 * magnetic_field - magnetisation_vector;

        double theta;
        if (abs(input_data.evaluation_points[p][0]*corrector) > 1.e-9)
        {
          theta = std::atan(input_data.evaluation_points[p][1]*corrector /
                            abs(input_data.evaluation_points[p][0]*corrector));
          if (input_data.evaluation_points[p][0]*corrector < 0.)
          {
            theta = M_PI - theta;
          }
          double cos = std::cos(theta);
          double sin = std::sin(theta);

          computed_quantities[p][0] =
              cos * magnetic_field_strength[0] + sin * magnetic_field_strength[1];
          computed_quantities[p][1] = -sin * magnetic_field_strength[0] +
                                      cos * magnetic_field_strength[1];
        }
        else
        {
          if (input_data.evaluation_points[p][1]*corrector > 0)
          {
            computed_quantities[p][0] = magnetic_field_strength[1];
            computed_quantities[p][1] = -magnetic_field_strength[0];
          }
          else
          {
            computed_quantities[p][0] = -magnetic_field_strength[1];
            computed_quantities[p][1] = magnetic_field_strength[0];
          }
        }
      }
    }
  };

  template <int dim>
  class LaplaceForcePostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    LaplaceForcePostprocessor(double b0_in, double current_density_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorVector<dim>("fLaplace", update_gradients |
                                                       update_quadrature_points |
                                                       update_hessians),
          magnetic_field_loading(b0_in), current_density(current_density_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    double current_density;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      // then loop over all of these inputs:
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), dim);

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // build current vector
        Tensor<1, 3> current_vector;
        current_vector[2] =
            j(input_data.evaluation_points[p][0]*corrector,
              input_data.evaluation_points[p][1]*corrector, current_density,geom,size,size2,exterior_size);

        Tensor<1, 3> force;

        force = cross_product_3d(current_vector, magnetic_field);

        computed_quantities[p][0] = force[0];
        computed_quantities[p][1] = force[1];
      }
    }
  };

  template <int dim>
  class Laplacien_A_Postprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Laplacien_A_Postprocessor()
        : DataPostprocessorScalar<dim>("laplacienA", update_quadrature_points |
                                                         update_gradients |
                                                         update_hessians) {}
    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        Assert(computed_quantities[p].size() == 1,
               ExcDimensionMismatch(computed_quantities[p].size(), 1));
        computed_quantities[p](0) = input_data.solution_hessians[p][0][0] +
                                    input_data.solution_hessians[p][1][1];
      }
    }
  };

  template <int dim>
  class Divb_Postprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Divb_Postprocessor()
        : DataPostprocessorScalar<dim>("Div_b", update_quadrature_points |
                                                    update_gradients |
                                                    update_hessians) {}
    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {

      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        Assert(computed_quantities[p].size() == 1,
               ExcDimensionMismatch(computed_quantities[p].size(), 1));
        computed_quantities[p](0) = input_data.solution_hessians[p][1][0] -
                                    input_data.solution_hessians[p][0][1];
      }
    }
  };

  template <int dim>
  class Laplacien_A_01_Postprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Laplacien_A_01_Postprocessor()
        : DataPostprocessorScalar<dim>("laplacienA01", update_quadrature_points |
                                                           update_gradients |
                                                           update_hessians) {}
    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        Assert(computed_quantities[p].size() == 1,
               ExcDimensionMismatch(computed_quantities[p].size(), 1));
        computed_quantities[p](0) = input_data.solution_hessians[p][0][1];
      }
    }
  };

  template <int dim>
  class Laplacien_A_00_Postprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Laplacien_A_00_Postprocessor()
        : DataPostprocessorScalar<dim>("laplacienA00", update_quadrature_points |
                                                           update_gradients |
                                                           update_hessians) {}
    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {

      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        Assert(computed_quantities[p].size() == 1,
               ExcDimensionMismatch(computed_quantities[p].size(), 1));
        computed_quantities[p](0) = input_data.solution_hessians[p][0][0];
      }
    }
  };

  template <int dim>
  class Laplacien_A_11_Postprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Laplacien_A_11_Postprocessor()
        : DataPostprocessorScalar<dim>("laplacienA11", update_quadrature_points |
                                                           update_gradients |
                                                           update_hessians) {}
    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {

      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        Assert(computed_quantities[p].size() == 1,
               ExcDimensionMismatch(computed_quantities[p].size(), 1));
        computed_quantities[p](0) = input_data.solution_hessians[p][1][1];
      }
    }
  };

  template <int dim>
  class Laplacien_A_10_Postprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Laplacien_A_10_Postprocessor()
        : DataPostprocessorScalar<dim>("laplacienA10", update_quadrature_points |
                                                           update_gradients |
                                                           update_hessians) {}
    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {

      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        Assert(computed_quantities[p].size() == 1,
               ExcDimensionMismatch(computed_quantities[p].size(), 1));
        computed_quantities[p](0) = input_data.solution_hessians[p][1][0];
      }
    }
  };

  template <int dim>
  class Current_Postprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Current_Postprocessor(double current_density_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in,bool is_alternative_in)
        : DataPostprocessorScalar<dim>("current", update_quadrature_points |
                                                      update_gradients),
          current_density(current_density_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in),is_alternative(is_alternative_in) {}
    double current_density;
    unsigned int geom;
    double size;
    double size2;
    double exterior_size;
    bool is_alternative;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {

      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        Assert(computed_quantities[p].size() == 1,
               ExcDimensionMismatch(computed_quantities[p].size(), 1));

        if(is_alternative){
          computed_quantities[p](0) =j_alternate(input_data.evaluation_points[p][0]*corrector,input_data.evaluation_points[p][1]*corrector, current_density,geom,size,size2,exterior_size);
        }
        else{
          computed_quantities[p](0) =j(input_data.evaluation_points[p][0]*corrector,input_data.evaluation_points[p][1]*corrector, current_density,geom,size,size2,exterior_size);
        }
            
      }
    }
  };

  template <int dim>
  class Chi_Postprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Chi_Postprocessor(double b0_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorScalar<dim>("chi", update_quadrature_points |
                                                  update_gradients),
          magnetic_field_loading(b0_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    unsigned int geom;
    double size;
    double size2;
double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {

      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        // get local local_magnetic_suceptibility
        double chi_local = local_magnetic_suceptibility(
            input_data.evaluation_points[p][0]*corrector,
            input_data.evaluation_points[p][1]*corrector, magnetic_field,geom,size,size2,exterior_size);

        computed_quantities[p](0) = chi_local;
      }
    }
  };

  template <int dim>
  class Bnorm_Postprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    Bnorm_Postprocessor(double b0_in, double current_density_in)
        : DataPostprocessorScalar<dim>("bnorm", update_quadrature_points |
                                                    update_gradients),
          magnetic_field_loading(b0_in), current_density(current_density_in) {}

    double magnetic_field_loading;
    double current_density;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {

        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] =
            magnetic_field_loading - input_data.solution_gradients[p][0];

        computed_quantities[p](0) = magnetic_field.norm();
      }
    }
  };

  template <int dim>
  class Sigma_Postprocessor : public DataPostprocessorTensor<dim>
  {
  public:
    Sigma_Postprocessor(double b0_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorTensor<dim>("sigma_mag", update_quadrature_points |
                                                    update_gradients),
          magnetic_field_loading(b0_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    unsigned int geom;
    double size;
    double size2;
    double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] = magnetic_field_loading - input_data.solution_gradients[p][0];

        Tensor<2,3> sigma;
        Tensor<2,3> Identity_tensor;
        Identity_tensor[0][0]=1.;
        Identity_tensor[1][1]=1.;
        Identity_tensor[2][2]=1.;
        double J1=magnetic_field*magnetic_field;
        
        sigma=1./mu0*(outer_product(magnetic_field,magnetic_field)-1./2.*contract<0,0>(magnetic_field,magnetic_field)*Identity_tensor)
              +2.*phi_J1(input_data.evaluation_points[p][0]*corrector,input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size)*(outer_product(magnetic_field,magnetic_field)-contract<0,0>(magnetic_field,magnetic_field)*Identity_tensor);
        
        for (unsigned int d=0;d<dim;++d){
          for (unsigned int e=0;e<dim;++e){
            computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))] = sigma[d][e];
          }
        }
      }
    }
  };

  template <int dim>
  class Sigma_Postprocessor_xx : public DataPostprocessorScalar<dim>
  {
  public:
    Sigma_Postprocessor_xx(double b0_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorScalar<dim>("sigma_mag_xx", update_quadrature_points |
                                                    update_gradients),
          magnetic_field_loading(b0_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    unsigned int geom;
    double size;
    double size2;
    double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] = magnetic_field_loading - input_data.solution_gradients[p][0];

        Tensor<2,3> sigma;
        Tensor<2,3> Identity_tensor;
        Identity_tensor[0][0]=1.;
        Identity_tensor[1][1]=1.;
        Identity_tensor[2][2]=1.;
        double J1=magnetic_field*magnetic_field;
        
        sigma=1./mu0*(outer_product(magnetic_field,magnetic_field)-1./2.*contract<0,0>(magnetic_field,magnetic_field)*Identity_tensor)
              +2.*phi_J1(input_data.evaluation_points[p][0]*corrector,input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size)*(outer_product(magnetic_field,magnetic_field)-contract<0,0>(magnetic_field,magnetic_field)*Identity_tensor);

        double adim=get_minf()*get_minf()*mu0;
        
        computed_quantities[p](0)=sigma[0][0]/adim;
      }
    }
  };

  template <int dim>
  class Sigma_Postprocessor_xy : public DataPostprocessorScalar<dim>
  {
  public:
    Sigma_Postprocessor_xy(double b0_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorScalar<dim>("sigma_mag_xy", update_quadrature_points |
                                                    update_gradients),
          magnetic_field_loading(b0_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    unsigned int geom;
    double size;
    double size2;
    double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] = magnetic_field_loading - input_data.solution_gradients[p][0];

        Tensor<2,3> sigma;
        Tensor<2,3> Identity_tensor;
        Identity_tensor[0][0]=1.;
        Identity_tensor[1][1]=1.;
        Identity_tensor[2][2]=1.;
        double J1=magnetic_field*magnetic_field;
        
        sigma=1./mu0*(outer_product(magnetic_field,magnetic_field)-1./2.*contract<0,0>(magnetic_field,magnetic_field)*Identity_tensor)
              +2.*phi_J1(input_data.evaluation_points[p][0]*corrector,input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size)*(outer_product(magnetic_field,magnetic_field)-contract<0,0>(magnetic_field,magnetic_field)*Identity_tensor);
        
        double adim=get_minf()*get_minf()*mu0;

        computed_quantities[p](0)=sigma[0][1]/adim;
      }
    }
  };

  template <int dim>
  class Sigma_Postprocessor_yy : public DataPostprocessorScalar<dim>
  {
  public:
    Sigma_Postprocessor_yy(double b0_in,unsigned int geom_in,double size_in,double size2_in,double exterior_size_in)
        : DataPostprocessorScalar<dim>("sigma_mag_yy", update_quadrature_points |
                                                    update_gradients),
          magnetic_field_loading(b0_in), geom(geom_in), size(size_in), size2(size2_in) ,exterior_size(exterior_size_in) {}
    double magnetic_field_loading;
    unsigned int geom;
    double size;
    double size2;
    double exterior_size;

    virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        // build local magnetic field
        Tensor<1, 3> magnetic_field;
        magnetic_field[0] = input_data.solution_gradients[p][1];
        magnetic_field[1] = magnetic_field_loading - input_data.solution_gradients[p][0];

        Tensor<2,3> sigma;
        Tensor<2,3> Identity_tensor;
        Identity_tensor[0][0]=1.;
        Identity_tensor[1][1]=1.;
        Identity_tensor[2][2]=1.;
        double J1=magnetic_field*magnetic_field;
        
        sigma=1./mu0*(outer_product(magnetic_field,magnetic_field)-1./2.*contract<0,0>(magnetic_field,magnetic_field)*Identity_tensor)
              +2.*phi_J1(input_data.evaluation_points[p][0]*corrector,input_data.evaluation_points[p][1]*corrector, J1,geom,size,size2,exterior_size)*(outer_product(magnetic_field,magnetic_field)-contract<0,0>(magnetic_field,magnetic_field)*Identity_tensor);
        
        double adim=get_minf()*get_minf()*mu0;

        computed_quantities[p](0)=sigma[1][1]/adim;
      }
    }
  };

} // namespace output

#endif // OUTPUT