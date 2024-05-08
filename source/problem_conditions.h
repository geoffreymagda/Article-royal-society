
#ifndef PB_COND
#define PB_COND

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

#include "tools.h"

using namespace dealii;
using namespace tools;

//#define HE_MODEL
//#define QUICK_SATURATION_MODEL
#define LANGEVIN

namespace pb_cond
{
  unsigned int refinement_level = 0;

  double mu0 = 4. * M_PI / 10000000.;
  double minf = 1.7 * 1e6;
  double chi = 2500.*0.;
  double chir=chi/(1.+chi);
  bool simplified_magnetism = false;
  double mu_mag = mu0 * (1. + chi);
  double alpha = 1.;
  double blimite = 0.8;
  double ms = (minf - blimite * chi / mu_mag) / alpha;
  double A = 3. * chi / ms * (1. + chi);
  double beta = 3. * chi / alpha / ms / mu_mag;

  bool is_in(unsigned int geom,double x, double y, double size,double size2,double exterior_size){
    double margin=1.000000000001;
    if(geom == 0){
      //circle
      double r = std::sqrt(x * x + y * y);
      return (r <= size*margin);
    }
    else if(geom ==1){
      return( (x*x<size*size*margin) && (y*y<size*size*margin) );
    }
    else if(geom ==2){
      return( (x*x<size*size*margin) && (y*y<size2*size2*margin) );
    }
    else if(geom ==3){
      double lx=size*size/size2;
      double ly=size2;
      return( x*x/lx/lx+y*y/ly/ly<=1.*margin );
    }
    else if (geom>2000){
      unsigned int reduced_index=geom-2000;
      assert(reduced_index>0);
      unsigned int local_geom=(reduced_index-reduced_index % 100)/100;
      unsigned int repeat=reduced_index % 100;
      double Lxlocal;
      if(local_geom==0){
        Lxlocal=2.*size * 2.+exterior_size;
      }
      else if(local_geom==3){
        Lxlocal=(2.*size * 2.+exterior_size)*size/size2;
      }
      else{
        Lxlocal=0;
        Assert(1==0, ExcMessage("wrong geom"));
      }
      
      for(unsigned int i=0;i<repeat;i++){
        if(is_in(local_geom,x-i*Lxlocal,y,size,size2, exterior_size)){
          return true;
        }
      }

      return false;
    }
    else{
      Assert(1==0, ExcMessage("geom must be 0 or 1 or 2 or 3"));
      return false;
    }
  }

  double is_in_parity(unsigned int geom,double x, double y, double size,double size2,double exterior_size){
    double margin=1.000000000001;
    if(geom == 0){
      //circle
      double r = std::sqrt(x * x + y * y);
      if(r <= size*margin){return 1.;}
      else{return 0.;}
    }
    else if(geom ==1){
      if((x*x<size*size*margin) && (y*y<size*size*margin)){return 1.;}
      else{return 0.;}
    }
    else if(geom ==2){
      if((x*x<size*size*margin) && (y*y<size2*size2*margin)){return 1.;}
      else{return 0.;}
    }
    else if(geom ==3){
      double lx=size*size/size2;
      double ly=size2;
      if(x*x/lx/lx+y*y/ly/ly<=1.*margin){return 1.;}
      else{return 0.;}
    }
    else if (geom>2000){
      unsigned int reduced_index=geom-2000;
      assert(reduced_index>0);
      unsigned int local_geom=(reduced_index-reduced_index % 100)/100;
      unsigned int repeat=reduced_index % 100;
      double Lxlocal;
      if(local_geom==0){
        Lxlocal=2.*size * 2.+exterior_size;
      }
      else if(local_geom==3){
        Lxlocal=(2.*size * 2.+exterior_size)*size/size2;
      }
      else{
        Lxlocal=0;
        Assert(1==0, ExcMessage("wrong geom"));
      }
      
      for(unsigned int i=0;i<repeat;i++){
        if(is_in(local_geom,x-i*Lxlocal,y,size,size2, exterior_size)){
          if(i%2==0){
            return 1.;
          }
          else{
            return -1.;
          }
        }
      }

      return 0.;
    }
    else{
      Assert(1==0, ExcMessage("geom must be 0 or 1 or 2 or 3"));
      return 0.;
    }
  }

  bool is_in_left(unsigned int geom,double x, double y, double size,double size2,double exterior_size){
    double margin=1.000000000001;
    if(geom == 0){
      //circle
      double r = std::sqrt(x * x + y * y);
      return (r <= size*margin);
    }
    else if(geom ==1){
      return( (x*x<size*size*margin) && (y*y<size*size*margin) );
    }
    else if(geom ==2){
      return( (x*x<size*size*margin) && (y*y<size2*size2*margin) );
    }
    else if(geom ==3){
      double lx=size*size/size2;
      double ly=size2;
      return( x*x/lx/lx+y*y/ly/ly<=1.*margin );
    }
    else if (geom>2000){
      unsigned int reduced_index=geom-2000;
      assert(reduced_index>0);
      unsigned int local_geom=(reduced_index-reduced_index % 100)/100;
      
      if(is_in(local_geom,x,y,size,size2, exterior_size)){
        return true;
      }


      return false;
    }
    else{
      Assert(1==0, ExcMessage("geom must be 0 or 1 or 2 or 3"));
      return false;
    }
  }



  bool is_in_number(unsigned int geom,double x, double y, double size,double size2,double exterior_size,unsigned int number){
    double margin=1.000000000001;
    if(geom == 0){
      if(number!=0){return false;}
      //circle
      double r = std::sqrt(x * x + y * y);
      return (r <= size*margin);
    }
    else if(geom ==1){
      if(number!=0){return false;}
      return( (x*x<size*size*margin) && (y*y<size*size*margin) );
    }
    else if(geom ==2){
      if(number!=0){return false;}
      return( (x*x<size*size*margin) && (y*y<size2*size2*margin) );
    }
    else if(geom ==3){
      if(number!=0){return false;}
      double lx=size*size/size2;
      double ly=size2;
      return( x*x/lx/lx+y*y/ly/ly<=1.*margin );
    }
    else if (geom>2000){
      unsigned int reduced_index=geom-2000;
      assert(reduced_index>0);
      unsigned int local_geom=(reduced_index-reduced_index % 100)/100;
      unsigned int repeat=reduced_index % 100;
      double Lxlocal;
      if(local_geom==0){
        Lxlocal=2.*size * 2.+exterior_size;
      }
      else if(local_geom==3){
        Lxlocal=(2.*size * 2.+exterior_size)*size/size2;
      }
      else{
        Lxlocal=0;
        Assert(1==0, ExcMessage("wrong geom"));
      }
      
      if(is_in(local_geom,x-number*Lxlocal,y,size,size2, exterior_size)){
          return true;
        }


      return false;
    }
    else{
      Assert(1==0, ExcMessage("wrong geom number"));
      return false;
    }
  }

#ifdef HE_MODEL

std::vector<double> alpha_list={2.423e-4,0.61e-4,-1.487e-4,6.435e-4,-9.935e-4,7.408e-4,-2.617e-4,0.365e-4};
std::vector<double> beta_list={-5.398e-1,0,0,0};
std::vector<double> gamma_list={372.498,0,0};

  double phi_J1(double x, double y, double J1, unsigned int geom, double size,double size2,double exterior_size)
  {
    if (is_in(geom,x,y,size, size2,exterior_size))
    {
      double results=-1./mu0/2.;
      for(unsigned int i=0;i<8;i++){
        results+=-1./mu0*alpha_list[i]*std::pow(J1,i);
      }
      for(unsigned int i=0;i<1;i++){
        results+=-1./mu0*beta_list[i]*std::pow(J1,i)*0.;
      }
      for(unsigned int i=0;i<1;i++){
        results+=-1./mu0*gamma_list[i]*std::pow(J1,i)*0.;
      }
      return results;
    }
    else
    {
      return 0.;
    }
  }

  double phi_J1_J1(double x, double y, double J1, unsigned int geom, double size,double size2,double exterior_size)
  {
    if (is_in(geom,x,y,size, size2,exterior_size) )
    {
      double results=0;
      for(unsigned int i=1;i<8;i++){
        results+=-1./mu0*alpha_list[i]* (double) i * std::pow(J1,i-1);
      }
      for(unsigned int i=1;i<2;i++){
        results+=-1./mu0*beta_list[i]* (double) i * std::pow(J1,i-1)*0.;
      }
      for(unsigned int i=1;i<2;i++){
        results+=-1./mu0*gamma_list[i]* (double) i * std::pow(J1,i-1)*0.;
      }
      return -results;
    }
    else
    {
      return 0.;
    }
  }
#endif // HE_MODEL

#ifdef HE_MODEL2

std::vector<double> alpha_list={7.3e-5,-97.03,250.427,-250.520,117.438,-13.709,-6.202};
std::vector<double> beta_list={-0.283,-1.872,-4.798};
std::vector<double> gamma_list={1.178,1.059,58.230};

  double phi_J1(double x, double y, double J1, unsigned int geom, double size,double size2,double exterior_size)
  {
    if (is_in(geom,x,y,size, size2,exterior_size))
    {
      //double results=-1./mu0/8.;
      double results=0.; //why not ?
      for(unsigned int i=0;i<6;i++){
        results+=alpha_list[i]*std::pow(J1,i);
      }
      for(unsigned int i=0;i<2;i++){
        results+=beta_list[i]*std::pow(J1,i)*0.;
      }
      for(unsigned int i=0;i<2;i++){
        results+=gamma_list[i]*std::pow(J1,i)*0.;
      }
      return results;
    }
    else
    {
      return 0.;
    }
  }

  double phi_J1_J1(double x, double y, double J1, unsigned int geom, double size,double size2,double exterior_size)
  {
    if (is_in(geom,x,y,size, size2,exterior_size) )
    {
      double results=0;
      for(unsigned int i=1;i<6;i++){
        results+=alpha_list[i]* (double) i * std::pow(J1,i-1);
      }
      for(unsigned int i=1;i<2;i++){
        results+=beta_list[i]* (double) i * std::pow(J1,i-1)*0.;
      }
      for(unsigned int i=1;i<2;i++){
        results+=gamma_list[i]* (double) i * std::pow(J1,i-1)*0.;
      }
      return results;
    }
    else
    {
      return 0.;
    }
  }
#endif // HE_MODEL2

#ifdef QUICK_SATURATION_MODEL

  double phi_J1(double x, double y, double J1, unsigned int geom, double size,double size2,double exterior_size)
  {
    if (is_in(geom,x,y,size, size2,exterior_size))
    {
      if (simplified_magnetism)
      {
        return -chi / 2. / mu0 / (1. + chi);
      }
      else
      {
        if (J1 > blimite * blimite)
        {
          if (abs(std::sqrt(J1) - blimite) < 1. / 1000000000.)
          {
            std::cout << abs(std::sqrt(J1) - blimite) << " " << J1 << " "
                      << coth((std::sqrt(J1) - blimite)) << std::endl;
          }
          return -blimite * chi / mu_mag / 2. / std::sqrt(J1) +
                 alpha * ms / beta *
                     (1. / 2. / std::sqrt(J1) / (std::sqrt(J1) - blimite) -
                      beta / 2. / std::sqrt(J1) *
                          coth(beta * (std::sqrt(J1) - blimite)));
        }
        else
        {
          return -chi / mu_mag / 2.;
        }
      }
    }
    else
    {
      return 0.;
    }
  }

  double phi_J1_J1(double x, double y, double J1, unsigned int geom, double size,double size2,double exterior_size)
  {
    if (is_in(geom,x,y,size, size2,exterior_size) && !simplified_magnetism)
    {
      if (J1 > blimite * blimite)
      {
        return blimite * chi / 4. / J1 / std::sqrt(J1) / mu_mag +
               alpha * ms / beta *
                   (-1. / 4. / J1 / std::sqrt(J1) / (std::sqrt(J1) - blimite) -
                    1. / 4. / J1 / (std::sqrt(J1) - blimite) /
                        (std::sqrt(J1) - blimite) +
                    beta / 4. / std::sqrt(J1) / J1 *
                        coth(beta * (std::sqrt(J1) - blimite)) +
                    beta * beta / 4. / J1 /
                        sinh(beta * (std::sqrt(J1) - blimite)) /
                        sinh(beta * (std::sqrt(J1) - blimite)));
      }
      else
      {
        return 0.;
      }
    }
    else
    {
      return 0.;
    }
  }

#endif // QUICK_SATURATION_MODEL

#ifdef LANGEVIN

  double phi_J1(double x, double y, double J1, unsigned int geom, double size,double size2,double exterior_size)
  {
    if (is_in(geom,x,y,size, size2,exterior_size))
    {
      if(J1<1.e-10){
        return -chir/2./mu0;
      }
      else{
        return -minf*tanh(chir*std::sqrt(J1)/minf/mu0)/(2.* std::sqrt(J1));
      }
    }
    else
    {
      return 0.;
    }
  }

  double phi_J1_J1(double x, double y, double J1, unsigned int geom, double size,double size2,double exterior_size)
  {
    if (is_in(geom,x,y,size, size2,exterior_size) && !simplified_magnetism)
    {
      if(J1<1.e-10){
        return chir*chir*chir/6./minf/minf/mu0/mu0/mu0;
      }
      else{
        return -chir*std::pow(1./cosh((chir*std::sqrt(J1))/minf/mu0),2.)/(4.*J1*mu0) + minf*tanh(chir*std::sqrt(J1)/minf/mu0)/4./J1/std::sqrt(J1);
      }
    }
    else
    {
      return 0.;
    }
  }

#endif // LANGEVIN

  Tensor<1, 2> magnetisation(double x, double y, Tensor<1, 2> b, unsigned int geom, double size,double size2,double exterior_size)
  {
    return -phi_J1(x, y, b * b,geom,size, size2, exterior_size) * 2. * b;
  }

  Tensor<1, 3> magnetisation(double x, double y, Tensor<1, 3> b, unsigned int geom, double size,double size2,double exterior_size)
  {
    return -phi_J1(x, y, b * b,geom,size, size2, exterior_size) * 2. * b;
  }

  Tensor<1, 2> local_magnetic_field_stress(double x, double y, Tensor<1, 2> b, unsigned int geom, double size,double size2,double exterior_size)
  {
    return 1. / mu0 * b - magnetisation(x, y, b,geom,size, size2, exterior_size);
  }

  Tensor<1, 3> local_magnetic_field_stress(double x, double y, Tensor<1, 3> b, unsigned int geom, double size,double size2,double exterior_size)
  {
    return 1. / mu0 * b - magnetisation(x, y, b,geom,size, size2, exterior_size);
  }

  double local_magnetic_suceptibility(double x, double y, Tensor<1, 2> b, unsigned int geom, double size,double size2,double exterior_size)
  {
    return magnetisation(x, y, b,geom,size, size2, exterior_size).norm() /
           local_magnetic_field_stress(x, y, b,geom,size, size2,exterior_size).norm();
  }

  double local_magnetic_suceptibility(double x, double y, Tensor<1, 3> b, unsigned int geom, double size,double size2,double exterior_size)
  {

    if (is_in(geom,x,y,size, size2, exterior_size))
    {
      #ifdef QUICK_SATURATION_MODEL
      if (b.norm() > blimite)
      {
        return magnetisation(x, y, b,geom,size, size2, exterior_size).norm() /
               local_magnetic_field_stress(x, y, b,geom,size, size2, exterior_size).norm();
      }
      else
      {
        return chi;
      }
      #endif // QUICK_SATURATION_MODEL

      #ifdef HE_MODEL
        return magnetisation(x, y, b,geom,size, size2, exterior_size).norm() /
               local_magnetic_field_stress(x, y, b,geom,size, size2, exterior_size).norm();
      #endif // HE_MODEL

      #ifdef HE_MODEL2
        return magnetisation(x, y, b,geom,size, size2, exterior_size).norm() /
               local_magnetic_field_stress(x, y, b,geom,size, size2, exterior_size).norm();
      #endif // HE_MODEL2

      #ifdef LANGEVIN
        if(b.norm()<1.e-10){
          return chi;
        }
        else{
          return magnetisation(x, y, b,geom,size, size2, exterior_size).norm() /
                local_magnetic_field_stress(x, y, b,geom,size, size2, exterior_size).norm();
        }
      #endif //LANGEVIN
      
    }
    else
    {
      return 0.;
    }
  }

  double j(double x, double y, double current_density_value, unsigned int geom, double size,double size2, double exterior_size)
  {
    double solution = 0.;
    if (is_in(geom,x,y,size, size2, exterior_size))
    {
      solution = current_density_value;
    }
    return solution;
  }

  double j_alternate(double x, double y, double current_density_value, unsigned int geom, double size,double size2, double exterior_size)
  {
    double sign=is_in_parity(geom,x,y,size, size2, exterior_size);
    double solution = current_density_value*sign;
    return solution;
  }

  double get_minf()
  {
    return minf;
  }

} // namespace pb_cond

#endif // PB_COND