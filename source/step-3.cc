/*
This code is based on the deal.II library
*/

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
#include <deal.II/grid/grid_tools.h>
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
#include <stdlib.h>     /* srand, rand */

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/error_estimator.h>

#include <thread>
#include <deal.II/base/thread_management.h>

#include "tools.h"
#include "problem_conditions.h"
#include "output.h"

using namespace dealii;
using namespace tools;
using namespace pb_cond;
using namespace output;

std::ofstream myfile;
double convergence_param = 1.;

class Step3
{
public:
  Step3(double current_density_in, double magnetic_field_in, unsigned int shape_function_in, unsigned int geom_in,double size_in,double size2_in,unsigned int number_refining_command_int,double size_multiplicator_in,double distance_multiplicator_in, bool is_alternative_current_in, bool periodic_in);

  double run(unsigned int number_refinment_cycles, bool output_control);

private:
  void make_grid();
  void make_grid_circle();
  void make_grid_square();
  void make_grid_rectangle();
  void make_grid_near_square();
  void make_grid_oval();
  void make_grid_multi();
  void make_grid_multi_direct_circle();
  void setup_system();
  void assemble_system();
  void solve();
  void solve_44();
  void output_results(unsigned int cycle) const;
  double get_total_stress(double time);
  double get_integral_force(double time);
  double get_integral_force_bis(double time);
  void refine_grid();
  void central_refinment(double base_size,double scaling_factor,unsigned int number_refining);
  void central_refinment_repeat(double base_size,double scaling_factor,unsigned int number_refining, double Lx,unsigned int repeat);

  Triangulation<2> triangulation;
  Triangulation<2> triangulation_after;
  Triangulation<2> copy2triangulation;
  Triangulation<2> copy2triangulation_after;
  Triangulation<2> memorytriangulation;

  Triangulation<2> exterior;
  Triangulation<2> interior;
  Triangulation<2> shell;
  Triangulation<2> interior_recover;
  Triangulation<2> exterior_recover;
  Triangulation<2> shell_recover;
  FE_Q<2> fe;
  DoFHandler<2> dof_handler;
  AffineConstraints<double> constraints;
  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> solution;
  Vector<double> initial_solution;
  bool initiate_solution = true;
  Vector<double> system_rhs;
  double current_density;
  double b0;
  unsigned int shape_function;
  std::string test_name;
  unsigned int total_refinment = refinement_level;

  unsigned int geometry;
  double size;
  double size2;
  unsigned int number_refining_command;
  double size_multiplicator;
  double distance_multiplicator;
  bool is_alternative_current;
  double inner_radius = size * 0.5;
  double exterior_size = size_multiplicator * size;
  double margin_for_building = 1. + 1. / 100000000.;
  double radius_adapted = inner_radius * margin_for_building;
  bool is_saturated=false;

  //bool periodic=true;
  bool periodic;

};

Step3::Step3(double current_density_in, double magnetic_field_in, unsigned int shape_function_in,unsigned int geom_in,double size_in,double size2_in,unsigned int number_refining_command_in,double size_multiplicator_in, double distance_multiplicator_in, bool is_alternative_current_in, bool periodic_in)
    : fe(shape_function_in), dof_handler(triangulation), current_density(current_density_in),
      b0(magnetic_field_in), shape_function(shape_function_in), geometry(geom_in), size(size_in), size2(size2_in),number_refining_command(number_refining_command_in),size_multiplicator(size_multiplicator_in),distance_multiplicator(distance_multiplicator_in), is_alternative_current(is_alternative_current_in), periodic(periodic_in) {}

void Step3::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<2>::estimate(dof_handler, QGauss<2 - 1>(fe.degree + 1),
                                   {}, solution, estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, estimated_error_per_cell, 0.3, 0.);
  triangulation.execute_coarsening_and_refinement();
}

void Step3::central_refinment(double base_size,double scaling_factor,unsigned int number_refining){
  for (unsigned int i=1;i<number_refining+1;i++){
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->center().norm() < size * base_size*std::pow(scaling_factor,i))
        cell->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();
    total_refinment++;
  }
}

void Step3::central_refinment_repeat(double base_size,double scaling_factor,unsigned int number_refining, double Lx,unsigned int repeat){
  for (unsigned int i=1;i<number_refining+1;i++){
    for (unsigned int j=0;j<repeat;j++){
      for (const auto &cell : triangulation.active_cell_iterators()){
        double distance=(cell->center()[0]-(double)j*Lx)*(cell->center()[0]-(double)j*Lx)+(cell->center()[1])*(cell->center()[1]);
        if (std::sqrt(distance) < size * base_size*std::pow(scaling_factor,i)){
          if (!cell->refine_flag_set())
          {
            cell->set_refine_flag();
          }
        }
      }
    }

    triangulation.execute_coarsening_and_refinement();
    total_refinment++;
  }
}

void Step3::make_grid(){
  if(geometry==0){
    make_grid_circle();
  }
  else if(geometry==1){
    make_grid_near_square();
  }
  else if(geometry==2){
    make_grid_rectangle();
  }
  else if (geometry==3){
    make_grid_oval();
  }
  else if (geometry>1000 && geometry<2000){
    make_grid_multi();
  }
  else if (geometry>2000){
    make_grid_multi_direct_circle();
  }
  else{
    Assert(1==0, ExcMessage("geom is not recognised"));
  }
}

class Scalexy
{
public:
  explicit Scalexy(const double factorx,const double factory)
    : factorx(factorx),factory(factory)
  {}
  Point<2>
  operator()(const Point<2> p) const
  {
    Point<2> results;
    results[0]=p[0]*factorx;
    results[1]=p[1]*factory;
    return results;
  }

private:
  const double factorx;
  const double factory;
};

void Step3::make_grid_multi_direct_circle(){
  total_refinment = refinement_level;

  double padding_multiplier=distance_multiplicator*4.;
  double exterior_size_vert = padding_multiplier*size;
  double lateral_padding = padding_multiplier*size;

  std::cout << "grid one generating: " << std::endl;
  GridGenerator::plate_with_a_hole(exterior, size * 3. / 2, size * 2.,
                                   exterior_size_vert / 2., exterior_size_vert / 2.,
                                   exterior_size / 2., exterior_size / 2.,
                                   Point<2>(0., 0.));

  std::cout << "grid two generating: " << std::endl;
  GridGenerator::hyper_ball_balanced(interior, Point<2>(0., 0.),
                                     size / 2.);

  std::cout << "grid three generating: " << std::endl;
  GridGenerator::hyper_shell(shell, Point<2>(0., 0.), size / 2.,
                             size * 3. / 2., 8);

  std::cout << "merged grid generating: " << std::endl;
  GridGenerator::merge_triangulations(exterior, shell, triangulation, 1.0e-12,
                                      true); // lose boundary id
  GridGenerator::merge_triangulations(triangulation, interior, triangulation,
                                      1.0e-12, true); // lose boundary id

  double Lxlocal=2.*size*2.+exterior_size;
  double Lylocal=2.*size*2.+exterior_size_vert;

  Tensor<1,2> move_vector;
  
  move_vector[0]=Lxlocal;

  std::cout<<"stating making grid"<<std::endl;
  unsigned int reduced_index=geometry-2000;
  assert(reduced_index>0);
  unsigned int local_geom=(reduced_index-reduced_index % 100)/100;
  std::cout<<"local geom is "<<local_geom<<std::endl;
  unsigned int repeat=reduced_index % 100;

  memorytriangulation.copy_triangulation(triangulation);

  //creating the initial mesh
  for (unsigned int i=1;i<repeat;i++){
    copy2triangulation.clear();
    copy2triangulation.copy_triangulation(memorytriangulation);
    GridTools::shift(i*move_vector, copy2triangulation);
    GridGenerator::merge_triangulations(triangulation, copy2triangulation, triangulation, 1.0e-12,
                                      true); // lose boundary id
  }

  std::vector<unsigned int> repetition(2);
  repetition[0]=padding_multiplier/2.;
  repetition[1]=2+padding_multiplier/2.;
  Point<2> p1;
  p1[0]=-Lxlocal/2.-lateral_padding;
  p1[1]=-Lylocal/2.;
  Point<2> p2;
  p2[0]=-Lxlocal/2.;
  p2[1]=Lylocal/2.;

  bool add_padding=true;
  if(periodic){
    add_padding=false;
  }

  if(add_padding && padding_multiplier != 0.){
    exterior.clear();
    GridGenerator::subdivided_hyper_rectangle(exterior,repetition,p1,p2);
    GridGenerator::merge_triangulations(triangulation, exterior, triangulation,
                                        1.0e-12, true); // lose boundary id

    Point<2> p1bis;
    Point<2> p2bis;

    p1bis[0]=-Lxlocal/2.+repeat*Lxlocal;
    p1bis[1]=-Lylocal/2.;
    p2bis[0]=-Lxlocal/2.+repeat*Lxlocal+lateral_padding;
    p2bis[1]=Lylocal/2.;

    std::cout<<"adding right padding"<<std::endl;

    exterior.clear();
    GridGenerator::subdivided_hyper_rectangle(exterior,repetition,p1bis,p2bis);
    GridGenerator::merge_triangulations(triangulation, exterior, triangulation,
                                        1.0e-12, true); // lose boundary id

    //std::cout<<"added padding"<<std::endl;
  }
  else{
    lateral_padding=0.;
  }

  //creating manifold
  std::vector<SphericalManifold<2>> manifold_list(0);
  
  triangulation.reset_all_manifolds();

  for (unsigned int i=0;i<repeat;i++){
    manifold_list.push_back(SphericalManifold<2>(Point<2>((double)i*Lxlocal, 0.)));
    triangulation.set_manifold(i+2, manifold_list[i]);

    for (const auto &cell : triangulation.active_cell_iterators())
    {
      
      bool should_be_circle = false;
      double x=cell->center()[0];
      double y=cell->center()[1];
      double distance=std::sqrt((x-(double)i*Lxlocal)*(x-(double)i*Lxlocal)+y*y);

      if ( distance < size*3./2. && distance > size /2.)
      {
        should_be_circle = true;
      }
      if (should_be_circle == true){
        cell->set_all_manifold_ids(i+2);
      }
    }
  }
  
  //std::cout<<"added manifolds"<<std::endl;

  //triangulation.refine_global(1);

  double base_size=2.;
  double scaling_factor=1.5;

  //std::cout<<"left="<<-Lxlocal/2.-lateral_padding<<"right"<<-Lxlocal/2. + ((double) repeat )*Lxlocal+lateral_padding<<"repeat"<<repeat<<"\n";

  for (typename Triangulation<2>::active_cell_iterator cell =
           triangulation.begin_active();
       cell != triangulation.end(); ++cell){
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f){
      //std::cout<<cell->face(f)->center()[0] <<std::endl;
      if (cell->face(f)->at_boundary()){
        if (cell->face(f)->center()[0] == -Lxlocal/2.-lateral_padding){
          cell->face(f)->set_boundary_id(4);
          //std::cout<<"left \n";
        }
        else if ( std::abs(cell->face(f)->center()[0] - (-Lxlocal/2. + ((double) repeat )*Lxlocal+lateral_padding)) <= 1.e-10 ){
          cell->face(f)->set_boundary_id(5);
          //std::cout<<"right \n";
          //bool temp=std::abs(cell->face(f)->center()[0] - (-Lxlocal/2. + ((double) repeat )*Lxlocal+lateral_padding)) <= 1.e-10 ;
          //std::cout<<"found right"<<cell->face(f)->center()[0]<<" "<< (-Lxlocal/2. + ((double) repeat )*Lxlocal+lateral_padding)<< " "<<(cell->face(f)->center()[0] - (-Lxlocal/2. + ((double) repeat )*Lxlocal+lateral_padding))<<" "<< temp <<"\n";
        }
        else if (cell->face(f)->center()[1] == -Lylocal/2.) // low
          cell->face(f)->set_boundary_id(6);
        else if (cell->face(f)->center()[1] == Lylocal/2.) // high
          cell->face(f)->set_boundary_id(7);
      }
    }
  }

  central_refinment_repeat(base_size,scaling_factor,number_refining_command,Lxlocal,repeat);
  
  //std::cout<<"added central refinement"<<std::endl;

  GridOut grid_out;
  std::ofstream outputname("grid4.gnuplot");
  //GridOutFlags::Gnuplot gnuplot_flags(false, 1, /*curved_interior_cells*/ true);
  //grid_out.set_flags(gnuplot_flags);
  MappingQGeneric<2> mapping(1);
  grid_out.write_gnuplot(triangulation, outputname, &mapping);
 
}

void Step3::make_grid_multi(){
  std::cout<<"stating making grid"<<std::endl;
  unsigned int reduced_index=geometry-1000;
  assert(reduced_index>0);
  unsigned int local_geom=(reduced_index-reduced_index % 100)/100;
  std::cout<<"local geom is "<<local_geom<<std::endl;

  double Lxlocal;
  double Lylocal;
  if(local_geom==0){
    make_grid_circle();
     Lxlocal=2.*size*4.+exterior_size;
     Lylocal=Lxlocal;
  }
  //else if(local_geom==1){
  //  make_grid_near_square();
  //}
  //else if(local_geom==2){
  //  make_grid_rectangle();
  //}
  else if (local_geom==3){
    make_grid_oval();
    Lxlocal=2.*size*4.+exterior_size;
    Lylocal=Lxlocal*size2/size;
    Lxlocal=Lxlocal*size/size2;
  }
  else{
    Assert(1==0, ExcMessage("geom is not recognised"));
  }

  std::cout<<"geom ini tria ok "<<local_geom<<std::endl;

  GridOut grid_out;
  
  MappingQGeneric<2> mapping(1);
  MappingQGeneric<2> mapping2(1);

  std::ofstream outputname2("grid.vtk");
  std::ifstream inname2("grid.vtk");
  grid_out.write_vtk(triangulation, outputname2);


  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation_after);
  grid_in.read_vtk(inname2);

  std::cout<<"preparing repeat "<<std::endl;
  unsigned int repeat=reduced_index % 100;
  Tensor<1,2> move_vector;
  
  move_vector[0]=Lxlocal;
  memorytriangulation.copy_triangulation(triangulation);

  for (unsigned int i=1;i<repeat;i++){
    std::ofstream outputname("gridmulti.vtk");
    std::ifstream inputname("gridmulti.vtk");
    copy2triangulation.clear();
    copy2triangulation.copy_triangulation(memorytriangulation);
    GridTools::shift(i*move_vector, copy2triangulation);
    grid_out.write_vtk(copy2triangulation, outputname);
    GridIn<2> grid_in2;
    grid_in2.attach_triangulation(copy2triangulation_after);
    grid_in2.read_vtk(inputname);

    GridGenerator::merge_triangulations(triangulation_after, copy2triangulation_after, triangulation_after, 1.0e-12,
                                      true); // lose boundary id
  }

  triangulation.clear();
  triangulation.copy_triangulation(triangulation_after);

  std::cout<<"test length"<<Lxlocal<<" "<<Lylocal<<" "<<repeat<<std::endl;

  for (typename Triangulation<2>::active_cell_iterator cell =
           triangulation.begin_active();
       cell != triangulation.end(); ++cell)
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        if (cell->face(f)->center()[0] == -Lxlocal/2.) // left
          cell->face(f)->set_boundary_id(0);
        else if (cell->face(f)->center()[0] == Lxlocal/2. + repeat*Lxlocal) // right
          cell->face(f)->set_boundary_id(1);
        else if (cell->face(f)->center()[1] == -Lylocal/2.) // low
          cell->face(f)->set_boundary_id(2);
        else if (cell->face(f)->center()[1] == Lylocal/2.) // high
          cell->face(f)->set_boundary_id(3);
}

void Step3::make_grid_oval()
{
  total_refinment = refinement_level;

  std::cout << "grid one generating: " << std::endl;
  GridGenerator::plate_with_a_hole(exterior, size * 3. / 2, size * 4.,
                                   exterior_size / 2., exterior_size / 2.,
                                   exterior_size / 2., exterior_size / 2.,
                                   Point<2>(0., 0.));

  std::cout << "grid two generating: " << std::endl;
  GridGenerator::hyper_ball_balanced(interior, Point<2>(0., 0.),
                                     size / 2.);

  std::cout << "grid three generating: " << std::endl;
  GridGenerator::hyper_shell(shell, Point<2>(0., 0.), size / 2.,
                             size * 3. / 2., 8);

  std::cout << "merged grid generating: " << std::endl;
  GridGenerator::merge_triangulations(exterior, shell, triangulation, 1.0e-12,
                                      true); // lose boundary id
  GridGenerator::merge_triangulations(triangulation, interior, triangulation,
                                      1.0e-12, true); // lose boundary id

  const SphericalManifold<2> manifold(Point<2>(0., 0.));
  triangulation.set_manifold(0, manifold);

  for (const auto &cell : triangulation.active_cell_iterators())
  {
    bool should_be_circle = false;
    if (cell->center().norm() > size / 2. &&
        cell->center().norm() < size*3./2.)
    {
      should_be_circle = true;
    }
    if (should_be_circle == true)
      // The Triangulation already has a SphericalManifold with
      // manifold id 0 (see the documentation of
      // GridGenerator::hyper_ball) so we just attach it to the outer
      // ring here:
      cell->set_all_manifold_ids(0);
  }

  triangulation.refine_global(refinement_level);

  double base_size=5.;
  double scaling_factor=1.5;

  central_refinment(base_size,scaling_factor,number_refining_command);

  /*for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->center().norm() <  inner_radius*1.5 && cell->center().norm() >
  inner_radius*0.5)
      //cell->set_refine_flag(RefinementCase<2>::cut_x);
      if(cell->center()[0]>cell->center()[1]){
        cell->set_refine_flag(RefinementCase<2>::cut_x);
      }
      else{
        cell->set_refine_flag(RefinementCase<2>::cut_y);
      }
  triangulation.execute_coarsening_and_refinement();
  total_refinment++;*/


  std::cout << (1. - 1. / std::pow(2., (double)total_refinment+2.)) << std::endl;

  GridOut grid_out;
  std::ofstream outputname("grid4.gnuplot");
  GridOutFlags::Gnuplot gnuplot_flags(false, 5, /*curved_interior_cells*/ true);
  grid_out.set_flags(gnuplot_flags);
  MappingQGeneric<2> mapping(3);
  grid_out.write_gnuplot(triangulation, outputname, &mapping);

  //std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

  for (typename Triangulation<2>::active_cell_iterator cell =
           triangulation.begin_active();
       cell != triangulation.end(); ++cell)
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        if (std::abs(cell->face(f)->center()[0] -( -exterior_size / 2. - 4.*size)<1.e-12)){// left
          cell->face(f)->set_boundary_id(4);
          //std::cout<<"left \n";
        }
        else if (std::abs(cell->face(f)->center()[0] +( -exterior_size / 2. - 4.*size)<1.e-12)){// right
          cell->face(f)->set_boundary_id(5);
          //std::cout<<"right \n";
        }
        else if (std::abs(cell->face(f)->center()[1] -( -exterior_size / 2. - 4.*size)<1.e-12)) // low
          cell->face(f)->set_boundary_id(6);
        else if (std::abs(cell->face(f)->center()[1] +( -exterior_size / 2. - 4.*size)<1.e-12)) // high
          cell->face(f)->set_boundary_id(7);

  std::cout<<size/size2<<" "<<size2/size<<std::endl;
  GridTools::transform(Scalexy(size/size2,size2/size), triangulation);
}

void Step3::make_grid_circle()
{
  total_refinment = refinement_level;
  std::cout<<exterior_size<<" "<<size<<" "<<std::endl;
  std::cout << "grid one generating: " << std::endl;
  GridGenerator::plate_with_a_hole(exterior, size * 3. / 2, size * 4.,
                                   exterior_size / 2., exterior_size / 2.,
                                   exterior_size / 2., exterior_size / 2.,
                                   Point<2>(0., 0.));

  std::cout << "grid two generating: " << std::endl;
  GridGenerator::hyper_ball_balanced(interior, Point<2>(0., 0.),
                                     size / 2.);

  std::cout << "grid three generating: " << std::endl;
  GridGenerator::hyper_shell(shell, Point<2>(0., 0.), size / 2.,
                             size * 3. / 2., 8);

  std::cout << "merged grid generating: " << std::endl;
  GridGenerator::merge_triangulations(exterior, shell, triangulation, 1.0e-12,
                                      true); // lose boundary id
  GridGenerator::merge_triangulations(triangulation, interior, triangulation,
                                      1.0e-12, true); // lose boundary id

  const SphericalManifold<2> manifold(Point<2>(0., 0.));
  triangulation.set_manifold(0, manifold);

  for (const auto &cell : triangulation.active_cell_iterators())
  {
    bool should_be_circle = false;
    if (cell->center().norm() > size / 2. &&
        cell->center().norm() < size*3./2.)
    {
      should_be_circle = true;
    }
    if (should_be_circle == true)
      cell->set_all_manifold_ids(0);
  }

  triangulation.refine_global(refinement_level);

  double base_size=5.;
  double scaling_factor=1.5;

  central_refinment(base_size,scaling_factor,number_refining_command);

  /*for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->center().norm() <  inner_radius*1.5 && cell->center().norm() >
  inner_radius*0.5)
      //cell->set_refine_flag(RefinementCase<2>::cut_x);
      if(cell->center()[0]>cell->center()[1]){
        cell->set_refine_flag(RefinementCase<2>::cut_x);
      }
      else{
        cell->set_refine_flag(RefinementCase<2>::cut_y);
      }
  triangulation.execute_coarsening_and_refinement();
  total_refinment++;*/


  std::cout << (1. - 1. / std::pow(2., (double)total_refinment+2.)) << std::endl;

  GridOut grid_out;
  std::ofstream outputname("grid4.gnuplot");
  GridOutFlags::Gnuplot gnuplot_flags(false, 5, /*curved_interior_cells*/ true);
  grid_out.set_flags(gnuplot_flags);
  MappingQGeneric<2> mapping(3);
  grid_out.write_gnuplot(triangulation, outputname, &mapping);

  //std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

  for (typename Triangulation<2>::active_cell_iterator cell =
           triangulation.begin_active();
       cell != triangulation.end(); ++cell)
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        if (cell->face(f)->center()[0] == -(exterior_size / 2.+4.*size)) {
          cell->face(f)->set_boundary_id(4);
          //std::cout<<"left boundary found"<<std::endl;
        }
          // left
        else if (cell->face(f)->center()[0] == (exterior_size / 2.+4.*size)) // right
          cell->face(f)->set_boundary_id(5);
        else if (cell->face(f)->center()[1] == -(exterior_size / 2.+4.*size)) // low
          cell->face(f)->set_boundary_id(6);
        else if (cell->face(f)->center()[1] == (exterior_size / 2.+4.*size)) // high
          cell->face(f)->set_boundary_id(7);
}

void Step3::make_grid_square()
{
  total_refinment = refinement_level;

  GridGenerator::subdivided_hyper_cube(triangulation, size_multiplicator,-  size_multiplicator * size /2., size_multiplicator * size /2.,true);

  triangulation.refine_global(refinement_level);

  double base_size=5.;
  double scaling_factor=1.5;

  central_refinment(base_size,scaling_factor,number_refining_command);
}

Point<2> position(double x, double y, double size, double epsilon){
  if(std::abs(x*x+y*y-2.*size*size)<1.e-10){
    return {x*epsilon,y*epsilon};
  }
  else{
    return {x,y};
    }
}

void Step3::make_grid_near_square()
{
  total_refinment = refinement_level;

  GridGenerator::subdivided_hyper_cube(triangulation, size_multiplicator,-  size_multiplicator * size /2., size_multiplicator * size /2.,true);

  triangulation.refine_global(refinement_level+1);
  total_refinment+=1;
  
  double base_size=5.;
  double scaling_factor=1.5;

  //central_refinment(base_size,scaling_factor,number_refining_command);

  triangulation.reset_all_manifolds();
  double epsilon=(1.-std::pow(2.,-(double)total_refinment-2.));
  std::cout<<"epsilon = "<<epsilon<<std::endl;
  GridTools::transform([&](const Point<2> &in){return position(in[0],in[1],size,epsilon);}, triangulation);

  GridOut grid_out;
  std::ofstream outputname("grid4.gnuplot");
  MappingQGeneric<2> mapping(1);
  grid_out.write_gnuplot(triangulation, outputname, &mapping);

  const SphericalManifold<2> manifold(Point<2>((1.-std::pow(2.,-(double)total_refinment))*size*0.9,(1.-std::pow(2.,-(double)total_refinment))*size*0.9));
  std::cout<<"center "<<(1.-std::pow(2.,-(double)total_refinment))*size*0.9<<std::endl;
  std::cout<<"min criteria "<<size-size*(std::pow(2.,-(double)total_refinment))<<std::endl;
  std::cout<<"max criteria"<<size+size*(std::pow(2.,-(double)total_refinment))<<std::endl;
  triangulation.set_manifold(0, manifold);

  for (const auto &cell : triangulation.active_cell_iterators())
  {
    bool should_be_circle = false;
    double x = cell->center()[0];
    double y = cell->center()[1];

    if (std::abs(x-size) < size*(std::pow(2.,-(double)total_refinment)) &&
        std::abs(y-size) < size*(std::pow(2.,-(double)total_refinment)) )
    {
      should_be_circle = true;
    }
    if (should_be_circle == true)
      cell->set_all_manifold_ids(0);
  }
  
  triangulation.refine_global(1);
  total_refinment+=1;

  central_refinment(2.,scaling_factor,1);

  GridOut grid_out2;
  std::ofstream outputname2("grid.gnuplot");
  MappingQGeneric<2> mapping2(1);
  grid_out2.write_gnuplot(triangulation, outputname2, &mapping2);

}

void Step3::make_grid_rectangle()
{
  total_refinment = refinement_level;
  std::vector<unsigned int> repetition(2);
  Point<2> p1;
  Point<2> p2;

  repetition[0]=(unsigned int) size_multiplicator;
  repetition[1]=(unsigned int) size_multiplicator;

  p1[0]=-size/2.*size_multiplicator;
  p1[1]=-size2/2.*size_multiplicator;
  p2[0]=size/2.*size_multiplicator;
  p2[1]=size2/2.*size_multiplicator;

  GridGenerator::subdivided_hyper_rectangle(triangulation,repetition,p1,p2,true);

  double base_size=5.;
  double scaling_factor=1.5;

  central_refinment(base_size,scaling_factor,number_refining_command);
}

void Step3::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  std::cout<< "is_alternative_current setup" << is_alternative_current << std::endl;

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  const std::vector<bool> truevector(1, true);
  ComponentMask truemask(truevector);
  //std::cout<<"periodicity to be added \n";
  if(periodic){
    DoFTools::make_periodicity_constraints<2,2>(dof_handler,4,5,0,constraints);
    //std::cout<<"periodicity added \n";
  }
  else{
    VectorTools::interpolate_boundary_values(
        dof_handler, 4, Functions::ZeroFunction<2>(), constraints);//left

    VectorTools::interpolate_boundary_values(
        dof_handler, 5, Functions::ZeroFunction<2>(), constraints);//right
  }

  VectorTools::interpolate_boundary_values(
      dof_handler, 6, Functions::ZeroFunction<2>(), constraints);//low

  VectorTools::interpolate_boundary_values(
      dof_handler, 7, Functions::ZeroFunction<2>(), constraints);//high

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  if (initiate_solution)
  {
    initial_solution.reinit(dof_handler.n_dofs());
    initiate_solution = false;
  }

  system_rhs.reinit(dof_handler.n_dofs());
}

void Step3::assemble_system()
{

  QGauss<2> quadrature_formula(fe.degree + 1);

  const unsigned int n_q_points = quadrature_formula.size();

  FEValues<2> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values |
                            update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<Tensor<1, 2>> initial_grad_A_total(n_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.get_function_gradients(initial_solution, initial_grad_A_total);

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {

      double x_coord = fe_values.quadrature_point(q_index)[0];
      double y_coord = fe_values.quadrature_point(q_index)[1];

      // compute H_ini
      double j_local;
      if(is_alternative_current){
        j_local =j_alternate(x_coord, y_coord, current_density,geometry,size,size2,exterior_size);
      }
      else{
        j_local =j(x_coord, y_coord, current_density,geometry,size,size2,exterior_size);
      }
      
      Vector<double> b0vector = Vector<double>(2);
      b0vector[1] = b0;
      Tensor<1, 2> bf;

      bf[0] = b0vector[0] + initial_grad_A_total[q_index][1];
      bf[1] = b0vector[1] - initial_grad_A_total[q_index][0];
      assert(b0vector[0] == 0);
      assert(b0vector[1] == b0);
      double J1 = bf * bf;

      // compute phi_mag dif
      double phi_J1_local = phi_J1(x_coord, y_coord, J1,geometry,size,size2,exterior_size);
      double phi_J1_J1_local = phi_J1_J1(x_coord, y_coord, J1,geometry,size,size2,exterior_size);

      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
        {
          cell_matrix(i, j) +=
              (4. * phi_J1_J1_local *
                   (bf * curl_from_grad(fe_values.shape_grad(i, q_index))) *
                   (bf * curl_from_grad(fe_values.shape_grad(j, q_index))) +
               2. * (phi_J1_local + 1. / 2. / mu0) *
                   (curl_from_grad(fe_values.shape_grad(i, q_index)) *
                    curl_from_grad(fe_values.shape_grad(j, q_index)))) *
              fe_values.JxW(q_index);
        }
      }
      for (const unsigned int i : fe_values.dof_indices())
      {
        cell_rhs(i) += (-j_local                            // f(x_q)
                        * fe_values.shape_value(i, q_index) // phi_i(x_q)
                        * fe_values.JxW(q_index));          // dx
        cell_rhs(i) +=
            (2. * (1. / 2. / mu0 + phi_J1_local) *
             (bf * curl_from_grad(fe_values.shape_grad(
                       i, q_index)))    // should be equivalent
                                        // to 2.*(1./2./mu0+phi_J1_local)*bf \times
                                        // fe_values.shape_grad(i, q_index)
                                        // grad_phi_i(x_q)
             * fe_values.JxW(q_index)); // dx
      }
    }

    // transfer to system

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }
}

double Step3::get_total_stress(double time)
{
  is_saturated=false;
  QGauss<2> quadrature_formula(fe.degree + 1);
  const unsigned int n_q_points = quadrature_formula.size();

  FEValues<2> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | update_hessians |
                            update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  Tensor<1, 3> cell_force;
  Tensor<1, 3> cell_force_r;
  Tensor<1, 3> total_force;
  Tensor<1, 3> total_force_r;
  Tensor<1, 3> cell_forcev2;
  Tensor<1, 3> cell_forcev2_r;
  Tensor<1, 3> total_forcev2;
  Tensor<1, 3> total_forcev2_r;
  Tensor<1, 3> cell_force_safe;
  Tensor<1, 3> total_force_safe;
  Tensor<1, 3> cell_forcev2_safe;
  Tensor<1, 3> total_forcev2_safe;

  Tensor<1, 3> cell_laplace_force;
  Tensor<1, 3> cell_laplace_force_r;
  Tensor<1, 3> total_laplace_force;
  Tensor<1, 3> total_laplace_force_r;
  Tensor<1, 3> magnetic_field;
  Tensor<1, 3> magnetisation_vector;
  Tensor<1, 3> current_vector;
  Tensor<1, 3> laplacien_vector_A;
  Tensor<2, 3> grad_b;
  Tensor<1, 3> grad_J1;
  Tensor<2, 3> Id_tensor;
  Id_tensor[0][0] = 1.;
  Id_tensor[1][1] = 1.;
  Id_tensor[2][2] = 1.;

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<1, 2>> initial_grad_A_total(n_q_points);
  std::vector<Tensor<2, 2>> initial_hessian_A_total(n_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {

    fe_values.reinit(cell);
    cell_force.clear();
    cell_laplace_force.clear();
    cell_forcev2.clear();
    cell_force_safe.clear();
    cell_forcev2_safe.clear();
    cell_force_r.clear();
    cell_laplace_force_r.clear();
    cell_forcev2_r.clear();

    fe_values.get_function_gradients(initial_solution, initial_grad_A_total);
    fe_values.get_function_hessians(initial_solution, initial_hessian_A_total);

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
      double test_interior=0.;
      double test_left=0.;
      double test_right=0.;
      double x_coord = fe_values.quadrature_point(q_index)[0];
      double y_coord = fe_values.quadrature_point(q_index)[1];

      if(is_in_left(geometry, x_coord,  y_coord, size* (1. - 1. / std::pow(2., (double)total_refinment+2.)),size2*(1. - 1. / std::pow(2., (double)total_refinment+2.)),exterior_size)){
        test_interior=1.;
      }

      if(is_in_left(geometry, x_coord,  y_coord, size,size,exterior_size)){
        test_left=1.;
      }

      if(is_in_number(geometry, x_coord,  y_coord, size,size,exterior_size,1)){
        test_right=1.;
      }

      magnetic_field.clear();
      magnetic_field[0] = initial_grad_A_total[q_index][1];
      magnetic_field[1] = b0 - initial_grad_A_total[q_index][0];
      double J1 = magnetic_field * magnetic_field;
      if(J1>blimite*blimite){
        is_saturated=true;
      }
      // build local magnetisation
      magnetisation_vector.clear();
      magnetisation_vector = magnetisation(x_coord, y_coord, magnetic_field,geometry,size,size2,exterior_size);

      // build current vector
      current_vector.clear();
      current_vector[2] = j(x_coord, y_coord, current_density,geometry,size,size2,exterior_size);

      // builgin laplacien vector of A ()
      Tensor<1, 3> laplacien_vector_A;
      laplacien_vector_A[2] = initial_hessian_A_total[q_index][0][0] +
                              initial_hessian_A_total[q_index][1][1];

      grad_b.clear();
      grad_b[0][0] = initial_hessian_A_total[q_index][1][0];
      grad_b[0][1] = initial_hessian_A_total[q_index][1][1];
      grad_b[1][0] = -initial_hessian_A_total[q_index][0][0];
      grad_b[1][1] = -initial_hessian_A_total[q_index][0][1];

      double chi_local =
          local_magnetic_suceptibility(x_coord, y_coord, magnetic_field,geometry,size,size2,exterior_size);

      grad_J1.clear();
      grad_J1 = 2. * grad_b * magnetic_field;

      cell_force +=
          fe_values.JxW(q_index) *test_left*
          (1. / mu0 / (1. + chi_local) *
               cross_product_3d(-laplacien_vector_A, magnetic_field) +
           magnetisation_vector * grad_b +
           4. * phi_J1_J1(x_coord, y_coord, J1,geometry,size,size2,exterior_size) * (magnetic_field * grad_b) *
               (outer_product(magnetic_field, magnetic_field) -
                J1 * Id_tensor));
      cell_laplace_force += fe_values.JxW(q_index) *test_left*
                            (cross_product_3d(current_vector, magnetic_field));
      cell_forcev2 += fe_values.JxW(q_index) *test_left*
                      (cross_product_3d(current_vector, magnetic_field) +
                       magnetisation_vector * grad_b);

      cell_force_safe +=
          fe_values.JxW(q_index) * test_interior *
          (1. / mu0 / (1. + chi_local) *
               cross_product_3d(-laplacien_vector_A, magnetic_field) +
           magnetisation_vector * grad_b +
           4. * phi_J1_J1(x_coord, y_coord, J1,geometry,size,size2,exterior_size) * (magnetic_field * grad_b) *
               (outer_product(magnetic_field, magnetic_field) -
                J1 * Id_tensor));
      cell_forcev2_safe += fe_values.JxW(q_index) * test_interior *
                           (cross_product_3d(current_vector, magnetic_field) +
                            magnetisation_vector * grad_b);

      cell_force_r +=
          fe_values.JxW(q_index) *test_right*
          (1. / mu0 / (1. + chi_local) *
               cross_product_3d(-laplacien_vector_A, magnetic_field) +
           magnetisation_vector * grad_b +
           4. * phi_J1_J1(x_coord, y_coord, J1,geometry,size,size2,exterior_size) * (magnetic_field * grad_b) *
               (outer_product(magnetic_field, magnetic_field) -
                J1 * Id_tensor));
      cell_laplace_force_r += fe_values.JxW(q_index) *test_right*
                            (cross_product_3d(current_vector, magnetic_field));
      cell_forcev2_r += fe_values.JxW(q_index) *test_right*
                      (cross_product_3d(current_vector, magnetic_field) +
                       magnetisation_vector * grad_b);
    }

    // transfer to system
    total_force += cell_force;
    total_forcev2 += cell_forcev2;
    total_force_safe += cell_force_safe;
    total_forcev2_safe += cell_forcev2_safe;
    total_laplace_force += cell_laplace_force;
    total_force_r += cell_force_r;
    total_forcev2_r += cell_forcev2_r;
    total_laplace_force_r += cell_laplace_force_r;
  }

  double integral_force=get_integral_force(0.);
  std::cout << "get_integral_force = " << integral_force << " \n";

  double integral_force2=get_integral_force_bis(0.);
  std::cout << "get_integral_force2 = " << integral_force2 << " \n";

  std::cout << "total force is fx=" << total_force[0]
            << "  fy=" << total_force[1]
            << "total force v2 is fx=" << total_forcev2[0]
            << "  fy=" << total_forcev2[1]
            << "total force safe is fx=" << total_force_safe[0]
            << "  fy=" << total_force[1]
            << "total force v2 safe is fx=" << total_forcev2_safe[0]
            << "  fy=" << total_forcev2[1]
            << "total laplace force is fx=" << total_laplace_force[0]
            << "  fy=" << total_laplace_force[1]
            << " total refinement = "
            << total_refinment 
            << "test saturation = "<< is_saturated
            <<"elapse time total = "<<time 
            <<"n dof = "<<dof_handler.n_dofs() << " total_force_r[0]="
          << total_force_r[0] << "total_force_r[1]"
          << total_force_r[1] << " total_forcev2_r[0]=" << total_forcev2_r[0] << " total_forcev2_r[1]=" << total_forcev2_r[1] << " total_laplace_force_r[0]="
          << total_laplace_force_r[0] << " total_laplace_force_r[1]=" << total_laplace_force_r[1]<< "is alternative current "<< is_alternative_current<<"size" <<size<<"size2"<<size2 << std::endl;

  myfile << current_density << "," << b0 << "," << total_force[0] << ","
         << total_force[1] << "," << total_forcev2[0] << "," << total_forcev2[1]
         << "," << total_force_safe[0] << "," << total_force_safe[1] << ","
         << total_forcev2_safe[0] << "," << total_forcev2_safe[1] << ","
         << total_laplace_force[0] << "," << total_laplace_force[1]
         << " ," << total_refinment  
         << ","<< is_saturated
         <<","<< time
         <<","<<shape_function
         <<"," <<size_multiplicator
         <<","<<dof_handler.n_dofs() << ","
         << total_force_r[0] << ","
         << total_force_r[1] << "," << total_forcev2_r[0] << "," << total_forcev2_r[1] << ","
         << total_laplace_force_r[0] << "," << total_laplace_force_r[1]<< ","
         << is_alternative_current << ","<<size<<","<<size2 << ","<<integral_force<< ","
         <<integral_force2<< ","<<periodic<<std::endl;

  return total_force[0];
}

double Step3::get_integral_force(double time)
{

  is_saturated=false;

  QGauss<1> face_quadrature_formula(fe.degree + 1);

  FEFaceValues<2> fe_face_values(fe,
                                      face_quadrature_formula,
                                      update_values | update_gradients| update_JxW_values | update_quadrature_points | update_normal_vectors| update_hessians);


  const unsigned int n_face_q_points = face_quadrature_formula.size();

  //double face_force_by_m=0;
  Tensor<1, 3> face_force_by_chi;
  Tensor<1, 3> magnetic_field;
  Tensor<1, 3> magnetisation_vector;
  Tensor<2, 3> Id_tensor;
  //Tensor<2, 3> Sigma_m_by_m;
  Tensor<2, 3> Sigma_m_by_chi;
  Id_tensor[0][0] = 1.;
  Id_tensor[1][1] = 1.;
  Id_tensor[2][2] = 1.;
  Tensor<1, 3> integral_force;

  Tensor<1,2> normal_tensor_2D;
  Tensor<1,3> normal_tensor_3D;
  unsigned int face_count=0;

  std::vector<Tensor<1, 2>> initial_grad_A_total(n_face_q_points);
  std::vector<Tensor<2, 2>> initial_hessian_A_total(n_face_q_points);

  double x_target=2.*size;
  double y_target=2.*size;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if(std::abs(cell->center()[0])>=x_target || std::abs(cell->center()[1])>=y_target ){
      continue;
    }

    for (const auto &face : cell->face_iterators())
    {

      fe_face_values.reinit(cell, face);

      if( !((std::abs(std::abs(face->center()[0])-x_target)<1.e-10 &&  std::abs(face->center()[1])<=y_target*(1.+1.e-10) ) ||
          (std::abs(std::abs(face->center()[1])-y_target)<1.e-10 &&  std::abs(face->center()[0])<=x_target*(1.+1.e-10) )  ) ){
        continue;
      }

      fe_face_values.get_function_gradients(initial_solution, initial_grad_A_total);
      fe_face_values.get_function_hessians(initial_solution, initial_hessian_A_total);

      face_count++;

      for (unsigned int q_index = 0; q_index < n_face_q_points; ++q_index)
      {

        double x_coord = fe_face_values.quadrature_point(q_index)[0];
        double y_coord = fe_face_values.quadrature_point(q_index)[1];

        // build magnetic field
        magnetic_field.clear();
        magnetic_field[0] = initial_grad_A_total[q_index][1];
        magnetic_field[1] = b0 - initial_grad_A_total[q_index][0];
        double J1 = magnetic_field * magnetic_field;

        // build local magnetisation
        magnetisation_vector.clear();
        magnetisation_vector = magnetisation(x_coord, y_coord, magnetic_field,geometry,size,size2,exterior_size);

        double chi_local =
        local_magnetic_suceptibility(x_coord, y_coord, magnetic_field,geometry,size,size2,exterior_size);

        Sigma_m_by_chi.clear();
        //Sigma_m_by_m=
        Sigma_m_by_chi=1./mu0*(outer_product(magnetic_field,magnetic_field)-1./2.*contract<0,0>(magnetic_field,magnetic_field)*Id_tensor);
          //-chi_local/mu0/(1.+chi_local)*(outer_product(magnetic_field,magnetic_field)-1.*Id_tensor);
        normal_tensor_3D.clear();
        normal_tensor_3D[0]=fe_face_values.normal_vector(q_index)[0];
        normal_tensor_3D[1]=fe_face_values.normal_vector(q_index)[1];
        //Rq return the outward vector, thus we need to be in the cell "inside"

        //std::cout<<"magnetic_field"<<magnetic_field[0]<<" "<<magnetic_field[1]<<"normal_tensor"<<fe_face_values.normal_vector(q_index)[0]<<" "<<fe_face_values.normal_vector(q_index)[1]<<
        //"Sigma_m_by_chi"<<Sigma_m_by_chi[0][0]<<" "<<Sigma_m_by_chi[1][1]<<contract<0,1>(normal_tensor_3D,Sigma_m_by_chi) <<std::endl;

        face_force_by_chi += contract<0,1>(normal_tensor_3D,Sigma_m_by_chi) *fe_face_values.JxW(q_index);
      }
      
    }

    // transfer to system
    integral_force += face_force_by_chi;
    face_force_by_chi.clear();
  }

  std::cout<<std::to_string(face_count)<<std::endl;

  return integral_force[0];
}

double Step3::get_integral_force_bis(double time)
{

  is_saturated=false;

  QGauss<1> face_quadrature_formula(fe.degree + 1);


  FEFaceValues<2> fe_face_values(fe,
                                      face_quadrature_formula,
                                      update_values | update_gradients| update_JxW_values | update_quadrature_points | update_normal_vectors| update_hessians);


  const unsigned int n_face_q_points = face_quadrature_formula.size();

  //double face_force_by_m=0;
  Tensor<1, 3> face_force_by_chi;
  Tensor<1, 3> magnetic_field;
  Tensor<1, 3> magnetisation_vector;
  Tensor<2, 3> Id_tensor;
  //Tensor<2, 3> Sigma_m_by_m;
  Tensor<2, 3> Sigma_m_by_chi;
  Id_tensor[0][0] = 1.;
  Id_tensor[1][1] = 1.;
  Id_tensor[2][2] = 1.;
  Tensor<1, 3> integral_force;

  Tensor<1,2> normal_tensor_2D;
  Tensor<1,3> normal_tensor_3D;
  unsigned int face_count=0;

  std::vector<Tensor<1, 2>> initial_grad_A_total(n_face_q_points);
  std::vector<Tensor<2, 2>> initial_hessian_A_total(n_face_q_points);

  double x_target=2.*size;
  double y_target=2.*size;

  double x_center=2.*size * 2.+exterior_size;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if(std::abs(cell->center()[0]-x_center)>=x_target || std::abs(cell->center()[1])>=y_target ){
      continue;
    }

    for (const auto &face : cell->face_iterators())
    {

      fe_face_values.reinit(cell, face);

      if( !((std::abs(std::abs(face->center()[0]-x_center)-x_target)<1.e-10 &&  std::abs(face->center()[1])<=y_target*(1.+1.e-10) ) ||
          (std::abs(std::abs(face->center()[1])-y_target)<1.e-10 &&  std::abs(face->center()[0]-x_center)<=x_target*(1.+1.e-10) )  ) ){
        continue;
      }

      fe_face_values.get_function_gradients(initial_solution, initial_grad_A_total);
      fe_face_values.get_function_hessians(initial_solution, initial_hessian_A_total);

      face_count++;

      for (unsigned int q_index = 0; q_index < n_face_q_points; ++q_index)
      {

        //std::cout<<"x"<<face->center()[0]<<"y"<<face->center()[1]<<std::endl;
        //START COMPUTE



        double x_coord = fe_face_values.quadrature_point(q_index)[0];
        double y_coord = fe_face_values.quadrature_point(q_index)[1];

        // build magnetic field
        magnetic_field.clear();
        magnetic_field[0] = initial_grad_A_total[q_index][1];
        magnetic_field[1] = b0 - initial_grad_A_total[q_index][0];
        double J1 = magnetic_field * magnetic_field;

        // build local magnetisation
        magnetisation_vector.clear();
        magnetisation_vector = magnetisation(x_coord, y_coord, magnetic_field,geometry,size,size2,exterior_size);

        double chi_local =
        local_magnetic_suceptibility(x_coord, y_coord, magnetic_field,geometry,size,size2,exterior_size);

        Sigma_m_by_chi.clear();
        //Sigma_m_by_m=
        Sigma_m_by_chi=1./mu0*(outer_product(magnetic_field,magnetic_field)-1./2.*contract<0,0>(magnetic_field,magnetic_field)*Id_tensor);
          //-chi_local/mu0/(1.+chi_local)*(outer_product(magnetic_field,magnetic_field)-1.*Id_tensor);
        normal_tensor_3D.clear();
        normal_tensor_3D[0]=fe_face_values.normal_vector(q_index)[0];
        normal_tensor_3D[1]=fe_face_values.normal_vector(q_index)[1];

        //std::cout<<"magnetic_field"<<magnetic_field[0]<<" "<<magnetic_field[1]<<"normal_tensor"<<fe_face_values.normal_vector(q_index)[0]<<" "<<fe_face_values.normal_vector(q_index)[1]<<
        //"Sigma_m_by_chi"<<Sigma_m_by_chi[0][0]<<" "<<Sigma_m_by_chi[1][1]<<contract<0,1>(normal_tensor_3D,Sigma_m_by_chi) <<std::endl;

        face_force_by_chi += contract<0,1>(normal_tensor_3D,Sigma_m_by_chi) *fe_face_values.JxW(q_index);
      }
    }

    // transfer to system
    integral_force += face_force_by_chi;
    face_force_by_chi.clear();
  }

  std::cout<<std::to_string(face_count)<<std::endl;
  return integral_force[0];
}


void Step3::solve()
{

  SolverControl solver_control(50000, 1e-6);

  SolverCG<Vector<double>> solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  constraints.distribute(solution);
}

void Step3::solve_44()
{
  std::cout << "solving" << std::endl;

  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);
  constraints.distribute(solution);
}

void Step3::output_results(unsigned int cycle) const
{
  GradientPostprocessor<2> gradient_postprocessor;
  BPostprocessor<2> b_postprocessor = BPostprocessor<2>(b0);
  ForcePostprocessor<2> forcePostprocessor =
      ForcePostprocessor<2>(b0, current_density,geometry,size,size2,exterior_size);
  ForcelambdaPostprocessor<2> forcelambdaPostprocessor=
      ForcelambdaPostprocessor<2>(b0, current_density,geometry,size,size2,exterior_size);
  ForceintPostprocessor<2> forceintPostprocessor =
      ForceintPostprocessor<2>(b0, current_density,geometry,size,size2,exterior_size);
  NonlinearForcePostprocessor<2> nonlinearForcePostprocessor =
      NonlinearForcePostprocessor<2>(b0, current_density,geometry,size,size2,exterior_size);
  MagneticstrengthcylindPostprocessor<2> magneticstrengthcylindPostprocessor =
      MagneticstrengthcylindPostprocessor<2>(b0,geometry,size,size2,exterior_size);
  BcylindricalPostprocessor<2> bcylindricalPostprocessor =
      BcylindricalPostprocessor<2>(b0);
  LaplaceForcePostprocessor<2> laplaceForcePostprocessor =
      LaplaceForcePostprocessor<2>(b0, current_density,geometry,size,size2,exterior_size);
  Curlh_zForcePostprocessor<2> curlh_zForcePostprocessor =
      Curlh_zForcePostprocessor<2>(b0, current_density,geometry,size,size2,exterior_size);
  Curlh_zbisForcePostprocessor<2> curlh_zbisForcePostprocessor =
      Curlh_zbisForcePostprocessor<2>(b0, current_density,geometry,size,size2,exterior_size);

  // Curlh_inplaneForcePostprocessor<2> curlh_inplaneForcePostprocessor =
  // Curlh_inplaneForcePostprocessor<2>(b0, current_density,geometry,size);
  Laplacien_A_Postprocessor<2> laplacien_A_Postprocessor;
  Divb_Postprocessor<2> divb_Postprocessor;
  MagnetisationPostprocessor<2> magnetisationPostprocessor =
      MagnetisationPostprocessor<2>(b0,geometry,size,size2,exterior_size);
  Current_Postprocessor<2> current_Postprocessor =
      Current_Postprocessor<2>(current_density,geometry,size,size2,exterior_size, is_alternative_current);
  MagneticstrengthPostprocessor<2> magneticstrengthPostprocessor =
      MagneticstrengthPostprocessor<2>(b0,geometry,size,size2,exterior_size);
  Chi_Postprocessor<2> chi_Postprocessor = Chi_Postprocessor<2>(b0,geometry,size,size2,exterior_size);
  Bnorm_Postprocessor<2> bnorm_Postprocessor = Bnorm_Postprocessor<2>(b0, current_density);
  Laplacien_A_00_Postprocessor<2> laplacien_A_00_Postprocessor;
  Laplacien_A_10_Postprocessor<2> laplacien_A_10_Postprocessor;
  Laplacien_A_01_Postprocessor<2> laplacien_A_01_Postprocessor;
  Laplacien_A_11_Postprocessor<2> laplacien_A_11_Postprocessor;

  Sigma_Postprocessor<2> sigma_Postprocessor =Sigma_Postprocessor<2>(b0,geometry,size,size2,exterior_size);
  Sigma_Postprocessor_xx<2> sigma_Postprocessor_xx =Sigma_Postprocessor_xx<2>(b0,geometry,size,size2,exterior_size);
  Sigma_Postprocessor_xy<2> sigma_Postprocessor_xy =Sigma_Postprocessor_xy<2>(b0,geometry,size,size2,exterior_size);
  Sigma_Postprocessor_yy<2> sigma_Postprocessor_yy =Sigma_Postprocessor_yy<2>(b0,geometry,size,size2,exterior_size);

  DataOut<2> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(initial_solution, "solution");
  data_out.add_data_vector(initial_solution, gradient_postprocessor);
  data_out.add_data_vector(initial_solution, b_postprocessor);
  data_out.add_data_vector(initial_solution, forcePostprocessor);
  data_out.add_data_vector(initial_solution, forcelambdaPostprocessor);
  data_out.add_data_vector(initial_solution, forceintPostprocessor);
  data_out.add_data_vector(initial_solution, nonlinearForcePostprocessor);
  data_out.add_data_vector(initial_solution, laplaceForcePostprocessor);
  data_out.add_data_vector(initial_solution, curlh_zForcePostprocessor);
  data_out.add_data_vector(initial_solution, curlh_zbisForcePostprocessor);
  
  // data_out.add_data_vector(initial_solution,
  // curlh_inplaneForcePostprocessor);
  
  data_out.add_data_vector(initial_solution, laplacien_A_Postprocessor);
  data_out.add_data_vector(initial_solution, divb_Postprocessor);
  data_out.add_data_vector(initial_solution, magnetisationPostprocessor);
  data_out.add_data_vector(initial_solution, current_Postprocessor);
  data_out.add_data_vector(initial_solution, magneticstrengthPostprocessor);
  data_out.add_data_vector(initial_solution, chi_Postprocessor);

  
  data_out.add_data_vector(initial_solution, bcylindricalPostprocessor);
  data_out.add_data_vector(initial_solution, magneticstrengthcylindPostprocessor);
  data_out.add_data_vector(initial_solution, laplacien_A_00_Postprocessor);
  data_out.add_data_vector(initial_solution, laplacien_A_10_Postprocessor);
  data_out.add_data_vector(initial_solution, laplacien_A_01_Postprocessor);
  data_out.add_data_vector(initial_solution, laplacien_A_11_Postprocessor);
  
  //data_out.add_data_vector(initial_solution, bnorm_Postprocessor);
  data_out.add_data_vector(initial_solution, sigma_Postprocessor_xx);
  data_out.add_data_vector(initial_solution, sigma_Postprocessor_xy);
  data_out.add_data_vector(initial_solution, sigma_Postprocessor_yy);
   
  data_out.build_patches(2);
  std::ofstream output(test_name + "solution" + std::to_string(cycle) + "refinement_level"+std::to_string(total_refinment) + "shape function"+ std::to_string(shape_function)+ "exterior"+std::to_string(size_multiplicator) + "distance"+std::to_string(distance_multiplicator) +  "is alternative current"+ std::to_string(is_alternative_current) + "radius"+ std::to_string(size) + "size2"+ std::to_string(size2) + ".vtk");
  data_out.write_vtk(output);
}

double Step3::run(unsigned int number_refinment_cycles, bool output_control)
{
  make_grid();
  std::string name = "j=" + std::to_string(current_density) +
                     " b0=" + std::to_string(b0);
  std::cout << "starting " << name << std::endl;

  std::cout<< "is_alternative_current run" << is_alternative_current<< std::endl;

  test_name = name;

  auto start_total = std::chrono::system_clock::now();
  auto end_total = std::chrono::system_clock::now();
  std::chrono::duration<double> elapse_time_total;

  for (unsigned int cycle_refinement = 0;
       cycle_refinement < number_refinment_cycles; cycle_refinement++)
  {
    //std::cout << "Refinement Cycle " << cycle_refinement << ':' << std::endl;

    unsigned int number_cycle = 100;
    double norm_limit = 0.0000000000001;
    double norm = 100;
    unsigned int cycle = 0;
    initiate_solution = true;

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapse_time;

    while (cycle < number_cycle && norm > norm_limit)
    {
      //std::cout << "Cycle " << cycle << ':' << std::endl;

      start = std::chrono::system_clock::now();
      setup_system();

      //end = std::chrono::system_clock::now();
      //elapse_time = end - start;
      //std::cout << "elapse time setup" << elapse_time.count() << " s\n";

      //output_results(cycle+cycle_refinement*number_cycle);

      //start = std::chrono::system_clock::now();
      assemble_system();

      //end = std::chrono::system_clock::now();
      //elapse_time = end - start;
      //std::cout << "elapse time assemble" << elapse_time.count() << " s\n";

      //start = std::chrono::system_clock::now();
      solve_44();

      end = std::chrono::system_clock::now();
      elapse_time = end - start;
      
      initial_solution.add(-1. * convergence_param, solution);
      norm = system_rhs.norm_sqr();
      
      //std::cout << "norm = " << norm << " \n";
      std::cout <<"Cycle " << cycle << ':' << "elapse time " << elapse_time.count() << "s, norm = " << norm <<  " \n";

      cycle += 1;
    }
    // refine_grid();
  }

  if (output_control)
  {
    output_results(1000);
  }

  end_total = std::chrono::system_clock::now();
  elapse_time_total = end_total - start_total;

  double fx_total = get_total_stress(elapse_time_total.count());
  return fx_total;
  return 0.;
}

class command_data
{
public:
  double do_a_run(double current_density, double b0, unsigned int shape_function_degree, bool output_control = false, unsigned int geom_in=0,double size_in=1./1000. ,double size2_in=1./1000.,unsigned int number_refining_command_in=3,double size_multiplicator_in=30., double distance_multiplicator_in=0.,bool is_alternative_current_in=false,bool periodic_in=false); 
  void unique_test(unsigned int shape_function_degree,double b_value, double j_value,  bool output_control = false, unsigned int geom_in=0,double size_in=1./1000.,double size2_in=1./1000.,unsigned int number_refining_command_in=3,double size_multiplicator_in=30., double distance_multiplicator_in=0.,bool is_alternative_current_in=false,bool periodic_in=false);
  void test_two_variable_para(unsigned int shape_function_degree,
                              double b_value_max, double j_value_max,
                              unsigned int number_step_b, unsigned int number_step_j,
                              unsigned int initial_number_step_b = 0,
                              unsigned int initial_number_step_j = 0,
                              unsigned int lowcut_j = 0,
                              bool output_control = false
                              , unsigned int geom_in=0,double size_in=1./1000.,double size2_in=1./1000.,unsigned int number_refining_command_in=3,double size_multiplicator_in=30., double distance_multiplicator_in=0.,bool is_alternative_current_in=false,bool periodic_in=false);
  void find_maximum_fixed_b(unsigned int shape_function_degree,
                            double b_value, double j_min,
                            double j_max,
                            double final_step_size = 1.,
                            bool output_control = false
                            , unsigned int geom_in=0,double size_in=1./1000.,double size2_in=1./1000.,unsigned int number_refining_command_in=3,double size_multiplicator_in=30., double distance_multiplicator_in=0.,bool is_alternative_current_in=false,bool periodic_in=false);
};

double command_data::do_a_run(double current_density, double b0, unsigned int shape_function_degree, bool output_control, unsigned int geom_in,double size_in,double size2_in,unsigned int number_refining_command_in,double size_multiplicator_in, double distance_multiplicator_in,bool is_alternative_current_in,bool periodic_in)
{
  //std::cout<<"shape function degree = " <<shape_function_degree<<std::endl;
  Step3 laplace_problem(current_density, b0, shape_function_degree,geom_in,size_in,size2_in,number_refining_command_in,size_multiplicator_in, distance_multiplicator_in,is_alternative_current_in, periodic_in);
  return laplace_problem.run(1, output_control);
}

void command_data::unique_test( unsigned int shape_function_degree,double b_value, double j_value, bool output_control, unsigned int geom_in,double size_in,double size2_in,unsigned int number_refining_command_in,double size_multiplicator_in, double distance_multiplicator_in,bool is_alternative_current_in,bool periodic_in)
{
  std::cout<<"shape function degree = " <<shape_function_degree<<std::endl;
  std::cout<< "is_alternative_current unique test " << is_alternative_current_in<< std::endl;
  Step3 laplace_problem(j_value, b_value, shape_function_degree,geom_in,size_in,size2_in,number_refining_command_in,size_multiplicator_in,distance_multiplicator_in,is_alternative_current_in, periodic_in);
  try{
    laplace_problem.run(1, output_control);
  }
  catch(...){
    myfile<<"failed test with config shape_function=" <<shape_function_degree<<" refinement level = "<<number_refining_command_in <<"size multiplicator = "<<size_multiplicator_in << "distance multiplicator"<< distance_multiplicator_in<< "is alternative current"<< is_alternative_current_in<< std::endl;
  }
}

void command_data::test_two_variable_para(unsigned int shape_function_degree, double b_value_max, double j_value_max,
                                          unsigned int number_step_b, unsigned int number_step_j,
                                          unsigned int initial_number_step_b,
                                          unsigned int initial_number_step_j,
                                          unsigned int lowcut_j,
                                          bool output_control, unsigned int geom_in,double size_in,double size2_in,unsigned int number_refining_command_in,double size_multiplicator_in, double distance_multiplicator_in,bool is_alternative_current_in,bool periodic_in)
{
  std::cout<<shape_function_degree<<std::endl;
  command_data command = command_data();
  Threads::TaskGroup<double> task_group;
  double b0_control_max = b_value_max;
  double current_density_control_max = j_value_max;
  for (unsigned int step_b = initial_number_step_b; step_b < number_step_b;
       step_b++)
  {
    double b0_control = b0_control_max * step_b / number_step_b;
    unsigned int step_j;
    if (initial_number_step_b == step_b)
    {
      step_j = initial_number_step_j;
    }
    else
    {
      step_j = lowcut_j;
    }
    for (; step_j < number_step_j; step_j++)
    {
      std::cout<<"step_b"<<step_b<<std::endl;
          std::cout<<"step_j"<<step_j<<std::endl;
      double current_density_control = current_density_control_max / (double)number_step_j * (double)step_j;
      command_data command = command_data();
      std::cout << current_density_control << " " << b0_control << std::endl;
      task_group += Threads::new_task(&command_data::do_a_run, command, current_density_control, b0_control, shape_function_degree, output_control,geom_in,size_in,size2_in, number_refining_command_in,size_multiplicator_in, distance_multiplicator_in,is_alternative_current_in, periodic_in);
    }
  }
  task_group.join_all();
}

void command_data::find_maximum_fixed_b(unsigned int shape_function_degree,
                                        double b_value, double j_min,
                                        double j_max,
                                        double final_step_size,
                                        bool output_control, unsigned int geom_in,double size_in,double size2_in,unsigned int number_refining_command_in,double size_multiplicator_in, double distance_multiplicator_in,bool is_alternative_current_in,bool periodic_in)
{
  double left = j_min;
  double right = j_max;
  command_data command = command_data();
  std::cout << right << " " << left << " " << right - left << " " << final_step_size << std::endl;
  while (right - left > final_step_size)
  {
    std::cout << "looking for minimum, current step size is " << right - left << std::endl;
    double m1 = left + (right - left) / 3.;
    Threads::Task<double> Threads_m1 = Threads::new_task(&command_data::do_a_run, command, m1, b_value, shape_function_degree, output_control,geom_in,size_in, size2_in,number_refining_command_in,size_multiplicator_in,  distance_multiplicator_in,distance_multiplicator_in, periodic_in);

    double m2 = left + (right - left) / 3. * 2.;
    Threads::Task<double> Threads_m2 = Threads::new_task(&command_data::do_a_run, command,  m2, b_value, shape_function_degree, output_control,geom_in,size_in,size2_in, number_refining_command_in,size_multiplicator_in,  distance_multiplicator_in,distance_multiplicator_in, periodic_in);

    Threads_m1.join();
    Threads_m2.join();

    if (Threads_m1.return_value() < Threads_m2.return_value())
    {
      left = m1;
    }
    else if (Threads_m1.return_value() > Threads_m2.return_value())
    {
      right = m2;
    }
    else
    {
      left = m1;
      right = m2;
    }
  }
  std::cout << "the maximum is between " << left << " and " << right << std::endl;
  myfile << "the maximum is between " << left << " and " << right << std::endl;
}

int main()
{
  deallog.depth_console(2);

  myfile.open("results.csv");
  myfile << "current density,magnetic "
            "field,fx,fy,fxv2,fyv2,fx_safe,fy_safe,fxv2_safe,fyv2_safe,"
            "flaplacex,flaplacey,refinment,is_saturated,elapse_time_total,shapefunction_degree,size multiplicator ,number dof, fx_r,fy_r,fx_safe_r,fy_safe_r,flaplacex_r,flaplacey_r,is alternative current,size,size2,integral_force,integral_force2,is_periodic\n";

  Threads::TaskGroup<void> task_group_command;
  MultithreadInfo::set_thread_limit(3);

  command_data command_main = command_data();

  double radius=1./1000.*1.5;
  
  //Fig total
  
  unsigned int shape_degree = 2;
  double b_max=0.8*2.;
  double j_max=4./2502./mu0/radius*b_max*2.;
  unsigned int b_step=16;
  unsigned int j_step=16;
  std::vector<double> step_list= {0.,1.,2.,4.,8.,16.};
  //std::vector<double> step_list= {0.,1.,2.,4.,16};
  unsigned int number_steps=step_list.size();
  bool print_output=false;
  unsigned int geom = 2002;//2 circular wire
  unsigned int number_refinment=5.;
  double size_mult=0.;
  double distance_mult=30.;

  bool alter_current=true;
  bool periodic=false;

  task_group_command+= Threads::new_task(&command_data::unique_test,command_main,shape_degree,0.1,4./2502./mu0/radius*0.1,
                                          true,2001,radius,radius,number_refinment,0,distance_mult,alter_current,periodic);

  /*task_group_command+= Threads::new_task(&command_data::unique_test,command_main,shape_degree,0.0001,4./2502./mu0/radius*0.0001,
                                          print_output,2001,radius,radius,number_refinment,0,distance_mult,alter_current,periodic);
  task_group_command+= Threads::new_task(&command_data::unique_test,command_main,shape_degree,0.5,4./2502./mu0/radius*0.5*100,
                                          print_output,2001,radius,radius,number_refinment,0,distance_mult,alter_current,periodic);
  task_group_command+= Threads::new_task(&command_data::unique_test,command_main,shape_degree,b_max,j_max,
                                          print_output,2001,radius,radius,number_refinment,0,distance_mult,alter_current,periodic);
                                          */

  //task_group_command+= Threads::new_task(&command_data::unique_test,command_main,shape_degree,b_max,j_max,
  //                                        print_output,2002,radius,radius,number_refinment,0,distance_mult,alter_current,periodic);

  //task_group_command+= Threads::new_task(&command_data::unique_test,command_main,shape_degree,0.,10.*j_max,
  //                                        print_output,2002,radius,radius,number_refinment,0,distance_mult,alter_current,periodic);

  /*myfile << "alter_current=true periodic=false \n";

  for(unsigned int i=0;i<number_steps;i++){
        task_group_command+= Threads::new_task(&command_data::test_two_variable_para,
                                          command_main,shape_degree,b_max,j_max,
                                          b_step,j_step,0,0,0,print_output,geom,
                                          radius,radius,number_refinment,step_list[i],
                                          distance_mult,alter_current,periodic);
  }

  task_group_command.join_all();

  myfile << "alter_current=true periodic=false bis \n";

  for(unsigned int i=0;i<number_steps;i++){
        task_group_command+= Threads::new_task(&command_data::test_two_variable_para,
                                          command_main,shape_degree,b_max,-j_max,
                                          b_step,j_step,0,0,0,print_output,geom,
                                          radius,radius,number_refinment,step_list[i],
                                          distance_mult,alter_current,periodic);
  }

  task_group_command.join_all();

  
  
  
  alter_current=false;

  myfile << "alter_current=false periodic=false \n";

  for(unsigned int i=0;i<number_steps;i++){
        task_group_command+= Threads::new_task(&command_data::test_two_variable_para,
                                          command_main,shape_degree,b_max,j_max,
                                          b_step,j_step,0,0,0,print_output,geom,
                                          radius,radius,number_refinment,step_list[i],
                                          distance_mult,alter_current,periodic);
  }

  task_group_command.join_all();

  
  periodic= true;

  myfile << "alter_current=false periodic=true \n";

  for(unsigned int i=0;i<number_steps;i++){
        task_group_command+= Threads::new_task(&command_data::test_two_variable_para,
                                          command_main,shape_degree,b_max,j_max,
                                          b_step,j_step,0,0,0,print_output,geom,
                                          radius,radius,number_refinment,step_list[i],
                                          distance_mult,alter_current,periodic);
  }

  periodic= true;

  task_group_command.join_all();

  alter_current=true;
  myfile << "alter_current=true periodic=true \n";

  for(unsigned int i=0;i<number_steps;i++){
        task_group_command+= Threads::new_task(&command_data::test_two_variable_para,
                                          command_main,shape_degree,b_max,j_max,
                                          b_step,j_step,0,0,0,print_output,geom,
                                          radius,radius,number_refinment,step_list[i],
                                          distance_mult,alter_current,periodic);
  }

  
  */
  
  

  task_group_command.join_all();

  myfile.close();

  return 0;
}
