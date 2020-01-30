/*
 * PoroElasBR1WG.cc
 * This is the old way.
 *

 *  Created on: Feb 15, 2018
 *      Author: ubuntu
 */

// @sect3{Include files}
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_bernardi_raugel.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/point.h>
#include <deal.II/fe/component_mask.h>


#include <fstream>
#include <iostream>
#include <math.h>


using namespace dealii;


template <int dim>
class PoroElasBR1WG
{
public:
  PoroElasBR1WG ();
  void run ();

private:
  void make_grid();
  void setup_dofs(double t);
  void make_grid_and_dofs ();
  void assemble_system (double t);
  void solve_time_step();
  void postprocess ();
  void postprocess_errors(double t, unsigned int n);
  void postprocess_darcy_velocity(double t);
  void projection_exact_solution(double t);
  void Interp_proj_exact_displ(double t);
  void Interp_proj_exact_solution (double t);
//  void compute_errors (double t);
  void output_results () const;

  Triangulation<dim>   triangulation;

  AffineConstraints<double> constraints;

  FE_RaviartThomas<dim>  fe_rt;
  DoFHandler<dim>      dof_handler_rt;

  FESystem<dim>          fe;
  DoFHandler<dim>      dof_handler;

  FE_DGQ<dim>            fe_dgq;
  DoFHandler<dim>      dof_handler_dgq;

  FE_BernardiRaugel<dim> fe_br;
  DoFHandler<dim>      dof_handler_br;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> solution;
  BlockVector<double> old_solution;
  BlockVector<double> system_rhs;

  Vector<double> old_solution_displacement;
  Vector<double> old_solution_pressure;
  Vector<double> old_solution_pressure_interior;


  Vector<double> div_displacement;  // projection of numerical dilation
  Vector<double> div_num_displacement;


  Vector<double> div_projection;
  Vector<double> pressure_projection;
  Vector<double> intrproj_displ;

  Vector<double> l2_l2_intp_errors_square;
  Vector<double> l2_l2_u_errors_square;
  Vector<double> l2_H1semi_displ_errors_square;
  Vector<double> l2_H1_displ_errors_square;

  BlockVector<double> intrproj_solution;
  BlockVector<double> old_intrproj_solution;

  double       total_time;
  double       time;
  double       time_step;
  unsigned int number_timestep;
  unsigned int timestep_number;
  const double lambda;
  const double mu;
  const double alpha;
  const double capacity;
};

// @sect3{Right hand side, boundary values, and exact solution}
// Coefficient matrix $\mathbf{K}$ is the identity matrix as a test example.
template <int dim>
class Coefficient : public TensorFunction<2, dim>
{
public:
  Coefficient()
    : TensorFunction<2, dim>()
  {}
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<Tensor<2, dim>> &  values) const override;
};

template <int dim>
void Coefficient<dim>::value_list(const std::vector<Point<dim>> &points,
                                  std::vector<Tensor<2, dim>> &  values) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));
  for (unsigned int p = 0; p < points.size(); ++p)
    {
      values[p].clear();
      for (unsigned int d = 0; d < dim; ++d)
//        values[p][d][d] = 1;
    	  values[p][d][d] = pow (10.0, -7.0); // Canteliver prob.
    }

  // Sandwiched low perm. layer
//  for (unsigned int p = 0; p < points.size(); ++p)
//    {
//      values[p].clear();
//      if ((1./4 <= points[p][0]) && (points[p][0]<= 3./4))
//      {
//      for (unsigned int d = 0; d < dim; ++d)
//        values[p][d][d] = pow (10.0, -8.0);
//      }
//      else
//      {
//          for (unsigned int d = 0; d < dim; ++d)
//            values[p][d][d] = 1;
//      }
//    }
}

template <int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution () : Function<dim>(dim+2) {}
  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &value) const;
};

template <int dim>
void
ExactSolution<dim>::vector_value (const Point<dim> &p,
                                  Vector<double>   &values) const
{
  Assert (values.size() == dim+2,
          ExcDimensionMismatch (values.size(), dim+1));

  // sinsin
//  values(0) = sin(M_PI*p[0])*sin(M_PI*p[1])*sin(M_PI/2*this->get_time());
//  values(1) = sin(M_PI*p[0])*sin(M_PI*p[1])*sin(M_PI/2*this->get_time());
//  values(2) = sin(M_PI/2.*this->get_time())*sin(M_PI*p[0])*sin(M_PI*p[1]);
//  values(3) = sin(M_PI/2.*this->get_time())*sin(M_PI*p[0])*sin(M_PI*p[1]);

  //coscos
//  values(0) = 1./(2.*M_PI)*sin(M_PI*p[0])*cos(M_PI*p[1])*sin(M_PI/2*this->get_time());
//  values(1) = 1./(2.*M_PI)*cos(M_PI*p[0])*sin(M_PI*p[1])*sin(M_PI/2*this->get_time());
//  values(2) = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1]);
//  values(3) = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1]);

  //Liu's
//  values(0) = 1./(3.*M_PI)*sin(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2])*sin(M_PI/2.*this->get_time());
//  values(1) = 1./(3.*M_PI)*cos(M_PI*p[0])*sin(M_PI*p[1])*cos(M_PI*p[2])*sin(M_PI/2.*this->get_time());
//  values(2) = 1./(3.*M_PI)*cos(M_PI*p[0])*cos(M_PI*p[1])*sin(M_PI*p[2])*sin(M_PI/2.*this->get_time());
//  values(3) = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2]);
//  values(4) = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2]);

  // Sandwiched and canteliver
  values(0) = 0;
  values(1) = 0;
  values(2) = 0;
  values(3) = 0;
}



template <int dim>
  class DisplacementSolution : public TensorFunction<1,dim>
  {
  public:
	DisplacementSolution () : TensorFunction<1,dim>() {}

    virtual Tensor<1,dim> vector_value (const Point<dim> &p) const;
  };
template <int dim>
Tensor<1,dim>
DisplacementSolution<dim>::vector_value (const Point<dim> &p) const
  {
	Tensor<1,dim> values;

	// sinsin
//	 values[0] = sin(M_PI*p[0])*sin(M_PI*p[1])*sin(M_PI/2*this->get_time());
//	 values[1] = sin(M_PI*p[0])*sin(M_PI*p[1])*sin(M_PI/2*this->get_time());

//coscos
//	  values[0] = 1./(2.*M_PI)*sin(M_PI*p[0])*cos(M_PI*p[1])*sin(M_PI/2*this->get_time());
//	  values[1] = 1./(2.*M_PI)*cos(M_PI*p[0])*sin(M_PI*p[1])*sin(M_PI/2*this->get_time());

	//liu's
//	values[0] = 1./(3.*M_PI)*sin(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2])*sin(M_PI/2.*this->get_time());
//	values[1] = 1./(3.*M_PI)*cos(M_PI*p[0])*sin(M_PI*p[1])*cos(M_PI*p[2])*sin(M_PI/2.*this->get_time());
//	values[2] = 1./(3.*M_PI)*cos(M_PI*p[0])*cos(M_PI*p[1])*sin(M_PI*p[2])*sin(M_PI/2.*this->get_time());

	// Sandwiched prob. and cantilever
	values[0] = 0;
	values[1] = 0;

    return values;
  }

template <int dim>
  class DisplacementGradient : public TensorFunction<2,dim>
  {
  public:
	DisplacementGradient () : TensorFunction<2,dim>() {}

    virtual Tensor<2,dim> vector_value (const Point<dim> &p) const;
  };

template <int dim>
Tensor<2,dim>
DisplacementGradient<dim>::vector_value (const Point<dim> &p) const
  {
	Tensor<2,dim> values;

    // coscos
//	values[0][0] = 1./(2*M_PI)*sin(M_PI/2*this->get_time())
//			         *M_PI*cos(M_PI*p[0])*cos(M_PI*p[1]);
//	values[0][1] = -1./(2*M_PI)*sin(M_PI/2*this->get_time())
//				     *M_PI*sin(M_PI*p[0])*sin(M_PI*p[1]);
//	values[1][0] = values[0][1];
//	values[1][1] = values[0][0];

	//liu's
//	values[0][0] = 1./(3.*M_PI)*sin(M_PI/2.*this->get_time())
//			     *M_PI*cos(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2]);
//	values[0][1] = -1./(3.*M_PI)*sin(M_PI/2.*this->get_time())
//				  *M_PI*sin(M_PI*p[0])*sin(M_PI*p[1])*cos(M_PI*p[2]);
//	values[0][2] = -1./(3.*M_PI)*sin(M_PI/2.*this->get_time())
//				  *M_PI*sin(M_PI*p[0])*cos(M_PI*p[1])*sin(M_PI*p[2]);
//	values[1][0] = values[0][1];
//	values[1][1] = values[0][0];
//	values[1][2] = -1./(3.*M_PI)*sin(M_PI/2.*this->get_time())
//				  *M_PI*cos(M_PI*p[0])*sin(M_PI*p[1])*sin(M_PI*p[2]);
//	values[2][0] = values[0][2];
//	values[2][1] = values[1][2];
//	values[2][2] = values[0][0];

    return values;
  }

// used to calculate the projection.
template <int dim>
class DisplacementDiv : public Function<dim>
{
public:
	DisplacementDiv()
    : Function<dim>(1)
  {}
  virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim>
double DisplacementDiv<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
  double return_value = 0;

//  return_value = sin(M_PI/2.*this->get_time())*(M_PI*cos(M_PI*p[0])*sin(M_PI*p[1]) + M_PI*sin(M_PI*p[0])*cos(M_PI*p[1]));

  //coscos
//  return_value = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1]);

  //liu's
//  return_value = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2]);
  return return_value;
}

template <int dim>
class PressureSolution : public Function<dim>
{
public:
	PressureSolution()
    : Function<dim>(2)
  {}
  virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim>
double PressureSolution<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double return_value = 0;
//  return_value = sin(M_PI/2.*this->get_time())*sin(M_PI*p[0])*sin(M_PI*p[1]);

//  return_value = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1]);

  //liu's
//  return_value = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2]);
  return return_value;
}

// Right hand side, f and S
template <int dim>
class FluidRightHandSide : public Function<dim>
{
public:
	FluidRightHandSide()
    : Function<dim>(1)
  {}
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const;
};

template <int dim>
double FluidRightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
  double return_value = 0.0;
  const double capacity = 0.0;
  const double alpha = 1.0;

//  return_value = M_PI/2.*cos(M_PI/2.*this->get_time())*(capacity*sin(M_PI*p[0])*sin(M_PI*p[1])
//		        + alpha*M_PI*cos(M_PI*p[0])*sin(M_PI*p[1])+ alpha*M_PI*sin(M_PI*p[0])*cos(M_PI*p[1]))
//				+ 2*M_PI*M_PI*sin(M_PI/2.*this->get_time())*sin(M_PI*p[0])*sin(M_PI*p[1]);

//  return_value = cos(M_PI*p[0])*cos(M_PI*p[1])
//		       *(M_PI/2*cos(M_PI/2.*this->get_time())*(capacity+alpha)
//		       +2*M_PI*M_PI*sin(M_PI/2.*this->get_time()));

  //liu's
//  return_value = ((alpha + capacity)*M_PI/2.*cos(M_PI/2.*this->get_time()) + 3*M_PI*M_PI*sin(M_PI/2.*this->get_time()))*
//		          cos(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2]);

//  return_value = 0;
  return return_value;
}

template <int dim>
class BodyRightHandSide : public TensorFunction<1,dim>
{
public:
  BodyRightHandSide () : TensorFunction<1,dim>() {}
  virtual Tensor<1,dim> value (const Point<dim>   &p) const override;
};

template <int dim>
Tensor<1,dim> BodyRightHandSide<dim>::value (const Point<dim>   &p) const
{
  Tensor<1,dim> return_value;
  const double mu = 1.;
  const double lambda = 1.;
  const double alpha =1.0;

//  return_value[0] = M_PI*M_PI*sin(M_PI/2.*this->get_time())*(sin(M_PI*p[0])*sin(M_PI*p[1])*(3*mu+lambda) -
//		            cos(M_PI*p[0])*cos(M_PI*p[1])*(lambda+mu))+M_PI*sin(M_PI/2.*this->get_time())*alpha*cos(M_PI*p[0])*sin(M_PI*p[1]);
//  return_value[1] = M_PI*M_PI*sin(M_PI/2.*this->get_time())*(sin(M_PI*p[0])*sin(M_PI*p[1])*(3*mu+lambda) -
//                    cos(M_PI*p[0])*cos(M_PI*p[1])*(lambda+mu))+M_PI*sin(M_PI/2.*this->get_time())*alpha*sin(M_PI*p[0])*cos(M_PI*p[1]);

//  return_value[0] = sin(M_PI/2.*this->get_time())*sin(M_PI*p[0])*cos(M_PI*p[1])*M_PI
//		           *(lambda + 2*mu - alpha);
//  return_value[1] = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*sin(M_PI*p[1])*M_PI
//		           *(lambda + 2*mu - alpha);


  //liu's
//  return_value[0] = (lambda + 2*mu - alpha)*M_PI*sin(M_PI/2.*this->get_time())
//                    *sin(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2]);
//  return_value[1] = (lambda + 2*mu - alpha)*M_PI*sin(M_PI/2.*this->get_time())
//                    *cos(M_PI*p[0])*sin(M_PI*p[1])*cos(M_PI*p[2]);
//  return_value[2] = (lambda + 2*mu - alpha)*M_PI*sin(M_PI/2.*this->get_time())
//                    *cos(M_PI*p[0])*cos(M_PI*p[1])*sin(M_PI*p[2]);

  // sandwiched and cantilever
  return_value[0] = 0;
  return_value[1] =	0;
  return return_value;

}

template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues () : Function<dim>(dim+2) {}
//    virtual double value (const Point<dim>   &p,
//                          const unsigned int  component = 0) const;
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };

  template <int dim>
  void
  BoundaryValues<dim>::vector_value (const Point<dim> &p,
                                     Vector<double>   &values) const
  {
	  // sinsin
//	  values(0) = sin(M_PI/2.*this->get_time())*sin(M_PI*p[0])*sin(M_PI*p[1]);
//	  values(1) = sin(M_PI/2.*this->get_time())*sin(M_PI*p[0])*sin(M_PI*p[1]);
//	  values(2) = sin(M_PI/2.*this->get_time())*sin(M_PI*p[0])*sin(M_PI*p[1]);
//	  values(3) = sin(M_PI/2.*this->get_time())*sin(M_PI*p[0])*sin(M_PI*p[1]);

	  // coscos
//	  values(0) = 1./(2.*M_PI)*sin(M_PI*p[0])*cos(M_PI*p[1])*sin(M_PI/2*this->get_time());
//	  values(1) = 1./(2.*M_PI)*cos(M_PI*p[0])*sin(M_PI*p[1])*sin(M_PI/2*this->get_time());
//	  values(2) = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1]);
//	  values(3) = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1]);

	  //liu's
//	  values(0) = 1./(3.*M_PI)*sin(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2])*sin(M_PI/2.*this->get_time());
//	  values(1) = 1./(3.*M_PI)*cos(M_PI*p[0])*sin(M_PI*p[1])*cos(M_PI*p[2])*sin(M_PI/2.*this->get_time());
//	  values(2) = 1./(3.*M_PI)*cos(M_PI*p[0])*cos(M_PI*p[1])*sin(M_PI*p[2])*sin(M_PI/2.*this->get_time());
//	  values(3) = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2]);
//	  values(4) = sin(M_PI/2.*this->get_time())*cos(M_PI*p[0])*cos(M_PI*p[1])*cos(M_PI*p[2]);

// Sandwiched prob. and canteliver
	  values(0) = 0;
	  values(1) = 0;
	  values(2) = 0;
	  values(3) = 0;

  }


template <int dim>
  class InitialValuesPressure : public Function<dim>
  {
  public:
    InitialValuesPressure () : Function<dim>(1) {}  //?? is it dim+2?

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };

  template <int dim>
  double
  InitialValuesPressure<dim>::value (const Point<dim>  &p,
                             const unsigned int component ) const
  {
    return ZeroFunction<dim>().value (p, component);
  }

  template <int dim>
  class InitialValuesDisplacement : public Function<dim>
  {
    public:
      InitialValuesDisplacement () : Function<dim>(dim) {}

      virtual void vector_value (const Point<dim> &p,
                                 Vector<double>   &value) const;
  };

  template <int dim>
  void
  InitialValuesDisplacement<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    ZeroFunction<dim>(dim).vector_value (p, values);
  }


 template <int dim>
 PoroElasBR1WG<dim>::PoroElasBR1WG ()
   :
   fe_rt (0),
   dof_handler_rt (triangulation), // for discrete weak gradient

   fe (FE_BernardiRaugel<dim>(1),1,  // what's for dim? 1? 2?
	   FE_DGQ<dim>(0), 1,
	   FE_FaceQ<dim>(0), 1),

   dof_handler (triangulation),

   fe_dgq(0),       // for projection
   dof_handler_dgq (triangulation),

   fe_br(1),
   dof_handler_br (triangulation),

//   refinement_level(n),

// general case
//   total_time(1.0),
//   time_step(1./16),  // delta t
//   timestep_number (1), // start from the first timestep
//   number_timestep(total_time/time_step),
//   lambda (1.),
//   mu (1.),
//   alpha (1.0),
//   capacity (0.0)

   // For canteliver bracket
   // In this cantilever bracket problem,
   // change coefficient, exactsolution, f, s,
   // set N.B.C displacement in make_grid_and_dofs;
   // in assemble_system part, set displacement N.B.C to local_rhs,
   // by now, I don't have the part to deal with pressure 0 N.B.C,
   // it should be in the assemble_system part, but since it is 0,
   // it doesn't affect results.
   // Then, in constraints part, no need to assign pressure D.B.C,
   // remember to delele pressure D.B.C. there.
   total_time(1),
   time_step(0.01),  // delta t
   number_timestep(total_time/time_step),
   timestep_number (1), // start from the first timestep
   lambda (1./7*pow(10.0,6.0)),
   mu (1.0*pow(10.0,6.0)/28.0),
   alpha (0.93),
   capacity(0.0)

   // For Sandwiched prob
   // In Sandwiched prob.,
   // change coefficient, exact solution, f, s;
   // set the pressure D.B.C edge to be with id=1 in make_grid_and_dofs;
   // in assemble_system, set pressure N.B.C and displ. N.B.C separately;
   // in assemble_system, change id = 0 to id = 1 in pressure D.B.C constraints;
   // in run, change time boundary to be time < 0.011.
//   total_time(0.01),
//   time_step(0.001),
//   number_timestep(total_time/time_step),
//   timestep_number (1),
//   lambda (1.),
//   mu (1.),
//   alpha (1.0),
//   capacity (0.0)


 {}


 template <int dim>
  void PoroElasBR1WG<dim>::make_grid_and_dofs ()
  {
    GridGenerator::hyper_cube (triangulation, 0, 1);
    triangulation.refine_global (5);

    dof_handler_rt.distribute_dofs (fe_rt);
    dof_handler.distribute_dofs(fe);
    dof_handler_dgq.distribute_dofs(fe_dgq);
    dof_handler_br.distribute_dofs(fe_br);

    DoFRenumbering::component_wise (dof_handler);
    std::vector<types::global_dof_index> dofs_per_component (dim+2);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    const unsigned int n_u = dofs_per_component[0],
                       n_p_interior = dofs_per_component[dim],
                       n_p_face =  dofs_per_component[dim+1],
                       n_p = dofs_per_component[dim]+ dofs_per_component[dim+1];

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p_interior << '+'<< n_p_face <<')'
              << std::endl
              << std::endl;

    std::cout << "   Number of rt degrees of freedom: "
              << dof_handler_rt.n_dofs()
              << std::endl;

    BlockDynamicSparsityPattern dsp(2, 2);
    dsp.block(0, 0).reinit (n_u, n_u);
    dsp.block(1, 0).reinit (n_p, n_u);
    dsp.block(0, 1).reinit (n_u, n_p);
    dsp.block(1, 1).reinit (n_p, n_p);
    dsp.collect_sizes ();
    DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit (sparsity_pattern);
 //
    solution.reinit (2);
    solution.block(0).reinit (n_u);
    solution.block(1).reinit (n_p);
    solution.collect_sizes ();

    old_solution.reinit (2);
    old_solution.block(0).reinit (n_u);
    old_solution.block(1).reinit (n_p);
    old_solution.collect_sizes ();

    intrproj_solution.reinit (2);
    intrproj_solution.block(0).reinit (n_u);
    intrproj_solution.block(1).reinit (n_p);
    intrproj_solution.collect_sizes ();

    old_intrproj_solution.reinit (2);
    old_intrproj_solution.block(0).reinit (n_u);
    old_intrproj_solution.block(1).reinit (n_p);
    old_intrproj_solution.collect_sizes ();


    system_rhs.reinit (2);
    system_rhs.block(0).reinit (n_u);
    system_rhs.block(1).reinit (n_p);
    system_rhs.collect_sizes ();


       // set N.B.C edge for Canteliver prob.
       // this is to set displacement N.B.C.
          typename Triangulation<dim>::cell_iterator
          cell = triangulation.begin (),
          endc = triangulation.end();
          for (; cell!=endc; ++cell)
             {
             for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                {
                  if ((std::fabs(cell->face(face_number)->center()(1) - (0)) < 1e-12)
                      ||
                      (std::fabs(cell->face(face_number)->center()(1) - (1)) < 1e-12)
					  ||
					  (std::fabs(cell->face(face_number)->center()(0) - (1)) < 1e-12))
                       cell->face(face_number)->set_boundary_id (1);
                 }
             }

           // set N.B.C edge for Sandwiched prob.
//              typename Triangulation<dim>::cell_iterator
//              cell = triangulation.begin (),
//              endc = triangulation.end();
//              for (; cell!=endc; ++cell)
//                 {
//                 for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
//                    {
//                      if ((std::fabs(cell->face(face_number)->center()(0) - (0)) < 1e-12))
//                           cell->face(face_number)->set_boundary_id (1);
//                     }
//                 }

  }

// To assemble local matrices and local rhs.
 template <int dim>
 void PoroElasBR1WG<dim>::assemble_system (double t)
 {
   system_matrix=0;
   system_rhs=0;

   QGauss<dim>  quadrature_formula(3);
   QGauss<dim-1>  face_quadrature_formula(3);

   FEValues<dim> fe_values_rt (fe_rt, quadrature_formula,
                               update_values   | update_gradients |
                               update_quadrature_points | update_JxW_values);

   FEValues<dim> fe_values (fe, quadrature_formula,
                              update_values   |
                              update_quadrature_points |  update_gradients |
                              update_JxW_values);

   FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                     update_values   | update_normal_vectors |
                                     update_quadrature_points |
                                     update_JxW_values);

   FEFaceValues<dim> fe_face_values_rt (fe_rt, face_quadrature_formula,
                                        update_values   | update_normal_vectors |
                                        update_quadrature_points | update_JxW_values);


   const unsigned int   dofs_per_cell_rt = fe_rt.dofs_per_cell;
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;
   const unsigned int   dofs_per_face = fe.dofs_per_face;
   const unsigned int   dofs_per_cell_u = fe.base_element(0).dofs_per_cell;
   const unsigned int   dofs_per_cell_p = fe.base_element(1).dofs_per_cell + fe.base_element(2).dofs_per_cell;


   const unsigned int   n_q_points    = fe_values.get_quadrature().size();
   const unsigned int   n_q_points_rt = fe_values_rt.get_quadrature().size();
   const unsigned int   n_face_q_points = fe_face_values.get_quadrature().size();


   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
   std::vector<types::global_dof_index> face_dof_indices(dofs_per_face);
   std::vector<unsigned int> face_to_local_indices(dofs_per_face);


   std::vector<types::global_dof_index> interior_local_dof_indices (fe.base_element(1).dofs_per_cell);
   DoFHandler<dim> interior_dof_handler (triangulation);
   interior_dof_handler.distribute_dofs (fe.base_element(1));

   DoFHandler<dim> u_dof_handler (triangulation);
   u_dof_handler.distribute_dofs (fe.base_element(0));

   // We will construct these cell matrices to solve for the pressure.
   FullMatrix<double>   cell_matrix_rt (dofs_per_cell_rt,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_F (dofs_per_cell,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_C (dofs_per_cell,dofs_per_cell_rt);
   FullMatrix<double>   local_matrix (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   local_matrix_p (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   local_matrix_u (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   cell_matrix_D (dofs_per_cell_rt,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_E (dofs_per_cell_rt,dofs_per_cell_rt);
   Vector<double>       local_rhs (dofs_per_cell);
   Vector<double>       cell_solution (dofs_per_cell);
   Vector<double>       average_phi_i_div(dofs_per_cell);
   Vector<double>       average_phi_j_div(dofs_per_cell);

   const Coefficient<dim> coefficient;
   std::vector<Tensor<2,dim>> coefficient_values (n_q_points_rt);

   const FEValuesExtractors::Vector velocities (0); // this is for fe_rt velocities.
   const FEValuesExtractors::Vector displacements (0); //
   const FEValuesExtractors::Scalar pressure_interior (dim); // this is for pressure
   const FEValuesExtractors::Scalar pressure_face (dim+1);

   const unsigned int dofs_per_face_u = fe.base_element(0).dofs_per_face;
   const unsigned int dofs_per_cell_intp = fe.base_element(1).dofs_per_cell;
   const unsigned int dofs_per_cell_facep = fe.base_element(2).dofs_per_cell;


   std::vector<Vector<double> > old_solution_values(n_q_points, Vector<double>(dim+2));

   // distribute global dofs of u.
   std::vector<types::global_dof_index> local_dof_indices_u(dofs_per_cell_u);
   std::vector<types::global_dof_index> face_dof_indices_u(dofs_per_face_u);
   std::vector<unsigned int> face_to_local_indices_u(dofs_per_face_u);
   std::vector<unsigned int> local_dof_indices_facep(dofs_per_cell_facep);

   std::vector<types::global_dof_index> local_dof_indices_intp(dofs_per_cell_intp);

   // f and s at this time
   BodyRightHandSide<dim> body_rhs_function;
   body_rhs_function.set_time (t);
   FluidRightHandSide<dim> fluid_rhs_function;
   fluid_rhs_function.set_time (t);
   DisplacementSolution<dim> displacement_exact_solution;
   displacement_exact_solution.set_time (t);
   BoundaryValues<dim> boundary_values;
   boundary_values.set_time(t);

   typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler.begin_active(),
   endc = dof_handler.end(),
   u_cell = u_dof_handler.begin_active(),
   intp_cell = interior_dof_handler.begin_active();
   typename DoFHandler<dim>::active_cell_iterator
   cell_rt = dof_handler_rt.begin_active();

   for (unsigned int index=0; cell!=endc; ++cell,++cell_rt, ++u_cell, ++intp_cell, ++index)
     {
	   // On each cell, cell matrices are different, so in every loop, they need to be re-computed.
       fe_values_rt.reinit (cell_rt);
       fe_values.reinit (cell);

       cell_matrix_rt = 0;
       cell_matrix_F = 0;
       cell_matrix_C = 0;
       local_matrix = 0;
       local_rhs = 0;
       coefficient.value_list (fe_values_rt.get_quadrature_points(),
                               coefficient_values);

       // to extract old solutions values.
       fe_values.get_function_values (old_solution, old_solution_values);

       cell->get_dof_indices (local_dof_indices);

       // This part is used to calculate Darcy block.
       for (unsigned int q=0; q<n_q_points_rt; ++q)
       {

    	   for (unsigned int i=0; i<dofs_per_cell_rt; ++i)
           {
    		 const Tensor<1,dim> phi_i_u = fe_values_rt[velocities].value (i, q);
             for (unsigned int j=0; j<dofs_per_cell_rt; ++j)
              {
               const Tensor<1,dim> phi_j_u = fe_values_rt[velocities].value (j, q);
               cell_matrix_rt(i,j) += (
            		                   phi_i_u * phi_j_u *
            		                   fe_values_rt.JxW(q));
              }
           }
        }
       cell_matrix_rt.gauss_jordan();

   	   for (unsigned int q=0; q<n_q_points; ++q)
   	       {
   	         for (unsigned int i=0; i<dofs_per_cell; ++i)
   	             {
   	         	   for (unsigned int k=0; k<dofs_per_cell_rt; ++k)
   	         	       {
   	         		     const double phi_k_u_div = fe_values_rt[velocities].divergence(k,q);
   	                     cell_matrix_F(i,k) -= (fe_values[pressure_interior].value(i,q) *
   	                                            phi_k_u_div *
   	                                            fe_values.JxW (q));
   	         	       }
   	              }
   	        }

       for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
           {
           	   fe_face_values.reinit (cell,face_n);
           	   fe_face_values_rt.reinit (cell_rt,face_n);
           	   for (unsigned int q=0; q<n_face_q_points; ++q)
           	       {
       	         	  const Tensor<1,dim> normal = fe_face_values.normal_vector(q);
           	          for (unsigned int i=0; i<dofs_per_cell; ++i)
           	              {
           	         	       for (unsigned int k=0; k<dofs_per_cell_rt; ++k)
           	                       {
           	                          const Tensor<1,dim> phi_k_u = fe_face_values_rt[velocities].value (k, q);
           	                          cell_matrix_F(i,k) += (fe_face_values[pressure_face].value(i,q) *
           	                                                (phi_k_u * normal)*
           	                                                 fe_face_values.JxW (q));
           	                        }
           	              }
           	        }
           	 }

     // We calculate @p cell_matrix_C by doing matrix multiplication via <code>SparseMatrix::mmult</code>.
       cell_matrix_F.mmult(cell_matrix_C,cell_matrix_rt);


     // calculate the average of divergence of displacement
       const double cell_area = cell->measure();
       std::vector<double> avg_div(dofs_per_cell, 0);
       for (unsigned int i = 0; i < dofs_per_cell; ++i)
       {
         for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
             {
               avg_div[i] += fe_values[displacements].divergence(i, q_index) *
                             fe_values.JxW(q_index) / cell_area;
             }
       }

       // for delta_t*(K*grad_p*grad_p)
       for (unsigned int q=0; q<n_q_points_rt; ++q)
           {
       	     for(unsigned int i = 0; i<dofs_per_cell; ++i)
       	        {
       		     for(unsigned int j = 0; j<dofs_per_cell; ++j)
       		        {
       			       for(unsigned int k = 0; k<dofs_per_cell_rt; ++k)
       			          {
       				        const Tensor<1,dim> phi_k_u = fe_values_rt[velocities].value (k, q);
       				        for(unsigned int l = 0; l<dofs_per_cell_rt; ++l)
       				           {
       					         const Tensor<1,dim> phi_l_u = fe_values_rt[velocities].value (l, q);
       					         local_matrix(i,j) += (
													   time_step *
													   coefficient_values[q] * cell_matrix_C[i][k] * cell_matrix_C[j][l] *
       							                       phi_k_u * phi_l_u
       							                         ) *
       							                       fe_values.JxW(q);
       				            }
       			           }
       		         }
       	         }
             }

       // for other terms
       for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
       {
         for (unsigned int i = 0; i < dofs_per_cell; ++i)
         {
        	 const Tensor<2,dim> grad_phi_i_u = fe_values[displacements].symmetric_gradient(i, q_index);
           for (unsigned int j = 0; j < dofs_per_cell; ++j)
           {
             const Tensor<2,dim> grad_phi_j_v = fe_values[displacements].symmetric_gradient(j, q_index);
             local_matrix(i, j) += (2. * mu *
            		                scalar_product(grad_phi_i_u, grad_phi_j_v)
                                    + lambda * avg_div[i] * avg_div[j]
									- alpha*fe_values[pressure_interior].value(j, q_index)* avg_div[i]
									+ capacity*(fe_values[pressure_interior].value(i,q_index) * fe_values[pressure_interior].value(j,q_index))
									+ alpha* (avg_div[j] * fe_values[pressure_interior].value (i,q_index)))
                                    * fe_values.JxW(q_index);
//             local_matrix(i, j) += (2. * mu *
//            		                scalar_product(grad_phi_i_u, grad_phi_j_v)
//                                    + lambda * avg_div[i] * avg_div[j])*
//									 fe_values.JxW(q_index);
           }
         }
       }

          // Take divergence of old displacement
          std::vector<double> div_old_displacement (n_q_points);
          fe_values[displacements].get_function_divergences (old_solution,
                    		                                 div_old_displacement);

          // Take average divergence of old displacement.
          // on each cell, there is one scalar value.
          std::vector<double> average_div_old_displacement (1,0);
          average_div_old_displacement[0] = 0;
          for(unsigned int q=0; q<n_q_points; ++q)
             {
            	average_div_old_displacement[0] += div_old_displacement[q]
											     * fe_values.JxW(q)/cell_area;
             }

       // To extract old interior pressure values.
       std::vector<double> old_interior_pressure (n_q_points);
       fe_values[pressure_interior].get_function_values (old_solution,
    		                                         old_interior_pressure);

          for (unsigned int q=0; q<n_q_points; ++q)
              {
              	const double fluid_rhs_value = fluid_rhs_function.value(fe_values.quadrature_point (q));
              	const Tensor<1,dim> body_rhs_value = body_rhs_function.value(fe_values.quadrature_point (q));
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                   {
                     const Tensor<1,dim> phi_i_v = fe_values[displacements].value (i, q);
                     const double phi_i_q = fe_values[pressure_interior].value(i,q);
                     local_rhs(i) += (scalar_product(body_rhs_value, phi_i_v) +
                    		          capacity * old_interior_pressure[q] * phi_i_q +
                      		          time_step * fluid_rhs_value  * phi_i_q +
									  alpha * average_div_old_displacement[0] * phi_i_q) *
                                      fe_values.JxW(q);
                    }
               }

         // Add N.B.C term to right hand side. For Canteliver prob.
             for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
             {
                if (cell->face(face_number)->at_boundary()
                    &&
				   (std::fabs(cell->face(face_number)->center()(1) - (1)) < 1e-12)
				    &&
                   (cell->face(face_number)->boundary_id() == 1))
                      {
                         fe_face_values.reinit (cell, face_number);
                         for (unsigned int q=0; q<n_face_q_points; ++q)
                           {
          //            	   std::cout<<"fe_value "<<fe_face_values.normal_vector(q)<<std::endl;
          //            	   const double neumann_value
          //            	                = -(exact_solution.gradient (fe_face_values.quadrature_point(q)) *
          //            	                   fe_face_values.normal_vector(q));
          //

                        	 Tensor<1, dim> neumann_value_displacement;
                             neumann_value_displacement[0] = 0;
                             neumann_value_displacement[1] = -1;

                             for (unsigned int i=0; i<dofs_per_cell; ++i)
                            	 local_rhs(i) +=  (neumann_value_displacement*
                              		         fe_face_values[displacements].value(i,q) *
                                               fe_face_values.JxW(q));
                           }
                       }
                else if(cell->face(face_number)->at_boundary()
    				    &&
                       (cell->face(face_number)->boundary_id() == 1))
                {
                   fe_face_values.reinit (cell, face_number);
                   for (unsigned int q=0; q<n_face_q_points; ++q)
                     {
    //            	   std::cout<<"fe_value "<<fe_face_values.normal_vector(q)<<std::endl;
    //            	   const double neumann_value
    //            	                = -(exact_solution.gradient (fe_face_values.quadrature_point(q)) *
    //            	                   fe_face_values.normal_vector(q));
    //

                	   Tensor<1, dim> neumann_value_displacement;
                       neumann_value_displacement[0] = 0;
                       neumann_value_displacement[1] = 0;

                       for (unsigned int i=0; i<dofs_per_cell; ++i)
                    	   local_rhs(i) +=  (neumann_value_displacement*
                        		         fe_face_values[displacements].value(i,q) *
                                         fe_face_values.JxW(q));
                     }
                 }

             }

         // Add N.B.C term to right hand side. For Sandwiched prob.
//         for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
//            {
//        	 // displacement N.B.C
//             if (cell->face(face_number)->at_boundary()
//                 &&
//             	(std::fabs(cell->face(face_number)->center()(0) - (0)) < 1e-12))
//                {
//                 fe_face_values.reinit (cell, face_number);
//                 for (unsigned int q=0; q<n_face_q_points; ++q)
//                    {
//                     Tensor<1, dim> neumann_value_displacement;
//                     neumann_value_displacement[0] = 1;
//                     neumann_value_displacement[1] = 0;
//
//                     for (unsigned int i=0; i<dofs_per_cell; ++i)
//                        local_rhs(i) +=  (neumann_value_displacement*
//                                          fe_face_values[displacements].value(i,q) *
//                                          fe_face_values.JxW(q));
//                     }
//                 }
//             // Pressure N.B.C
//             if (cell->face(face_number)->at_boundary()
//                 &&
//                ((std::fabs(cell->face(face_number)->center()(0) - (1)) < 1e-12)
//                 ||
//				 (std::fabs(cell->face(face_number)->center()(1) - (0)) < 1e-12)
//				 ||
//				 (std::fabs(cell->face(face_number)->center()(1) - (1)) < 1e-12)))
//             {
//                 fe_face_values.reinit (cell, face_number);
//                 for (unsigned int q=0; q<n_face_q_points; ++q)
//                 {
//                  const double neumann_value_pressure = 0;
//
//                  for (unsigned int i=0; i<dofs_per_cell; ++i)
//                     local_rhs(i) -=  time_step*(neumann_value_pressure*
//                                       fe_face_values[pressure_face].value(i,q) *
//                                       fe_face_values.JxW(q));
//                  }
//             }
//             }

          // Assign the D.B.C. Separate pressure and displacement.
          {
            constraints.clear();
//            ComponentMask  face_pressure_mask = fe.component_mask(pressure_face); // delete if test Cantilever
//            VectorTools::interpolate_boundary_values(dof_handler,
//                  		                             0,
////            		                                 1, // for Sandwiched pressure D.B.C
//													 boundary_values,  // need to change this?
////													 Functions::ZeroFunction<2>(4),
//          											 constraints,
//          											 face_pressure_mask);


         // Now, for displacement boundary conditions.
         // Linearly interpolate the boundary conditions at every vertex
         // and set the bubble function DoF to 0
            for (unsigned int i=0; i<fe.base_element(2).dofs_per_cell; ++i) // i<dofs_per_cell_facep
            {
            local_dof_indices_facep[i] = local_dof_indices[fe.component_to_system_index(dim+1,i)];
            }
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
             {
            	cell->face(f)->get_dof_indices(face_dof_indices);
            	unsigned int index_i = 0;
            	for(unsigned int i = 0; i<dofs_per_face; ++i)
            	{
            		unsigned int j;
            		for(j = 0; j< dofs_per_cell_facep; ++j)

            			if(face_dof_indices[i] == local_dof_indices_facep[j])
            				break;

            		if(j == dofs_per_cell_facep && index_i < dofs_per_face_u)
            		{
            			face_dof_indices_u[index_i] = face_dof_indices[i];
            			++index_i;
            		}
            	}
            	for(unsigned int i = 0; i<dofs_per_face_u; ++i)
                 for (unsigned int dof_face = 0; dof_face < dofs_per_face_u; ++dof_face)
                 {
                	 for (unsigned int dof_cell = 0; dof_cell < dofs_per_cell; ++dof_cell)
                   {
                	   if( local_dof_indices[dof_cell] == face_dof_indices_u[dof_face] )
                       {face_to_local_indices_u[dof_face] = dof_cell;
                       }
                   }
                 }

                if (cell->at_boundary(f))
                 {
                    fe_face_values.reinit(cell,f);
                    std::vector<double> vertex_values_u(dofs_per_face_u - 1, 0);
                    for(unsigned int dof = 0; dof < dofs_per_face_u - 1; ++dof)
                    {
                        vertex_values_u[dof] = displacement_exact_solution.vector_value
						                       (cell->face(f)->vertex(dof/dim))[dof % dim];

                    }
//
                  if(cell->face(f)->boundary_id()==0)
                  {
                    for (unsigned int dof = 0; dof < dofs_per_face_u - 1; ++dof)
                      {
                    	constraints.add_line(face_dof_indices_u[dof]);
                    	constraints.set_inhomogeneity(face_dof_indices_u[dof], vertex_values_u[dof]);
//                    	std::cout<<"nodes "<<face_dof_indices_u[dof]<<" "<<vertex_values_u[dof]<<std::endl;
                      }

                    	double res_flux= 0;
                    	double bubble_flux = 0;
                    	Tensor<1, dim> normal_vector = fe_face_values.normal_vector(0);
                    	for (unsigned int q_index = 0; q_index < n_face_q_points; ++q_index)
                    	 {
                    	 Tensor<1, dim> phi_i;
                    	 Tensor<1, dim> res = displacement_exact_solution.vector_value
                    			             (fe_face_values.quadrature_point (q_index));

                    	  for (unsigned int d=0; d<dim; ++d)
                    	    {
                    		  for (unsigned int dof = 0; dof < dofs_per_face_u - 1; ++dof)
                    		  {
                    		      res[d] -= vertex_values_u[dof]*
                    		    		    fe_face_values.shape_value_component(face_to_local_indices_u[dof], q_index, d);
                    		  }
                    	    }
                    	  for (unsigned int d=0; d<dim; ++d)
                    	   {
                    		 phi_i[d] = fe_face_values.shape_value_component(face_to_local_indices_u[dofs_per_face_u-1], q_index, d);
                           }
                    	  bubble_flux += scalar_product(phi_i,normal_vector)*fe_face_values.JxW(q_index);

                    	  res_flux += scalar_product(res,normal_vector)*fe_face_values.JxW(q_index);
                    	  }
                    	 double bubble_coeff = 0;
                    	 bubble_coeff = res_flux/bubble_flux;
                    	 constraints.add_line(face_dof_indices_u[dofs_per_face_u - 1]);
                    	 constraints.set_inhomogeneity(face_dof_indices_u[dofs_per_face_u - 1], bubble_coeff);
//                    	 std::cout<<"bubble "<<face_dof_indices_u[dofs_per_face_u - 1]<<" "<<bubble_coeff<<std::endl;
                       }

                	 }
                  }

          constraints.close();
          }

          // distribute from local to global matrix and global rhs.
          cell->get_dof_indices (local_dof_indices);
          constraints.distribute_local_to_global(
                      local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs, true);

    }

 }


 template <int dim>
 void PoroElasBR1WG<dim>::solve_time_step ()
 {

	   SparseDirectUMFPACK A_direct;
	   A_direct.initialize(system_matrix);
	   A_direct.vmult(solution, system_rhs);

	   constraints.distribute(solution);

 }

 template <int dim>
 void PoroElasBR1WG<dim>::postprocess ()
 {// By now, this part is to calculate numerical dilation.

  div_displacement.reinit(triangulation.n_active_cells());
  div_displacement = 0;

//  div_num_displacement.reinit(dof_handler_br.n_dofs());
//  div_num_displacement = 0;

  QGauss<dim> quadrature_formula(3);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_quadrature_points | update_values |
                            update_gradients | update_JxW_values);
  FEValues<dim> fe_values_dgq(fe_dgq, quadrature_formula, update_quadrature_points);

  FEValues<dim> fe_values_rt (fe_rt, quadrature_formula,
                              update_values   | update_gradients |
                              update_quadrature_points | update_JxW_values);


  const FEValuesExtractors::Vector displacements(0);


  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = fe_values.get_quadrature().size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_dgq(1);
  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  typename DoFHandler<dim>::active_cell_iterator cell_dgq = dof_handler_dgq.begin_active();

  double cell_div_displacement;

  for (; cell != endc; ++cell, ++cell_dgq)
  {
    fe_values.reinit(cell);
    fe_values_dgq.reinit(cell_dgq);
    cell->get_dof_indices(local_dof_indices);
    cell_dgq->get_dof_indices(local_dof_indices_dgq);
//    std::cout<<local_dof_indices_dgq[0]<<std::endl;
    const double cell_area = cell->measure();
    cell_div_displacement = 0;

    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
    	  cell_div_displacement += solution(local_dof_indices[i])
    			                  *fe_values[displacements].divergence(i, q_index)
								  *fe_values.JxW(q_index)/cell_area;
      }
    }
    div_displacement[local_dof_indices_dgq[0]] = cell_div_displacement;
   }
 }


 ////////////////////////////////////////////////////////////////////////////////
 // calculate H1 seminorm of displacement
template <int dim>
void PoroElasBR1WG<dim>::postprocess_errors (double t, unsigned int n)
{
	   QGauss<dim>  quadrature_formula(3);
	   QGauss<dim-1>  face_quadrature_formula(3);

	   FEValues<dim> fe_values (fe, quadrature_formula,
	                              update_values   |
	                              update_quadrature_points |  update_gradients |
	                              update_JxW_values);


	   const unsigned int   dofs_per_cell = fe.dofs_per_cell;

	   const unsigned int   n_q_points    = fe_values.get_quadrature().size();

	   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	   const FEValuesExtractors::Vector displacements (0); //
	   const FEValuesExtractors::Scalar pressure_interior (dim); // this is for pressure
	   const FEValuesExtractors::Scalar pressure_face (dim+1);

	   DisplacementGradient<dim> displacement_exact_gradients;
	   displacement_exact_gradients.set_time (t);

	   	// compute l2 errors of displacement and int. pressure for each time step.
	   	const ComponentSelectFunction<dim>
	   	pressure_interior_mask (dim, dim+2);
	   	const ComponentSelectFunction<dim>
	   	velocity_mask(std::make_pair(0, dim), dim+2);

	   	ExactSolution<dim> exact_solution;
	   	exact_solution.set_time(t);
	   	Vector<double> cellwise_errors (triangulation.n_active_cells());

	   	VectorTools::integrate_difference (dof_handler, solution, exact_solution,
	   	                                   cellwise_errors, quadrature_formula,
	   	                                   VectorTools::L2_norm,
	   	                                   &pressure_interior_mask);
	   	const double intp_l2_error = VectorTools::compute_global_error(triangulation,
	   	                                                               cellwise_errors,
	   	                                                               VectorTools::L2_norm);

	   	VectorTools::integrate_difference (dof_handler, solution, exact_solution,
	   	                                   cellwise_errors, quadrature_formula,
	   	                                   VectorTools::L2_norm,
	   	                                   &velocity_mask);
	   	const double u_l2_error = VectorTools::compute_global_error(triangulation,
	   	                                                            cellwise_errors,
	   	                                                            VectorTools::L2_norm);

	   //	std::cout << "Errors: ||e_p||_L2 = " << intp_l2_error
	   //			  << ",   ||e_u||_L2 = " << u_l2_error<<std::endl;

	    double global_intp_l2_error_square = 0;
	    global_intp_l2_error_square = intp_l2_error*intp_l2_error;
	    double global_u_l2_error_square = 0;
	    global_u_l2_error_square = u_l2_error*u_l2_error;

	    l2_l2_intp_errors_square(timestep_number-1) = time_step * global_intp_l2_error_square;
	    l2_l2_u_errors_square(timestep_number-1) = time_step * global_u_l2_error_square;
/////////////////////////////////////

		   std::vector<Tensor<2, dim> >  div_displ_gradients (n_q_points);

		   double L2_err_displ_sqr_local = 0;
		   double L2_err_displ_sqr_global = 0;

		   typename DoFHandler<dim>::active_cell_iterator
		   cell = dof_handler.begin_active(),
		   endc = dof_handler.end();
		   for (unsigned int index=0; cell!=endc; ++cell, ++index)
		     {
		       fe_values.reinit (cell);
		       cell->get_dof_indices (local_dof_indices);

		       std::vector<Tensor<2, dim>>  displ_grad_values(n_q_points);
		       std::vector<Tensor<2, dim>>  displ_grad_values_diff(n_q_points);
		       L2_err_displ_sqr_local = 0;
		       for(unsigned int q=0; q<n_q_points; ++q)
		       {
		    	   displ_grad_values_diff[q] = 0;
		    	   for(unsigned int i=0; i<dofs_per_cell; ++i)
		    	   {
		    	   displ_grad_values[q] += solution(local_dof_indices[i])
		    			                *fe_values[displacements].gradient(i,q);
		    	   }
		    	   displ_grad_values_diff[q] = displ_grad_values[q]
									         - displacement_exact_gradients.vector_value(fe_values.quadrature_point (q));
		    	   L2_err_displ_sqr_local += displ_grad_values_diff[q].norm_square()*fe_values.JxW(q);

		       }
		       L2_err_displ_sqr_global += L2_err_displ_sqr_local;
		     }
		   l2_H1semi_displ_errors_square(n-1) = time_step*L2_err_displ_sqr_global;
		   l2_H1_displ_errors_square(n-1) = time_step
				                        *(L2_err_displ_sqr_global+global_u_l2_error_square);

}
 ////////////////////////////////////////////////////////////////////////////////
 // calculate numerical Darcy velocity
template <int dim>
void PoroElasBR1WG<dim>::postprocess_darcy_velocity (double t)
{
	// change "t" to number of time step to make sure it runs this part.
if (t == 10)
{
	std::cout<<"run the final time "<<std::endl;
	  QGauss<dim> quadrature_formula(3);
	  QGauss<dim-1>  face_quadrature_formula(3);
	  QMidpoint<dim>  quadrature_formula_velocity;
	  FEValues<dim> fe_values(fe,
	                          quadrature_formula,
	                          update_quadrature_points | update_values |
	                            update_gradients | update_JxW_values);
	  FEValues<dim> fe_values_dgq(fe_dgq, quadrature_formula, update_quadrature_points);

	  FEValues<dim> fe_values_rt (fe_rt, quadrature_formula,
	                              update_values   | update_gradients |
	                              update_quadrature_points | update_JxW_values);

	  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
	                                    update_values   | update_normal_vectors |
	                                    update_quadrature_points | update_JxW_values);

	  FEFaceValues<dim> fe_face_values_rt (fe_rt, face_quadrature_formula,
	                                       update_values   | update_normal_vectors |
	                                       update_quadrature_points | update_JxW_values);

	  FEValues<dim> fe_values_rt_velocity (fe_rt, quadrature_formula_velocity,
	                                       update_values   |
	                                       update_quadrature_points );


	  const FEValuesExtractors::Vector displacements(0);


	  const unsigned int dofs_per_cell = fe.dofs_per_cell;
	  const unsigned int dofs_per_cell_rt = fe_rt.dofs_per_cell;
	  const unsigned int dofs_per_cell_rt_velocity = fe_rt.dofs_per_cell;

	  const unsigned int n_q_points_rt = fe_values_rt.get_quadrature().size();
	//   std::cout<<n_q_points_rt<<std::endl;
	  const unsigned int n_q_points    = fe_values.get_quadrature().size();
	  const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
//	  const unsigned int n_face_q_points_rt = fe_face_values_rt.get_quadrature().size();
	  const unsigned int n_q_points_rt_velocity = fe_values_rt_velocity.get_quadrature().size();

	  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	  std::vector<types::global_dof_index> local_dof_indices_dgq(1);
	  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
	                                                 endc = dof_handler.end();

//  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  FullMatrix<double>   cell_matrix_rt (dofs_per_cell_rt,dofs_per_cell_rt);
  FullMatrix<double>   cell_matrix_F (dofs_per_cell,dofs_per_cell_rt);
  FullMatrix<double>   cell_matrix_C (dofs_per_cell,dofs_per_cell_rt);
  FullMatrix<double>   local_matrix (dofs_per_cell,dofs_per_cell);
  FullMatrix<double>   cell_matrix_D (dofs_per_cell_rt,dofs_per_cell_rt);
  FullMatrix<double>   cell_matrix_E (dofs_per_cell_rt,dofs_per_cell_rt);
  Vector<double>       cell_rhs (dofs_per_cell);
  Vector<double>       cell_solution (dofs_per_cell);
  Tensor<1,dim>        velocity_cell;
  std::vector<Tensor<1,dim>> velocity_cell_center(triangulation.n_active_cells());
  Tensor<1,dim>        velocity_face;
  Tensor<1,dim>        exact_velocity_face;
  Tensor<0,dim>        difference_velocity_cell_sqr;
  Tensor<0,dim>        difference_velocity_face_sqr;
  Tensor<0,dim>        L2_err_velocity_cell_sqr_local;
  Tensor<0,dim>        L2_err_velocity_cell_sqr_global;
  Tensor<0,dim>        flux_face;
  Tensor<0,dim>        difference_flux_face_sqr;
  Tensor<0,dim>        L2_err_flux_sqr;
  Tensor<0,dim>        L2_err_flux_face_sqr_local;
  Tensor<0,dim>        err_flux_each_face;
  Tensor<0,dim>        err_flux_face;

//  typename DoFHandler<dim>::active_cell_iterator
//  cell = dof_handler.begin_active(),
//  endc = dof_handler.end();

  typename DoFHandler<dim>::active_cell_iterator
  cell_rt = dof_handler_rt.begin_active();

  const Coefficient<dim> coefficient;
  std::vector<Tensor<2,dim>> coefficient_values (n_q_points_rt);
  const FEValuesExtractors::Vector velocities (0);
//  const FEValuesExtractors::Vector displacements (0);
  const FEValuesExtractors::Scalar pressure_interior (dim);
  const FEValuesExtractors::Scalar pressure_face (dim+1);

//  Velocity<dim> exact_velocity;

  std::ofstream center_out("center_coordinates.txt");

  for (unsigned int cell_index=0; cell!=endc; ++cell,++cell_rt, ++cell_index)
      {

       fe_values_rt.reinit (cell_rt);
       fe_values.reinit (cell);
       fe_values_rt_velocity.reinit (cell_rt);
       cell_matrix_rt = 0;
       cell_matrix_E = 0;
       cell_matrix_D = 0;
       cell_matrix_C = 0;
       cell_matrix_F = 0;
       velocity_cell = 0;
       coefficient.value_list (fe_values_rt.get_quadrature_points(),
                               coefficient_values);

       // The component of this cell matrix is the integral of $\mathbf{K} \mathbf{w} \cdot \mathbf{w}$.
       for (unsigned int q=0; q<n_q_points_rt; ++q)
           {
              for (unsigned int i=0; i<dofs_per_cell_rt; ++i)
                  {
           		 const Tensor<1,dim> phi_i_u = fe_values_rt[velocities].value (i, q);

                    for (unsigned int j=0; j<dofs_per_cell_rt; ++j)
                     {
                      const Tensor<1,dim> phi_j_u = fe_values_rt[velocities].value (j, q);

                      cell_matrix_E(i,j) += (coefficient_values[q] *
                   		                   phi_j_u * phi_i_u *
                   		                   fe_values_rt.JxW(q));
                     }
                  }
               }

       // This is the Gram matrix on the cell.
        for (unsigned int q=0; q<n_q_points_rt; ++q)
             {
               for (unsigned int i=0; i<dofs_per_cell_rt; ++i)
                   {
                     const Tensor<1,dim> phi_i_u = fe_values_rt[velocities].value (i, q);
                     for (unsigned int j=0; j<dofs_per_cell_rt; ++j)
                         {
                           const Tensor<1,dim> phi_j_u = fe_values_rt[velocities].value (j, q);
                           cell_matrix_rt(i,j) += (
                          		                   phi_i_u * phi_j_u *
                          		                   fe_values_rt.JxW(q));
                          }
                    }
               }

        // We take the inverse of the Gram matrix, take matrix multiplication and get the matrix with coefficients of projection.
         cell_matrix_rt.gauss_jordan();
         cell_matrix_rt.mmult(cell_matrix_D,cell_matrix_E);

        // This is to extract pressure values of the element.
         cell->get_dof_indices (local_dof_indices);
         cell_solution = 0;
         for (unsigned int i=0; i<dofs_per_cell; ++i)
             {
            	cell_solution(i) = solution(local_dof_indices[i]);
             }

          // This cell matrix will be used to calculate the coefficients of the Gram matrix.
          // This part is the same as the part in evaluating pressure.
          for (unsigned int q=0; q<n_q_points; ++q)
              {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
               	 {
                      for (unsigned int k=0; k<dofs_per_cell_rt; ++k)
                          {
                	          const double phi_k_u_div = fe_values_rt[velocities].divergence(k,q);
                	          cell_matrix_F(i,k) -= (fe_values[pressure_interior].value(i,q) *
                	                                 phi_k_u_div *
                	                                 fe_values.JxW (q));
                	        }
                	  }
                }

           for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
               {
                  fe_face_values.reinit (cell,face_n);
                  fe_face_values_rt.reinit (cell_rt,face_n);
                  for (unsigned int q=0; q<n_face_q_points; ++q)
                      {
                    	 const Tensor<1,dim> normal = fe_face_values.normal_vector(q);
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        	  {
                        	    for (unsigned int k=0; k<dofs_per_cell_rt; ++k)
                        	        {
                        	          const Tensor<1,dim> phi_k_u = fe_face_values_rt[velocities].value (k, q);
                        	          cell_matrix_F(i,k) += (fe_face_values[pressure_face].value(i,q) *
                        	                                (phi_k_u * normal)*
                        	                                 fe_face_values.JxW (q));
                        	         }
                        	   }
                        }
                 }

            cell_matrix_F.mmult(cell_matrix_C,cell_matrix_rt);

            // From previous calculations, we have all the coefficients. Then we get the coefficient @p beta.
            Vector<double> beta (dofs_per_cell_rt);
            beta = 0;
            for(unsigned int k = 0; k<dofs_per_cell_rt;++k)
                {
                  for (unsigned int j = 0; j<dofs_per_cell_rt; ++j)
                       {
                         for (unsigned int i = 0; i<dofs_per_cell;++i)
                             {
                                beta(k) += -(cell_solution(i)*
                                   		  cell_matrix_C(i,j) *
                                   	      cell_matrix_D(k,j));
                              }
                        }
                 }

             // We calculate numerical velocity on each quadrature point,
             // find the squares of difference between numerical and exact velocity on the quadrature point.
             // Then we approximate the error over all quadrature points and add it to the global $L_2$ error.
//             L2_err_velocity_cell_sqr_local = 0;
//             for (unsigned int q=0; q<n_q_points_rt; ++q)
//                 {
////               	 difference_velocity_cell_sqr = 0;
//               	 velocity_cell = 0;
//                    for(unsigned int k = 0; k<dofs_per_cell_rt;++k)
//                       {
//                         const Tensor<1,dim> phi_k_u = fe_values_rt[velocities].value (k, q);
//                         velocity_cell += beta(k)*phi_k_u;
//                       }
//                    difference_velocity_cell_sqr = (velocity_cell - exact_velocity.value (fe_values_rt.quadrature_point (q)))*
//                       		                    (velocity_cell - exact_velocity.value (fe_values_rt.quadrature_point (q)));
//                    L2_err_velocity_cell_sqr_local += difference_velocity_cell_sqr*fe_values_rt.JxW(q);
//                  }

//              L2_err_velocity_cell_sqr_global += L2_err_velocity_cell_sqr_local;

              for (unsigned int q=0; q<n_q_points_rt_velocity; ++q)
                  {
                    for(unsigned int k = 0; k<dofs_per_cell_rt_velocity;++k)
                       {
                         const Tensor<1,dim> phi_k_u = fe_values_rt_velocity[velocities].value (k, q);
//                          std::cout << "phi_k_u " <<phi_k_u<<std::endl;
                         velocity_cell_center[cell_index] += beta(k)*phi_k_u;
                       }
                  }
//               std::cout<<cell->center()[0]<<std::endl; // x-dir
//               std::cout<<cell->center()[1]<<std::endl; // y-dir
//               std::cout<<velocity_cell_center[cell_index]<<std::endl;
//
//
              center_out<<cell->center()<<" "<< 0 << " "<<std::endl;

     }

 std::ofstream out("velocity.txt");
 for(int i=0; i<triangulation.n_active_cells();++i)
 {
 out<<velocity_cell_center[i]<<" "<<0<<std::endl;
 }
}

}
////////////////////////////////////////////////////////////////////////////////
template <int dim>
void PoroElasBR1WG<dim>::projection_exact_solution (double t)
  {
	// exact divergence displacement
 div_projection.reinit(triangulation.n_active_cells());
 pressure_projection.reinit(triangulation.n_active_cells());

 QGauss<dim> quadrature_formula(fe.degree+2);
 FEValues<dim> fe_values(fe,
	                     quadrature_formula,
	                     update_quadrature_points | update_values |
	                     update_gradients | update_JxW_values);
 FEValues<dim> fe_values_dgq(fe_dgq, quadrature_formula, update_quadrature_points |
			                 update_values |
                             update_gradients | update_JxW_values);


 const unsigned int dofs_per_cell = fe.dofs_per_cell;
 const unsigned int n_q_points    = quadrature_formula.size();

 std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
 std::vector<types::global_dof_index> local_dof_indices_dgq(1);


  DisplacementDiv<dim> exact_displacement_div;
  exact_displacement_div.set_time(t);
  PressureSolution<dim> exact_pressure;
  exact_pressure.set_time(t);

  double cell_div_projection;
  double cell_pres_projection;

  typename DoFHandler<dim>::active_cell_iterator
     cell = dof_handler.begin_active(),
     endc = dof_handler.end();
     typename DoFHandler<dim>::active_cell_iterator
     cell_rt = dof_handler_rt.begin_active();
     typename DoFHandler<dim>::active_cell_iterator
	 cell_dgq = dof_handler_dgq.begin_active();

     for (; cell!=endc; ++cell,++cell_rt,++cell_dgq)
     {
   	  fe_values.reinit(cell);
   	  fe_values_dgq.reinit(cell_dgq);
   	  cell->get_dof_indices(local_dof_indices);
   	  cell_dgq->get_dof_indices(local_dof_indices_dgq);
   	  cell_div_projection = 0;
      cell_pres_projection = 0;

   	  const double cell_area = cell->measure();

   	  for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
   	  {
   		const double exact_displdiv = exact_displacement_div.value(fe_values.quadrature_point (q_index));
   		const double exact_press_proj = exact_pressure.value(fe_values.quadrature_point (q_index));
   		cell_div_projection += exact_displdiv * fe_values.JxW(q_index)/cell_area;
   		cell_pres_projection += exact_press_proj * fe_values.JxW(q_index)/cell_area;

   	  }
   	  div_projection[local_dof_indices_dgq[0]] = cell_div_projection;
   	  pressure_projection[local_dof_indices_dgq[0]] = cell_pres_projection;

     }
  }

template <int dim>
void PoroElasBR1WG<dim>::Interp_proj_exact_displ (double t)
{
	intrproj_displ.reinit(dof_handler_br.n_dofs());
	QGauss<dim> quadrature_formula(fe_br.degree+1);
	QGauss<dim-1> face_quadrature_formula(fe_br.degree+1);

	FEValues<dim> fe_values_br(fe_br,
	                        quadrature_formula,
	                        update_quadrature_points | update_values |
	                        update_gradients | update_JxW_values);


	FEFaceValues<dim> fe_face_values_br (fe_br, face_quadrature_formula,
	                                  update_values    | update_normal_vectors |
	                                  update_quadrature_points  | update_JxW_values);

	const unsigned int dofs_per_cell = fe_br.dofs_per_cell;
    const unsigned int dofs_per_face = fe_br.dofs_per_face;
	const unsigned int n_q_points    = quadrature_formula.size();
	const unsigned int n_q_face_points= face_quadrature_formula.size();

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	std::vector<types::global_dof_index> face_dof_indices(dofs_per_face);
	std::vector<unsigned int> face_to_local_indices(dofs_per_face);

	DisplacementSolution<dim> exact_displacement;
	exact_displacement.set_time(t);

	typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler_br.begin_active(),
    endc = dof_handler_br.end();


	for (unsigned int index = 0; cell != endc; ++cell, ++index)
	  {
	    fe_values_br.reinit(cell);
	    cell->get_dof_indices(local_dof_indices);

	    // Linearly interpolate the boundary conditions at every vertex
	    // and set the bubble function DoF to 0
	    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
	        {

	          cell->face(f)->get_dof_indices(face_dof_indices);
	          for (unsigned int dof_face = 0; dof_face < dofs_per_face; ++dof_face)
	          {
	            for (unsigned int dof_cell = 0; dof_cell < dofs_per_cell; ++dof_cell)
	            {
	              if( local_dof_indices[dof_cell] == face_dof_indices[dof_face] )
	                face_to_local_indices[dof_face] = dof_cell;

	            }
	          }

	          fe_face_values_br.reinit(cell,f);
	          std::vector<double> boundary_values(dofs_per_face - 1, 0);

	          // set nodal coeffs.
	          for(unsigned int dof = 0; dof < dofs_per_face - 1; ++dof)
	             {
	               boundary_values[dof] = exact_displacement.vector_value(cell->face(f)->vertex(dof/dim))[dof % dim];
	             }

	          for (unsigned int dof = 0; dof < dofs_per_face - 1; ++dof)
	          {
	            intrproj_displ[face_dof_indices[dof]] = boundary_values[dof];
	          }

              // set bubble coeffs.
	          double res_flux= 0;
	          double bubble_flux = 0;
	          Tensor<1, dim> normal_vector = fe_face_values_br.normal_vector(0);
	          for (unsigned int q_index = 0; q_index < n_q_face_points; ++q_index)
	            {
	               Tensor<1, dim> phi_i;
	               Tensor<1, dim> res = exact_displacement.vector_value(fe_face_values_br.quadrature_point (q_index));

	               for (unsigned int d=0; d<dim; ++d)
	               {
	                 for (unsigned int dof = 0; dof < dofs_per_face - 1; ++dof)
	                    res[d] -= boundary_values[dof]*fe_face_values_br.shape_value_component(face_to_local_indices[dof], q_index, d);
	               }
	               for (unsigned int d=0; d<dim; ++d)
	               {
	                 phi_i[d] = fe_face_values_br.shape_value_component(face_to_local_indices[dofs_per_face-1], q_index, d);
	               }

	                    bubble_flux += scalar_product(phi_i,normal_vector)*fe_face_values_br.JxW(q_index);
	                    res_flux += scalar_product(res,normal_vector)*fe_face_values_br.JxW(q_index);
	            }
	           double bubble_coeff = res_flux/bubble_flux;
	           intrproj_displ[face_dof_indices[dofs_per_face - 1]] = bubble_coeff;

	        }
	  }
}

template <int dim>
void PoroElasBR1WG<dim>::Interp_proj_exact_solution (double t)
{
	// Take projection of int. pressure and face pressure.
	// By now, this part only works for the lowest order WG.
	// If needed later, need to redefine following vectors.

	QGauss<dim> quadrature_formula(3);
	QGauss<dim-1> face_quadrature_formula(3);

	FEValues<dim> fe_values(fe,
	                        quadrature_formula,
	                        update_quadrature_points | update_values |
	                        update_gradients | update_JxW_values);

	FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
	                                  update_values    | update_normal_vectors |
	                                  update_quadrature_points  | update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int dofs_per_face = fe.dofs_per_face;
	const unsigned int n_q_points    = quadrature_formula.size();
	const unsigned int n_q_face_points= face_quadrature_formula.size();

	const unsigned int dofs_per_cell_u = fe.base_element(0).dofs_per_cell;
	const unsigned int dofs_per_face_u = fe.base_element(0).dofs_per_face;
	const unsigned int dofs_per_cell_intp = fe.base_element(1).dofs_per_cell;
	const unsigned int dofs_per_cell_facep = fe.base_element(2).dofs_per_cell;
	const unsigned int dofs_per_face_p = fe.base_element(2).dofs_per_face;

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	std::vector<types::global_dof_index> face_dof_indices(dofs_per_face);
	std::vector<unsigned int> face_to_local_indices(dofs_per_face);
    std::vector<types::global_dof_index> face_dof_indices_u(dofs_per_face_u);
//    std::vector<unsigned int> face_dof_indices_facep(dofs_per_face_p);
	std::vector<unsigned int> face_to_local_indices_u(dofs_per_face_u);
	std::vector<unsigned int> local_dof_indices_facep(dofs_per_cell_facep);
	std::vector<unsigned int> local_dof_indices_intp(dofs_per_cell_intp);

	double face_dof_indices_facep;

	DisplacementSolution<dim> exact_displacement;
	exact_displacement.set_time(t);

	PressureSolution<dim> exact_pressure;
	exact_pressure.set_time(t);

	double cell_int_pres_projection;
	double cell_face_pres_projection;

	typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active(),
    endc = dof_handler.end();


	for (unsigned int index = 0; cell != endc; ++cell, ++index)
	  {
	    fe_values.reinit(cell);
	    cell->get_dof_indices(local_dof_indices);
	    double area = cell->measure();
	    cell_int_pres_projection = 0;
	    cell_face_pres_projection = 0;
	    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
	     {
	       const double exact_int_press_proj = exact_pressure.value(fe_values.quadrature_point (q_index));
	       cell_int_pres_projection += exact_int_press_proj * fe_values.JxW(q_index)/area;
	     }
	    for (unsigned int i=0; i<fe.base_element(1).dofs_per_cell; ++i) // i<dofs_per_cell_facep
	      {
	       local_dof_indices_intp[i] = local_dof_indices[fe.component_to_system_index(dim,i)];
	       intrproj_solution[local_dof_indices_intp[i]] = cell_int_pres_projection;
	      }

	    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
	    {
	      fe_face_values.reinit (cell, f);
	      cell->face(f)->get_dof_indices(face_dof_indices);
	      cell_face_pres_projection = 0;
	      double face_length = cell->face(f)->measure();

		  for (unsigned int i=0; i<fe.base_element(2).dofs_per_cell; ++i) // i<dofs_per_cell_facep
		    {
		     local_dof_indices_facep[i] = local_dof_indices[fe.component_to_system_index(dim+1,i)];
		    }

      	face_dof_indices_facep = 0;
      	  for(unsigned int i = 0; i<dofs_per_face; ++i)
      	  {
      		 unsigned int j;
      		 for(j = 0; j< dofs_per_cell_facep; ++j)
      		 {
      		   if(face_dof_indices[i] == local_dof_indices_facep[j])
      			 face_dof_indices_facep = local_dof_indices_facep[j];
      		 }
      	  }

      	for (unsigned int q_index = 0; q_index < n_q_face_points; ++q_index)
      	  {
      		const double exact_face_press_proj = exact_pressure.value(fe_face_values.quadrature_point (q_index));
      		cell_face_pres_projection += exact_face_press_proj * fe_face_values.JxW(q_index)/face_length;
          }

	    intrproj_solution[face_dof_indices_facep] = cell_face_pres_projection;

	    // Linearly interpolate the boundary conditions at every vertex
	    // and set the bubble function DoF to 0
    	unsigned int index_i = 0;
        for(unsigned int i = 0; i<dofs_per_face; ++i)
    	  {
    		 unsigned int j;
    		 for(j = 0; j< dofs_per_cell_facep; ++j)

    		   if(face_dof_indices[i] == local_dof_indices_facep[j])
    				break;

    		 if(j == dofs_per_cell_facep && index_i < dofs_per_face_u)
    		  {
    			face_dof_indices_u[index_i] = face_dof_indices[i];
    			++index_i;
    		  }
    	   }
	    for(unsigned int i = 0; i<dofs_per_face_u; ++i)
	     for (unsigned int dof_face = 0; dof_face < dofs_per_face_u; ++dof_face)
	       {
	         for (unsigned int dof_cell = 0; dof_cell < dofs_per_cell; ++dof_cell)
	          {
	            if( local_dof_indices[dof_cell] == face_dof_indices_u[dof_face] )
	              {
	            	face_to_local_indices_u[dof_face] = dof_cell;
	              }
	           }
	       }
         	for(unsigned int i = 0; i<dofs_per_face_u; ++i)
              for (unsigned int dof_face = 0; dof_face < dofs_per_face_u; ++dof_face)
              {
             	 for (unsigned int dof_cell = 0; dof_cell < dofs_per_cell; ++dof_cell)
                {
             	   if( local_dof_indices[dof_cell] == face_dof_indices_u[dof_face] )
                    {face_to_local_indices_u[dof_face] = dof_cell;
                    }
                }
              }
            fe_face_values.reinit(cell,f);
            std::vector<double> vertex_values_u(dofs_per_face_u - 1, 0);
            for(unsigned int dof = 0; dof < dofs_per_face_u - 1; ++dof)
            {
                vertex_values_u[dof] = exact_displacement.vector_value
				                       (cell->face(f)->vertex(dof/dim))[dof % dim];
            }

            for (unsigned int dof = 0; dof < dofs_per_face_u - 1; ++dof)
             {
            	intrproj_solution[face_dof_indices_u[dof]] = vertex_values_u[dof];
             }

            double res_flux= 0;
            double bubble_flux = 0;
            Tensor<1, dim> normal_vector = fe_face_values.normal_vector(0);
            for (unsigned int q_index = 0; q_index < n_q_face_points; ++q_index)
              {
                Tensor<1, dim> phi_i;
                Tensor<1, dim> res = exact_displacement.vector_value
                    			     (fe_face_values.quadrature_point (q_index));

                for (unsigned int d=0; d<dim; ++d)
                  {
                    for (unsigned int dof = 0; dof < fe.base_element(0).dofs_per_face - 1; ++dof)
                      {
                    	res[d] -= vertex_values_u[dof]*fe_face_values.shape_value_component(face_to_local_indices_u[dof], q_index, d);
                      }
                   }
                for (unsigned int d=0; d<dim; ++d)
                   {
                     phi_i[d] = fe_face_values.shape_value_component(face_to_local_indices_u[dofs_per_face_u-1], q_index, d);
                    }
                bubble_flux += scalar_product(phi_i,normal_vector)*fe_face_values.JxW(q_index);

                res_flux += scalar_product(res,normal_vector)*fe_face_values.JxW(q_index);
                }
             double bubble_coeff = 0;
             bubble_coeff = res_flux/bubble_flux;
             intrproj_solution[face_dof_indices_u[dofs_per_face_u-1]] = bubble_coeff;

               }
	  }
}

//////////////////////////////////////////////////////////////////////////////////
//template <int dim>
//void PoroElasBR1WG<dim>::compute_errors (double t)
//{
//	// compute l2 errors for each time step.
//	const ComponentSelectFunction<dim>
//	pressure_interior_mask (dim, dim+2);
//	const ComponentSelectFunction<dim>
//	velocity_mask(std::make_pair(0, dim), dim+2);
//
//	ExactSolution<dim> exact_solution;
//	exact_solution.set_time(t);
//	Vector<double> cellwise_errors (triangulation.n_active_cells());
//
//	QGauss<dim> quadrature_formula(3);
//
//	VectorTools::integrate_difference (dof_handler, solution, exact_solution,
//	                                   cellwise_errors, quadrature_formula,
//	                                   VectorTools::L2_norm,
//	                                   &pressure_interior_mask);
//	const double intp_l2_error = VectorTools::compute_global_error(triangulation,
//	                                                               cellwise_errors,
//	                                                               VectorTools::L2_norm);
//
//	VectorTools::integrate_difference (dof_handler, solution, exact_solution,
//	                                   cellwise_errors, quadrature_formula,
//	                                   VectorTools::L2_norm,
//	                                   &velocity_mask);
//	const double u_l2_error = VectorTools::compute_global_error(triangulation,
//	                                                            cellwise_errors,
//	                                                            VectorTools::L2_norm);
//
////	std::cout << "Errors: ||e_p||_L2 = " << intp_l2_error
////			  << ",   ||e_u||_L2 = " << u_l2_error<<std::endl;
//
//}


template <int dim>
void PoroElasBR1WG<dim>::output_results() const
{
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// 3D_solution
//	std::vector<std::string> solution_names;
//	solution_names.push_back ("u_x_direction");
//	solution_names.push_back ("u_y_direction");
//	solution_names.push_back ("u_z_direction");
//    solution_names.push_back ("p_interior");
//    solution_names.push_back ("p_edge");
//
//
//	DataOut<dim> data_out;
//    data_out.attach_dof_handler (dof_handler);
//    data_out.add_data_vector (solution, solution_names);
//
//    data_out.build_patches (fe.degree+1);
//
//    std::ostringstream filename;
//    filename << "solution-"
//             << Utilities::int_to_string(timestep_number,3)
//	         << ".vtk";
//
//   std::ofstream output (filename.str().c_str());
//   data_out.write_vtk (output);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D_solution
	std::vector<std::string> solution_names;
	solution_names.push_back ("u_x_direction");
	solution_names.push_back ("u_y_direction");
    solution_names.push_back ("p_interior");
    solution_names.push_back ("p_edge");


	DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, solution_names);

    data_out.build_patches (fe.degree+1);
//    data_out.build_patches ();

    std::ostringstream filename;
    filename << "solution-"
             << Utilities::int_to_string(timestep_number,3)
	         << ".vtk";

   std::ofstream output (filename.str().c_str());
   data_out.write_vtk (output);

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // DataOut for numerical divergence
//   DataOut<dim> data_out_dgq;
//   data_out_dgq.attach_dof_handler(dof_handler_dgq);
//
//   std::string div_name = "div_u";
//   data_out_dgq.add_data_vector(div_displacement, div_name, data_out_dgq.type_cell_data);
//
//   data_out_dgq.build_patches (fe.degree+1);
//   std::ostringstream filename_div;
//   filename_div << "solution-div"
//            << Utilities::int_to_string(timestep_number,3)
//	         << ".vtk";
//   std::ofstream output_div (filename_div.str().c_str());
//   data_out_dgq.write_vtk(output_div);

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // DataOut for projection of numerical divergence
   DataOut<dim> data_out_br;
   data_out_br.attach_dof_handler(dof_handler_br);

   std::string div_name = "div_u";
   data_out_br.add_data_vector(div_displacement, div_name, data_out_br.type_cell_data);

   data_out_br.build_patches (fe.degree+1);
   std::ostringstream filename_div;
   filename_div << "solution-div"
            << Utilities::int_to_string(timestep_number,3)
	         << ".vtk";
   std::ofstream output_div (filename_div.str().c_str());
   data_out_br.write_vtk(output_div);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D_interpolation_projection_exact_solution
//   	std::vector<std::string> solution_names_exact;
//   	solution_names_exact.push_back ("exact_u_x_direction");
//   	solution_names_exact.push_back ("exact_u_y_direction");
//   	solution_names_exact.push_back ("exact_p_interior");
//   	solution_names_exact.push_back ("exact_p_edge");
//
//
//   	DataOut<dim> data_out_exact;
//   	data_out_exact.attach_dof_handler (dof_handler);
//   	data_out_exact.add_data_vector (intrproj_solution, solution_names);
//
////    data_out.build_patches (fe.degree+1);
//   	data_out_exact.build_patches();
//
//    std::ostringstream filename_exact;
//    filename_exact << "solution-exact"
//             << Utilities::int_to_string(timestep_number,3)
//   	         << ".vtk";
//
//    std::ofstream output_exact (filename_exact.str().c_str());
//    data_out_exact.write_vtk (output_exact);

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataOut for divergence
//     DataOut<dim> data_out_dgq;
//     data_out_dgq.attach_dof_handler(dof_handler_dgq);
//     std::string div_name = "div_u";
//     std::string pres_proj_name = "proj_p";
//     data_out_dgq.add_data_vector(div_projection, div_name, data_out_dgq.type_cell_data);
//     data_out_dgq.add_data_vector(pressure_projection, pres_proj_name, data_out_dgq.type_cell_data);
//     data_out_dgq.build_patches(fe_dgq.degree+1);
//     std::ostringstream filename_proj_divdispl;
//     filename_proj_divdispl << "proj_divdispl-"
//                  << Utilities::int_to_string(timestep_number,3)
//     	         << ".vtk";
//
//     std::ostringstream filename_proj_pres;
//     filename_proj_pres << "proj_pres-"
//                       << Utilities::int_to_string(timestep_number,3)
//          	         << ".vtk";
//
//     std::ofstream divu_proj_output (filename_proj_divdispl.str().c_str());
//     data_out_dgq.write_vtk (divu_proj_output);
//
//     std::ofstream pres_proj_output (filename_proj_pres.str().c_str());
//     data_out_dgq.write_vtk (pres_proj_output);

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//// DataOut for intrp_proj_displ, intrproj_displ
//     DataOut<dim> data_out_br;
//     data_out_br.attach_dof_handler(dof_handler_br);
//     std::string intrp_proj_displ_name = "intrp_proj_u";
//     data_out_br.add_data_vector(intrproj_displ, intrp_proj_displ_name, data_out_br.type_cell_data);
//     data_out_br.build_patches(fe_br.degree+1);
//     std::ostringstream filename_intrproj_displ;
//     filename_intrproj_displ << "intrproj_displ-"
//                  << Utilities::int_to_string(timestep_number,3)
//     	         << ".vtk";
//
//
//     std::ofstream intrproj_u_output (filename_intrproj_displ.str().c_str());
//     data_out_br.write_vtk (intrproj_u_output);

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

   }

 template <int dim>
 void PoroElasBR1WG<dim>::run ()
 {
   std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

   make_grid_and_dofs();

   old_solution = 0; // This part is to take projection of initial condition.

   // rewrite l2l2 errors of displ and intp into different class
//   unsigned int number_timestep = total_time/time_step;
//   Vector<double> l2_l2_intp_errors_square(number_timestep);
//   Vector<double> l2_l2_u_errors_square(number_timestep);
//   Vector<double> l2_H1_displ_errors_square(number_timestep);
   l2_l2_intp_errors_square.reinit(number_timestep);
   l2_l2_u_errors_square.reinit(number_timestep);
   l2_H1semi_displ_errors_square.reinit(number_timestep);
   l2_H1_displ_errors_square.reinit(number_timestep);

   double l2_l2_intp_errors_global_time = 0;
   double l2_l2_u_errors_global_time = 0;
   double l2_H1semi_u_errors_global_time = 0;
   double l2_H1_u_errors_global_time = 0;


   for (timestep_number=1, time=time_step; time <= 1; time+=time_step, ++timestep_number)
     {
       std::cout << "Time step " << timestep_number
                 << " at t=" << time << " time_step "<<time_step
                 << std::endl;

   assemble_system (time);

   solve_time_step();

   postprocess();

   postprocess_errors(time,timestep_number);

   postprocess_darcy_velocity(timestep_number);

//   projection_exact_solution(time);

//   Interp_proj_exact_displ(time);

//  Interp_proj_exact_solution (time);

//  const ComponentSelectFunction<dim>
//  pressure_interior_mask (dim, dim+2);
//  const ComponentSelectFunction<dim>
//  velocity_mask(std::make_pair(0, dim), dim+2);
//
//  ExactSolution<dim> exact_solution;
//  exact_solution.set_time(time);
//
//  Vector<double> cellwise_errors (triangulation.n_active_cells());
//
//
//  VectorTools::integrate_difference (dof_handler, solution, exact_solution,
//	                                 cellwise_errors, QGauss<dim>(3),
//	                                 VectorTools::L2_norm,
//	                                   &pressure_interior_mask);
//  const double intp_l2_error = VectorTools::compute_global_error(triangulation,
//	                                                             cellwise_errors,
//	                                                             VectorTools::L2_norm);
//
//  VectorTools::integrate_difference (dof_handler, solution, exact_solution,
//	                                 cellwise_errors, QGauss<dim>(3),
//	                                 VectorTools::L2_norm,
//	                                   &velocity_mask);
//  const double u_l2_error = VectorTools::compute_global_error(triangulation,
//	                                                          cellwise_errors,
//	                                                          VectorTools::L2_norm);
//
//
//  double global_intp_l2_error_square = 0;
//  global_intp_l2_error_square = intp_l2_error*intp_l2_error;
//  double global_u_l2_error_square = 0;
//  global_u_l2_error_square = u_l2_error*u_l2_error;
//
//  l2_l2_intp_errors_square(timestep_number-1) = time_step * global_intp_l2_error_square;
//  l2_l2_u_errors_square(timestep_number-1) = time_step * global_u_l2_error_square;

  output_results ();

  old_solution = solution;

 } // the end of each time step

////   double l2_l2_intp_errors_global_time;
////   double l2_l2_u_errors_global_time;
   double l2_l2_intp_errors;
   double l2_l2_u_errors;
   double l2_h1semi_u_errors; // H1 semi-norm of displacement
   double l2_h1_u_errors;
   for(unsigned int i=0; i<number_timestep; ++i)
   {
     l2_l2_intp_errors_global_time += l2_l2_intp_errors_square(i);
     l2_l2_u_errors_global_time += l2_l2_u_errors_square(i);
     l2_H1semi_u_errors_global_time += l2_H1semi_displ_errors_square(i);
     l2_H1_u_errors_global_time += l2_H1_displ_errors_square(i);
   }

   l2_l2_intp_errors = std::sqrt(l2_l2_intp_errors_global_time);
   l2_l2_u_errors = std::sqrt(l2_l2_u_errors_global_time);
   l2_h1semi_u_errors = std::sqrt(l2_H1semi_u_errors_global_time);
   l2_h1_u_errors = std::sqrt(l2_H1_u_errors_global_time);
   std::cout<<"L2L2Error_intp "<<l2_l2_intp_errors<<", "
		    <<"L2L2Error_u "<<l2_l2_u_errors<<", "
			<<"L2h1semiError_u "<<l2_h1semi_u_errors<<", "
			<<"L2h1Error_u "<<l2_h1_u_errors<<std::endl;

}


 int main ()
 {
   deallog.depth_console (2);

   PoroElasBR1WG<2> PoroElasBR1WGTest;
   PoroElasBR1WGTest.run ();

   return 0;
 }
