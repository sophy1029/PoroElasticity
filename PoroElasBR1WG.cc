/*
 * PoroElasBR1WG.cc

 *  Created on: Feb 15, 2018
 *      Author: ubuntu
 */

// @sect3{Include files}
// This program is based on step-7, step-20 and step-51,
// we add these include files.
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
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
//#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
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
//#include <deal.II/fe/fe_bernardi_raugel.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/point.h>
#include <deal.II/fe/component_mask.h>


#include <fstream>
#include <iostream>

using namespace dealii;

// @sect3{The WGDarcyEquation class template}

template <int dim>
class PoroElasBR1WG
{
public:
  PoroElasBR1WG ();
  void run ();

private:
  void make_grid_and_dofs ();
//  void solve_time_step();
  void assemble_system ();
//  void solve ();

  Triangulation<dim>   triangulation;

  ConstraintMatrix     constraints;

  FE_RaviartThomas<dim>  fe_rt;
  DoFHandler<dim>      dof_handler_rt;

  FESystem<dim>          fe;
  DoFHandler<dim>      dof_handler;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

//  Vector<double>       solution_displacement, solution_pressure;
//  Vector<double>       old_solution_displacement, old_solution_pressure;
//  Vector<double>       system_rhs;

  BlockVector<double> solution;
  BlockVector<double> old_solution;
  BlockVector<double> system_rhs;
  Vector<double> old_solution_displacement;
  Vector<double> old_solution_pressure;

  double       time;
  double       time_step;
  unsigned int timestep_number;
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
        values[p][d][d] = 1;
    }
}

// Exact solutions, u and p
template <int dim>
class DisplacementSolution : public Function<dim>
{
public:
  DisplacementSolution () : Function<dim>(dim) {}
  virtual void value (const Point<dim>   &p) const;
};

template <int dim>
void DisplacementSolution<dim>::value (const Point<dim>   &p) const
{
  value(0) = -1./(4*M_PI)*cos(2*M_PI*p[0])*sin(2*M_PI*p[1])*sin(2*M_PI*this->get_time());
  value(1) = -1./(4*M_PI)*sin(2*M_PI*p[0])*cos(2*M_PI*p[1])*sin(2*M_PI*this->get_time());

}

template <int dim>
class PressureSolution : public Function<dim>
{
public:
	PressureSolution()
    : Function<dim>(1)
  {}
  virtual double value(const Point<dim> &p, const unsigned int) const;
};

template <int dim>
double PressureSolution<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double return_value = 0;
  return_value        = sin(2*M_PI*p[0])*sin(2*M_PI*p[1])*sin(2*M_PI*this->get_time());
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
  return_value        = 8*M_PI*M_PI*sin(2*M_PI*p[0])*sin(2*M_PI*p[1])*sin(2*M_PI*this->get_time())+
		                2*M_PI*sin(2*M_PI*p[0])*sin(2*M_PI*p[1])*cos(2*M_PI*this->get_time());
  return return_value;
}

template <int dim>
  class BodyRightHandSide : public Function<dim>
  {
  public:
	BodyRightHandSide () : Function<dim>(dim) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };
template <int dim>
  void
  BodyRightHandSide<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));

    values(0) = -4*M_PI*cos(2*M_PI*p[0])*sin(2*M_PI*p[1])*sin(2*M_PI*this->get_time());
    values(1) = -4*M_PI*sin(2*M_PI*p[0])*cos(2*M_PI*p[1])*sin(2*M_PI*this->get_time());
  }

// Displacement D.B.C., pressure is N.B.C., calculated later(?)
template <int dim>
  class DisplacementBoundary : public Function<dim>
  {
  public:
	DisplacementBoundary () : Function<dim>(dim) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };
template <int dim>
  void
  DisplacementBoundary<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));

    values(0) = -1./(4*M_PI)*cos(2*M_PI*p[0])*sin(2*M_PI*p[1])*sin(2*M_PI*this->get_time());
    values(1) = -1./(4*M_PI)*sin(2*M_PI*p[0])*cos(2*M_PI*p[1])*sin(2*M_PI*this->get_time());
  }

// Initial values, u(0,x)=0 and p(0,x)=0
template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
    InitialValues () : Function<dim>(dim+2) {}  //?? is it dim+2?

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };

  template <int dim>
  double
  InitialValues<dim>::value (const Point<dim>  &p,
                             const unsigned int component ) const
  {
    return ZeroFunction<dim>(dim+2).value (p, component);
//	  return 0;
  }

  template <int dim>
    void
    InitialValues<dim>::vector_value (const Point<dim> &p,
                                      Vector<double>   &values) const
    {
      ZeroFunction<dim>(dim+2).vector_value (p, values);
//	  values(0) = 0;
//	  values(1) = 0;
    }


//template <int dim>
//  class InitialValuesPressure : public Function<dim>
//  {
//  public:
//    InitialValuesPressure () : Function<dim>(dim+2) {}  //?? is it dim+2?
//
//    virtual double value (const Point<dim>   &p,
//                          const unsigned int  component = 0) const;
//  };
//
//  template <int dim>
//  double
//  InitialValuesPressure<dim>::value (const Point<dim>  &p,
//                             const unsigned int component ) const
//  {
//    return ZeroFunction<dim>(dim+2).value (p, component);
//  }
//
//  template <int dim>
//    class InitialValuesDisplacement : public Function<dim>
//    {
//    public:
//      InitialValuesDisplacement () : Function<dim>(dim+2) {}  //?? is it dim+2?
//
//      virtual void vector_value (const Point<dim> &p,
//                                 Vector<double>   &value) const;
//    };
//
//  template <int dim>
//  void
//  InitialValuesDisplacement<dim>::vector_value (const Point<dim> &p,
//                                    Vector<double>   &values) const
//  {
//    ZeroFunction<dim>(dim+2).vector_value (p, values);
//  }

// @sect3{WGDarcyEquation class implementation}

// @sect4{WGDarcyEquation::WGDarcyEquation}

 template <int dim>
 PoroElasBR1WG<dim>::PoroElasBR1WG ()
   :
   fe_rt (0),
   dof_handler_rt (triangulation), // for discrete weak gradient

//   fe (FE_BernardiRaugel<dim>(1),2,  // what's for dim? 1? 2?
   fe (FE_Q<dim>(1), dim,
	   FE_DGQ<dim>(0), 1,
	   FE_FaceQ<dim>(0), 1),

   dof_handler (triangulation),

   time_step (1./64) // what's time_step? need to set later, different from step-23

 {}

// @sect4{WGDarcyEquation::make_grid}

 template <int dim>
 void PoroElasBR1WG<dim>::make_grid_and_dofs ()
 {
   GridGenerator::hyper_cube (triangulation, 0, 1);
   triangulation.refine_global (0);

   dof_handler_rt.distribute_dofs (fe_rt);
   dof_handler.distribute_dofs(fe);

   DoFRenumbering::component_wise (dof_handler);
   std::vector<types::global_dof_index> dofs_per_component (dim+2);
   DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
   const unsigned int n_u = 2*dofs_per_component[0],
                      n_p_interior = dofs_per_component[dim],
                      n_p_face = dofs_per_component[dim+1],
                      n_p = dofs_per_component[dim] + dofs_per_component[dim+1];

//   std::vector<unsigned int> block_component (dim+1,0);
//   block_component[dim] = 1;
//   DoFRenumbering::component_wise (dof_handler, block_component);
//   std::vector<types::global_dof_index> dofs_per_block (2);
//   DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
//   const unsigned int n_u = dofs_per_block[0],
//                      n_p = dofs_per_block[1];


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
   DoFTools::make_sparsity_pattern (dof_handler, dsp);

   sparsity_pattern.copy_from(dsp);
   system_matrix.reinit (sparsity_pattern);
//
   solution.reinit (2);
   solution.block(0).reinit (n_u);
   solution.block(1).reinit (n_p);
   solution.collect_sizes ();

//   old_solution_displacement.reinit(n_u);
//   old_solution_pressure.reinit(n_p);
   old_solution.reinit (2);
   old_solution.block(0).reinit(n_u);
   old_solution.block(1).reinit (n_p);
   old_solution.collect_sizes ();

   system_rhs.reinit (2);
   system_rhs.block(0).reinit (n_u);
   system_rhs.block(1).reinit (n_p);
   system_rhs.collect_sizes ();     // not sure about rhs.
 }

 // From step-26, this function is the one that solves the actual linear system
 // for a single time step.
// template<int dim>
//  void PoroElasBR1WG<dim>::solve_time_step()
//   {
//     SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
//     SolverCG<BlockVector<double>> cg(solver_control);
//
//     cg.solve(system_matrix, solution, system_rhs,
//    		 PreconditionIdentity());
//
////     constraints.distribute(solution); // needed?
//
//     std::cout << "     " << solver_control.last_step()
//               << " CG iterations." << std::endl;
//   }

 // @sect4{WGDarcyEquation::assemble_system}

 // First, we allocate quadrature points and <code>FEValues</code> for cells and faces.
 // Then we allocate space for all cell matrices and right hand side.
 // The following definitions have been explained in previous tutorials.
 template <int dim>
 void PoroElasBR1WG<dim>::assemble_system ()
 {
   QGauss<dim>  quadrature_formula(fe_rt.degree+1);
   QGauss<dim-1>  face_quadrature_formula(fe_rt.degree+1);
//   const RightHandSide<dim> right_hand_side;

   // We define objects to evaluate values, gradients of shape functions at the quadrature points.
   // Since we need shape functions and normal vectors on faces, we need <code>FEFaceValues</code>.
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
//   const unsigned int   dofs_per_cell_pressure = // need this one



   const unsigned int   n_q_points    = fe_values.get_quadrature().size();
   const unsigned int   n_q_points_rt = fe_values_rt.get_quadrature().size();
   const unsigned int   n_face_q_points = fe_face_values.get_quadrature().size();
//   const unsigned int   n_face_q_points_rt = fe_face_values_rt.get_quadrature().size();

   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

   // We will construct these cell matrices to solve for the pressure.
   FullMatrix<double>   cell_matrix_rt (dofs_per_cell_rt,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_F (dofs_per_cell,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_C (dofs_per_cell,dofs_per_cell_rt);
   FullMatrix<double>   local_matrix (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   local_matrix_DarcyStf (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   local_matrix_DarcyMass (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   local_matrix_Darcy(dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   local_matrix_ElasStrain (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   local_matrix_DarcyElas (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   local_matrix_ElasDarcy (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   cell_matrix_D (dofs_per_cell_rt,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_E (dofs_per_cell_rt,dofs_per_cell_rt);
   Vector<double>       cell_rhs (dofs_per_cell);
   Vector<double>       cell_solution (dofs_per_cell);

   const Coefficient<dim> coefficient;
   std::vector<Tensor<2,dim>> coefficient_values (n_q_points_rt);

   const FEValuesExtractors::Vector velocities (0); // this is for fe_rt velcocities.
   const FEValuesExtractors::Vector displacements (0); //
   const FEValuesExtractors::Scalar pressure_interior (dim); // this is for pressure
   const FEValuesExtractors::Scalar pressure_face (dim+1); // maybe it is with this dim.

   typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler.begin_active(),
   endc = dof_handler.end();
   typename DoFHandler<dim>::active_cell_iterator
   cell_rt = dof_handler_rt.begin_active();

   for (; cell!=endc; ++cell,++cell_rt)
     {
	   // On each cell, cell matrices are different, so in every loop, they need to be re-computed.
       fe_values_rt.reinit (cell_rt);
       fe_values.reinit (cell);
       cell_matrix_rt = 0;
       cell_matrix_F = 0;
       cell_matrix_C = 0;
       local_matrix = 0;
//       local_matrix_DarcyStf = 0;
//       local_matrix_DarcyMass = 0;
//       local_matrix_ElasStrain = 0;
//       local_matrix_DarcyElas = 0;
//       local_matrix_ElasDarcy = 0;
//       local_matrix_Darcy = 0;
       cell_rhs = 0;
       coefficient.value_list (fe_values_rt.get_quadrature_points(),
                               coefficient_values);

       // This cell matrix is the integral of all basis functions of <code>FE_RaviartThomas</code>.
       // The loop is over all quadrature points defined on <code>FE_RaviartThomas</code>.
       // Next we take the inverse of this matrix by using <code>gauss_jordan()</code>.
       for (unsigned int q=0; q<n_q_points_rt; ++q)
       {

    	   for (unsigned int i=0; i<dofs_per_cell_rt; ++i)
           {
    		 const Tensor<1,dim> phi_i_u = fe_values_rt[velocities].value (i, q);
//    		 const double phi_i_u_div = fe_values_rt[velocities].divergence(i,q);
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


     // Construct the local matrix.
       for (unsigned int q=0; q<n_q_points_rt; ++q)
           {
       	     for(unsigned int i = 0; i<dofs_per_cell; ++i)
       	        {
       	    	const SymmetricTensor<2,dim> phi_i_symmgrad
       	    	              = fe_values[displacements].symmetric_gradient (i,q);
       	    	const double phi_i_div
       	    	              = fe_values[displacements].divergence (i,q);
       		     for(unsigned int j = 0; j<dofs_per_cell; ++j)
       		        {
       		    	const SymmetricTensor<2,dim> phi_j_symmgrad
       		    	                  = fe_values[displacements].symmetric_gradient (j,q);
       		    	const double phi_j_div
       		    	                  = fe_values[displacements].divergence (j,q);
       			       for(unsigned int k = 0; k<dofs_per_cell_rt; ++k)
       			          {
       				        const Tensor<1,dim> phi_k_u = fe_values_rt[velocities].value (k, q);
       				        for(unsigned int l = 0; l<dofs_per_cell_rt; ++l)
       				           {
       					         const Tensor<1,dim> phi_l_u = fe_values_rt[velocities].value (l, q);
       					         local_matrix(i,j) += (2 *1*(phi_i_symmgrad * phi_j_symmgrad) +
                    		                           1*(phi_i_div * phi_j_div) -
                                                       1*(fe_values[pressure_interior].value (i,q)* phi_j_div) +
                                                       1*(fe_values[pressure_interior].value(i,q) *
                                                       fe_values[pressure_interior].value(j,q)) +
       					        		               1*coefficient_values[q]*cell_matrix_C[i][k]*cell_matrix_C[j][l]*
       							                       phi_k_u*phi_l_u +
       							                       1*(phi_i_div*fe_values[pressure_interior].value (j,q)))*
       							                       fe_values_rt.JxW(q);

       				            }
       			           }
       		         }
       	         }
             }
       for (unsigned int i=0; i<dofs_per_cell; ++i)
                     for (unsigned int j=0; j<dofs_per_cell; ++j)
                         {
       std::cout<< "i " << i << " j "<<j<<" "<< local_matrix(i,j)<<std::endl;
                         }

          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                  {
                    system_matrix.add(local_dof_indices[i],
                                      local_dof_indices[j],
                                      local_matrix(i,j));
                  }

//     // In the local matrix of this element, component is
//     // $\int_{E} \mathbf{K} c_{ik} c_{jl} \mathbf{w}_k \cdot \mathbf{w}_l.$
//     // We have calculated coefficients $c$ from the previous step.
//     // And these basis functions $\mathbf{w}$ are defined in the interior.
//     // This is for Darcy stiff. matrix.
//       for (unsigned int q=0; q<n_q_points_rt; ++q)
//           {
//	         for(unsigned int i = 0; i<dofs_per_cell; ++i)
//	            {
//		          for(unsigned int j = 0; j<dofs_per_cell; ++j)
//		             {
//			           for(unsigned int k = 0; k<dofs_per_cell_rt; ++k)
//			              {
//				            const Tensor<1,dim> phi_k_u = fe_values_rt[velocities].value (k, q);
//				            for(unsigned int l = 0; l<dofs_per_cell_rt; ++l)
//				               {
//					             const Tensor<1,dim> phi_l_u = fe_values_rt[velocities].value (l, q);
//					             local_matrix_DarcyStf(i,j) += coefficient_values[q]*cell_matrix_C[i][k]*cell_matrix_C[j][l]*
//							                                   phi_k_u*phi_l_u*
//							                                   fe_values_rt.JxW(q);
//				                }
//			               }
//		              }
//	             }
//             }
//
//       // Now, for mass matrix in Darcy part.
//       for (unsigned int q=0; q<n_q_points_rt; ++q)
//            {
//       	       for(unsigned int i = 0; i<dofs_per_cell; ++i)
//       	          {
//       		        for(unsigned int j = 0; j<dofs_per_cell; ++j)
//       		           {
//       		        	local_matrix_DarcyMass(i,j) += (fe_values[pressure_interior].value(i,q) *
//       		        			                        fe_values[pressure_interior].value(j,q)*
//       		        	   	                            fe_values.JxW (q));
//       		           }
//       	          }
//            }
//       local_matrix_Darcy = local_matrix_DarcyMass + local_matrix_DarcyStf;
//      // Now, for Elasticity strain-strain
//      // But, need to take average for phi_i_div and phi_j_div
//      // have to do this later!!!!
//       for (unsigned int q=0; q<n_q_points; ++q)
//        for (unsigned int i=0; i<dofs_per_cell; ++i)
//          {
//            const SymmetricTensor<2,dim> phi_i_symmgrad
//              = fe_values[displacements].symmetric_gradient (i,q);
//            const double phi_i_div
//              = fe_values[displacements].divergence (i,q);
//            for (unsigned int j=0; j<dofs_per_cell; ++j)
//              {
//                const SymmetricTensor<2,dim> phi_j_symmgrad
//                  = fe_values[displacements].symmetric_gradient (j,q);
//                const double phi_j_div
//                  = fe_values[displacements].divergence (j,q);
//                local_matrix_ElasStrain(i,j) += (2 *1*(phi_i_symmgrad * phi_j_symmgrad) +
//                		             1*(phi_i_div * phi_j_div)) *
//                                     fe_values.JxW(q);
//              }
//           }
//       // Now, for Darcy-Elasticity, local_matrix_DarcyElas
//       // Later, need to take average of divergence!!!!
//       for (unsigned int q=0; q<n_q_points; ++q)
//           for (unsigned int i=0; i<dofs_per_cell; ++i)
//               {
//                 for (unsigned int j=0; j<dofs_per_cell; ++j)
//                     {
//                	   const double phi_j_div = fe_values[displacements].divergence (j,q);
//                       local_matrix_DarcyElas(i,j) -=  1*(fe_values[pressure_interior].value (i,q)*
//                    		                           phi_j_div)*
//                                                       fe_values.JxW(q);
//                     }
//                 }
//       // Now, for Elasticity-Darcy, local_matrix_ElasDarcy
//       // same as previous, need to do average later!
//       for (unsigned int q=0; q<n_q_points; ++q)
//            for (unsigned int i=0; i<dofs_per_cell; ++i)
//                {
//               	  const double phi_i_div = fe_values[displacements].divergence (i,q);
//                  for (unsigned int j=0; j<dofs_per_cell; ++j)
//                      {
//                        local_matrix_ElasDarcy(i,j) +=  1*(phi_i_div*
//                           		                        fe_values[pressure_interior].value (j,q)) *
//                                                        fe_values.JxW(q);
//                      }
//                }
//       //
//       cell->get_dof_indices (local_dof_indices);
//       for (unsigned int i=0; i<dofs_per_cell; ++i)
//           for (unsigned int j=0; j<dofs_per_cell; ++j)
//               {
//                 system_matrix.add(local_dof_indices[i],
//                                   local_dof_indices[j],
//                                   local_matrix_ElasStrain(i,j));
//                 system_matrix.add(local_dof_indices[i],
//                                   local_dof_indices[j],
//                                   local_matrix_Darcy(i,j));
//                 system_matrix.add(local_dof_indices[i],
//                                   local_dof_indices[j],
//                                   local_matrix_ElasDarcy(i,j));
//                 system_matrix.add(local_dof_indices[i],
//                                   local_dof_indices[j],
//                                   local_matrix_DarcyElas(i,j));
//               }
    }
//   for (unsigned int i=0; i<13; ++i)
//                 for (unsigned int j=0; j<13; ++j)
//                     {
//                	 std::cout<<"system" <<std::endl;
//                	 std::cout<< "i " << i << " j "<<j<<" "<< system_matrix(i,j)<<std::endl;
//                     }

 }

 // @sect4{WGDarcyEquation<dim>::solve}

 // Solving the system of the Darcy equation. Now, we have pressures in interiors and on faces.
// template <int dim>
// void PoroElasBR1WG<dim>::solve ()
// {
//   SolverControl           solver_control (1000, 1e-8 * system_rhs.l2_norm());
//   SolverCG<>              solver (solver_control);
//   solver.solve (system_matrix, solution, system_rhs,
//                 PreconditionIdentity());
// }

 // @sect4{WGDarcyEquation::run}

 // This is the final function of the main class. It calls the other functions of our class.
 template <int dim>
 void PoroElasBR1WG<dim>::run ()
 {
   std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
   make_grid_and_dofs();
//   VectorTools::project (dof_handler, constraints, QGauss<dim>(3),
//                         InitialValuesDisplacement<dim>(),
//                         old_solution_displacement);
//   VectorTools::project (dof_handler, constraints, QGauss<dim>(3),
//                         InitialValuesPressure<dim>(),
//                         old_solution_pressure);
//   VectorTools::project (dof_handler, constraints, QGauss<dim>(3),
//                         InitialValues<dim>(),
//                         old_solution);
//   solve_time_step();
   assemble_system ();
//   solve ();
 }

 // @sect3{The <code>main</code> function}

 // This is the main function. We can change the dimension here to run in 3d.
 int main ()
 {
   deallog.depth_console (2);
   PoroElasBR1WG<2> PoroElasBR1WGTest;
   PoroElasBR1WGTest.run ();

   return 0;
 }

