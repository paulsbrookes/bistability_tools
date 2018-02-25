cdef extern from "gsl/gsl_sf_legendre.h":
  double gsl_sf_legendre_P1(double x)


cdef extern from "math.h":
    double sin(double x)


cdef extern from "gsl/gsl_sf_result.h":
  ctypedef struct gsl_sf_result:
    double val
    double err


cdef extern from "gsl/gsl_sf_mathieu.h":
  int gsl_sf_mathieu_a(int order, double qq, gsl_sf_result * result);


def mathieu_a(int order, double qq):
    cdef gsl_sf_result result
    cdef int outcome
    outcome = gsl_sf_mathieu_a(order,qq,&result)
    return result.val
