#include <Python.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>


static PyObject *
distribution_normal_logpdf(PyObject *self, PyObject *args) {
  double x;
  double loc;
  double scale;
  if (!PyArg_ParseTuple(args, "ddd", &x, &loc, &scale))
        return NULL;
 return PyFloat_FromDouble(-0.9189385332046727 - log(scale) - pow(x-loc,2) / (2*pow(scale,2)));
}

static PyObject *
distribution_truncated_normal_logpdf(PyObject *self, PyObject *args) {
  double x;
  double loc;
  double scale;
  double a;
  double b;

  double a1;
  double a2;
  double a3;

  if (!PyArg_ParseTuple(args, "ddddd", &x, &loc, &scale, &a, &b))
        return NULL;
  if (x < a || a > b){
    return PyFloat_FromDouble(-INFINITY);
  }
  if (x > b){
    return PyFloat_FromDouble(-INFINITY);
  }
  a1 = (x - loc) / scale;
  a2 = (b - loc) / scale;
  a3 = (a - loc) / scale;

  return PyFloat_FromDouble(M_LN2 - 0.9189385332046727 - 0.5*pow(a1,2) - log(scale) - log(erf(a2/M_SQRT2) - erf(a3/M_SQRT2)));
}

static PyObject *
distribution_lognormal_logpdf(PyObject *self, PyObject *args) {
  double x;
  double mu;
  double sigma;

  if (!PyArg_ParseTuple(args, "ddd", &x, &mu, &sigma))
    return NULL;
  if (x <= 0) {
    return PyFloat_FromDouble(-INFINITY);
  }
  return PyFloat_FromDouble(-0.9189385332046727 - log(x) - log(sigma) - pow(log(x) - mu, 2) / (2*pow(sigma,2)));

}

static PyMethodDef DistributionMethods[] = {
  {"normal_logpdf", distribution_normal_logpdf, METH_VARARGS, "logpdf"},
  {"truncated_normal_logpdf", distribution_truncated_normal_logpdf, METH_VARARGS, "logpdf"},
  {"lognormal_logpdf", distribution_lognormal_logpdf, METH_VARARGS, "lognormal logpdf"},
  {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef distributionmodule = {
  PyModuleDef_HEAD_INIT,
  "distribution",
  NULL,
  -1,
  DistributionMethods
};

PyMODINIT_FUNC
PyInit_distribution(void){
  return PyModule_Create(&distributionmodule);
}

int
main(int argc, char *argv[]) {
  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
  if (program == NULL) {
      fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
      exit(1);
  }

  /* Add a built-in module, before Py_Initialize */
  PyImport_AppendInittab("distribution", PyInit_distribution);

  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Optionally import the module; alternatively,
     import can be deferred until the embedded script
     imports it. */
  PyImport_ImportModule("distribution");

  PyMem_RawFree(program);
  return 0;
}
