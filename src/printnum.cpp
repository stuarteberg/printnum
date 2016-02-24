#include <iostream>
#include <boost/python.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL printnum_cpp_module_PyArray_API
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>


/*
 * Boost python converter for numpy scalars, e.g. numpy.uint32(123).
 * Enables automatic conversion from numpy.intXX, floatXX
 * in python to C++ char, short, int, float, etc.
 * When casting from float to int (or wide int to narrow int),
 * normal C++ casting rules apply.
 *
 * Like all boost::python converters, this enables automatic conversion for function args
 * exposed via boost::python::def(), as well as values converted via boost::python::extract<>().
 *
 * Copied from the VIGRA C++ library source code (MIT license).
 * http://ukoethe.github.io/vigra
 * https://github.com/ukoethe/vigra
 */
template <typename ScalarType>
struct NumpyScalarConverter
{
    NumpyScalarConverter()
    {
        using namespace boost::python;
        converter::registry::push_back( &convertible, &construct, type_id<ScalarType>());
    }

    // Determine if obj_ptr is a supported numpy.number
    static void* convertible(PyObject* obj_ptr)
    {
        if (PyArray_IsScalar(obj_ptr, Float32) ||
            PyArray_IsScalar(obj_ptr, Float64) ||
            PyArray_IsScalar(obj_ptr, Int8)    ||
            PyArray_IsScalar(obj_ptr, Int16)   ||
            PyArray_IsScalar(obj_ptr, Int32)   ||
            PyArray_IsScalar(obj_ptr, Int64)   ||
            PyArray_IsScalar(obj_ptr, UInt8)   ||
            PyArray_IsScalar(obj_ptr, UInt16)  ||
            PyArray_IsScalar(obj_ptr, UInt32)  ||
            PyArray_IsScalar(obj_ptr, UInt64))
        {
            return obj_ptr;
        }
        return 0;
    }

    static void construct( PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        using namespace boost::python;

        // Grab pointer to memory into which to construct the C++ scalar
        void* storage = ((converter::rvalue_from_python_storage<ScalarType>*) data)->storage.bytes;

        // in-place construct the new scalar value
        ScalarType * scalar = new (storage) ScalarType;

        if (PyArray_IsScalar(obj_ptr, Float32))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Float32);
        else if (PyArray_IsScalar(obj_ptr, Float64))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Float64);
        else if (PyArray_IsScalar(obj_ptr, Int8))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Int8);
        else if (PyArray_IsScalar(obj_ptr, Int16))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Int16);
        else if (PyArray_IsScalar(obj_ptr, Int32))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Int32);
        else if (PyArray_IsScalar(obj_ptr, Int64))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Int64);
        else if (PyArray_IsScalar(obj_ptr, UInt8))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, UInt8);
        else if (PyArray_IsScalar(obj_ptr, UInt16))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, UInt16);
        else if (PyArray_IsScalar(obj_ptr, UInt32))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, UInt32);
        else if (PyArray_IsScalar(obj_ptr, UInt64))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, UInt64);

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};

/*
 * A silly function to test scalar conversion.
 * The first arg tests automatic function argument conversion.
 * The second arg is used to demonstrate explicit conversion via boost::python::extract<>()
 */
void print_number( uint32_t number, boost::python::object other_number )
{
    using namespace boost::python;
    std::cout << "The number is: " << number << std::endl;
    std::cout << "The other number is: " << extract<int16_t>(other_number) << std::endl;
}

/*
 * Instantiate the python extension module 'printnum'.
 *
 * Example Python usage:
 *
 *     import numpy as np
 *     from printnum import print_number
 *     print_number( np.uint8(123), np.int64(-456) )
 *
 *     ## That prints the following:
 *     # The number is: 123
 *     # The other number is: -456
 */
BOOST_PYTHON_MODULE(printnum)
{
    using namespace boost::python;

    // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
    import_array();

    // Register conversion for all scalar types.
    NumpyScalarConverter<signed char>();
    NumpyScalarConverter<short>();
    NumpyScalarConverter<int>();
    NumpyScalarConverter<long>();
    NumpyScalarConverter<long long>();
    NumpyScalarConverter<unsigned char>();
    NumpyScalarConverter<unsigned short>();
    NumpyScalarConverter<unsigned int>();
    NumpyScalarConverter<unsigned long>();
    NumpyScalarConverter<unsigned long long>();
    NumpyScalarConverter<float>();
    NumpyScalarConverter<double>();

    // Expose our C++ function as a python function.
    def("print_number", &print_number, (arg("number"), arg("other_number")));
}
