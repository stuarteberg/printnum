#include <iostream>
#include <boost/python.hpp>
#include <boost/cstdint.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL printnum_cpp_module_PyArray_API
#include <numpy/arrayobject.h>

using boost::uint8_t;
using boost::uint16_t;
using boost::uint32_t;
using boost::uint64_t;

using boost::int8_t;
using boost::int16_t;
using boost::int32_t;
using boost::int64_t;

template <typename ScalarType>
std::string dtype_name() { return "UNKNOWN DTYPE"; }

template <> std::string dtype_name<uint8_t>()  { return "uint8"; }
template <> std::string dtype_name<uint16_t>() { return "uint16"; }
template <> std::string dtype_name<uint32_t>() { return "uint32"; }
template <> std::string dtype_name<uint64_t>() { return "uint64"; }

template <> std::string dtype_name<int8_t>()  { return "int8"; }
template <> std::string dtype_name<int16_t>() { return "int16"; }
template <> std::string dtype_name<int32_t>() { return "int32"; }
template <> std::string dtype_name<int64_t>() { return "int64"; }

template <typename ScalarType>
struct NumpyScalarConverter
{
    NumpyScalarConverter()
    {
        using namespace boost::python;
        converter::registry::push_back( &convertible, &construct, type_id<ScalarType>());
    }

    // Determine if obj_ptr is a numpy.number
    static void* convertible(PyObject* obj_ptr)
    {
        using namespace boost::python;
        object arg = object(handle<>(borrowed(obj_ptr)));
        object isinstance = import("__builtin__").attr("isinstance");
        object numpy_number = import("numpy").attr("number");
        if ( isinstance(arg, numpy_number) )
        {
            return obj_ptr;
        }
        return 0;
    }

    static void construct( PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        using namespace boost::python;

        // Grab pointer to memory into which to construct the std::string
        void* storage = ((converter::rvalue_from_python_storage<ScalarType>*) data)->storage.bytes;

        // in-place construct the new value
        // extracted from the python object
        ScalarType * scalar = new (storage) ScalarType;

        object numpy = import("numpy");
        object original_scalar = object(handle<>(borrowed(obj_ptr)));
        object resized_scalar = numpy.attr(dtype_name<ScalarType>().c_str())(original_scalar);
        object bytes = resized_scalar.attr("tobytes")();

        Py_buffer py_buffer;
        PyObject_GetBuffer(bytes.ptr(), &py_buffer, PyBUF_SIMPLE);

        (*scalar) = *(static_cast<ScalarType*>(py_buffer.buf));
        PyBuffer_Release(&py_buffer);

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};

void print_number( uint32_t number, boost::python::object other_number )
{
    using namespace boost::python;
    std::cout << "The number is: " << number << std::endl;
    std::cout << "The other number is: " << extract<uint32_t>(other_number) << std::endl;
}

BOOST_PYTHON_MODULE(printnum)
{
    using namespace boost::python;

    // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
    import_array();

    NumpyScalarConverter<uint8_t>();
    NumpyScalarConverter<uint16_t>();
    NumpyScalarConverter<uint32_t>();
    NumpyScalarConverter<uint64_t>();

    NumpyScalarConverter<int8_t>();
    NumpyScalarConverter<int16_t>();
    NumpyScalarConverter<int32_t>();
    NumpyScalarConverter<int64_t>();


    def("print_number", &print_number, (arg("number")));
}
