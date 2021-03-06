#ifndef UTILS_HPP
#define UTILS_HPP

#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <limits>
#include "Assert.h"
#include "Random.h"

#ifdef USE_BOOST_PYTHON
#include "boost/shared_ptr.hpp"
#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"
#endif

using namespace std;


inline void GetMaxMin(const vector<double>& a_Vals, double& a_Min, double& a_Max)
{
    a_Max = std::numeric_limits<double>::min();
    a_Min = std::numeric_limits<double>::max();
    for(vector<double>::const_iterator t_It = a_Vals.begin(); t_It != a_Vals.end(); ++t_It)
    {
        const double t_CurrentVal = (*t_It);
        if (t_CurrentVal > a_Max) a_Max = t_CurrentVal;

        if (t_CurrentVal < a_Min) a_Min = t_CurrentVal;
    }
}

//converts an integer to a string
inline std::string itos(const int a_Arg)
{
    std::ostringstream t_Buffer;

    //send the int to the ostringstream
    t_Buffer << a_Arg;

    //capture the string
    return t_Buffer.str();
}

//converts a double to a string
inline std::string ftos(const double a_Arg)
{
    std::ostringstream t_Buffer;

    //send the int to the ostringstream
    t_Buffer << a_Arg;

    //capture the string
    return t_Buffer.str();
}

//clamps the first argument between the second two
inline void Clamp(double &a_Arg, const double a_Min, const double a_Max)
{
    ASSERT(a_Min <= a_Max);

    if (a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }

    if (a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

//clamps the first argument between the second two
inline void Clamp(float &a_Arg, const double a_Min, const double a_Max)
{
    ASSERT(a_Min <= a_Max);

    if (a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }

    if (a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

//clamps the first argument between the second two
inline void Clamp(int &a_Arg, const int a_Min, const int a_Max)
{
    ASSERT(a_Min <= a_Max);

    if (a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }

    if (a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

//rounds a double up or down depending on its value
inline int Rounded(const double a_Val)
{
    const int t_Integral = static_cast<int>(a_Val);
    const double t_Mantissa = a_Val - t_Integral;

    if (t_Mantissa < 0.5)
    {
        return t_Integral;
    }

    else
    {
        return t_Integral + 1;
    }
}

//rounds a double up or down depending on whether its
//mantissa is higher or lower than offset
inline int RoundUnderOffset(const double a_Val, const double a_Offset)
{
    //ASSERT(a_Offset < 1 && a_Offset > -1); ???!? Should this be a test for the offset
    const int t_Integral = static_cast<int>(a_Val);
    const double t_Mantissa = a_Val - t_Integral;

    if (t_Mantissa < a_Offset)
    {
        return t_Integral;
    }
    else
    {
        return t_Integral + 1;
    }
}


// Scales the value "a", that is in range [a_min .. a_max] into its relative value in the range [tr_min .. tr_max]
// Example: A=2, in the range [0 .. 4] .. we want to scale it to the range [-12 .. 12] .. we get 0..
inline void Scale(    double& a,
                    const double a_min,
                    const double a_max,
                    const double a_tr_min,
                    const double a_tr_max)
{
//        ASSERT((a >= a_min) && (a <= a_max));
//        ASSERT(a_min <= a_max);
//        ASSERT(a_tr_min <= a_tr_max);

    const double t_a_r = a_max - a_min;
    const double t_r = a_tr_max - a_tr_min;
    const double rel_a = (a - a_min) / t_a_r;
    a = a_tr_min + t_r * rel_a;
}

// Scales the value "a", that is in range [a_min .. a_max] into its relative value in the range [tr_min .. tr_max]
// Example: A=2, in the range [0 .. 4] .. we want to scale it to the range [-12 .. 12] .. we get 0..
inline void Scale(    float& a,
                    const double a_min,
                    const double a_max,
                    const double a_tr_min,
                    const double a_tr_max)
{
//        ASSERT((a >= a_min) && (a <= a_max));
//        ASSERT(a_min <= a_max);
//        ASSERT(a_tr_min <= a_tr_max);

    const double t_a_r = a_max - a_min;
    const double t_r = a_tr_max - a_tr_min;
    const double rel_a = (a - a_min) / t_a_r;
    a = a_tr_min + t_r * rel_a;
}

inline void Scale(vector<double>& a_Values, const double a_tr_min, const double a_tr_max)
{
    double t_max = std::numeric_limits<double>::min(), t_min = std::numeric_limits<double>::max();
    GetMaxMin(a_Values, t_min, t_max);
    vector<double> t_ValuesScaled;
    for(vector<double>::const_iterator t_It = a_Values.begin(); t_It != a_Values.end(); ++t_It)
    {
        double t_ValueToBeScaled = (*t_It);
        Scale(t_ValueToBeScaled, t_min, t_max, 0, 1); // !!!!!!!!!!!!!!!!??????????
        t_ValuesScaled.push_back(t_ValueToBeScaled);
    }

    a_Values = t_ValuesScaled;
}

inline double Abs(double x)
{
	if (x<0)
	{
		return -x;
	}
	else
	{
		return x;
	}
}



#ifdef USE_BOOST_PYTHON
namespace py = boost::python;
using namespace py;


template<typename T>
inline
std::vector< T > py_list_to_std_vector( const boost::python::object& iterable )
{
    return std::vector< T >( py::stl_input_iterator< T >( iterable ),
                             py::stl_input_iterator< T >( ) );
}

template <class T>
inline
py::list std_vector_to_py_list(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    py::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}

template <class C>
struct VectorPickleSuite: public py::pickle_suite { BOOST_STATIC_ASSERT(sizeof(C)==0); };

template <typename  T>
struct VectorPickleSuite < std::vector<T> >: public py::pickle_suite
{
    static py::tuple getinitargs(const std::vector<T>& o)
    {
        return py::make_tuple();
    }

    static py::tuple getstate(py::object obj)
    {
        const std::vector<T>& o = py::extract<const std::vector<T>&>(obj)();

        return py::make_tuple(py::list(o));
    }

    static void setstate(py::object obj, py::tuple state)
    {
        std::vector<T>& o = py::extract<std::vector<T>&>(obj)();

        py::stl_input_iterator<typename std::vector<T>::value_type> begin(state[0]), end;
        o.insert(o.begin(),begin,end);
    }
};
#endif


#endif

