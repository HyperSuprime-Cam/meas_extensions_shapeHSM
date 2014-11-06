// This is a dummy include file that re-implements the "Template Matrix/Vector"
// (TMV) package by Mike Jarvis (http://code.google.com/p/tmv-cpp/) using ndarray
// and Eigen.  This allows us to use the HSM code from GalSim without installing
// TMV.  Changes to the original TMV's APIs used by HSM will need to be reflected
// in this TMV.h file.

#ifndef TMV_H
#define TMV_H

#include "ndarray.h"
#include "ndarray/types.h"
#include "ndarray/eigen.h"

namespace tmv {

// Forward declarations
template <typename T> class Vector;
template <typename T> class Matrix;
template <typename T> class MatrixView;
template <typename T> class ConstMatrixView;
template <typename T> std::ostream & operator<<(std::ostream & os, Vector<T> const& thing);
template <typename T> std::ostream & operator<<(std::ostream & os, Matrix<T> const& thing);
template <typename T> std::ostream & operator<<(std::ostream & os, MatrixView<T> const& thing);
template <typename T> std::ostream & operator<<(std::ostream & os, ConstMatrixView<T> const& thing);

/// Dummy enum.
///
/// HSM only uses NonConj, so it doesn't seem to have any particular meaning except
/// to satisfy the API.
enum Dummy {
    NonConj,
};

/// A vector
template <typename T>
class Vector {
    typedef ndarray::Array<T,1,1> Base;
public:
    /// Ctor
    Vector(size_t n) : _base(ndarray::allocate(n)) {}
    Vector(size_t n, T const value) : _base(ndarray::allocate(n)) { _base.deep() = value; }

    /// Vector indexing
    T& operator[](int x) {
        return _base[x];
    }
    T const& operator[](int x) const {
        return _base[x];
    }

    /// Pointer to constant data
    T const* cptr() const { return _base.getData(); }
    /// Pointer to data
    T* ptr() const { return _base.getData(); }

    friend std::ostream& operator<< <>(std::ostream&, Vector<T> const&);
private:
    Base _base;
};

/// A matrix
template <typename T>
class Matrix {
    typedef ndarray::Array<T,2,0> Base;
public:
    /// Ctor
    Matrix(size_t nx, size_t ny) : _base(ndarray::allocate(nx, ny)) {}
    Matrix(size_t nx, size_t ny, T const value) : _base(ndarray::allocate(nx, ny)) { _base.deep() = value; }
    Matrix(Matrix<T> const& other) : _base(other._base) {}
    template <typename U, int N, int V>
    Matrix(ndarray::ArrayRef<U,N,V> const& other) : _base(other) {}
    Matrix(Base const& base) : _base(base) {}

    /// Matrix indexing
    T& operator()(int x, int y) {
        return _base[x][y];
    }
    T const& operator()(int x, int y) const {
        return _base[x][y];
    }
    typename Base::Reference operator[](int x) const {
        return _base[x];
    }

    /// Assignment
    Matrix<T>& operator=(Matrix<T> const& rhs) {
        if (this != &rhs) {
            _base = rhs._base;
        }
        return *this;
    }
    Matrix<T>& operator=(ConstMatrixView<T> const& rhs) { return _base = rhs._base; }

    /// Matrix multiplication
    Matrix<T> operator*(Matrix<T> const& other) const {
        // This does an allocation to hold the temporary, which may not be necessary with a bit of work
        Base result = ndarray::allocate(this->_base.getShape()[0], other._base.getShape()[1]);
        result.asEigen() = this->_base.asEigen() * other._base.asEigen();
        return result;
    }

    /// Matrix transpose
    ConstMatrixView<T> transpose() const {
        typename Base::Index const shape = _base.getShape(), stride = _base.getStrides();
        return ConstMatrixView<T>(_base.getData(), shape[1], shape[0], stride[1], stride[0], NonConj);
    }

    friend std::ostream& operator<< <>(std::ostream&, Matrix<T> const&);
protected:
    friend class MatrixView<T>;
    friend class ConstMatrixView<T>;
    Base _base;
};

/// A view into a Matrix
template <typename T>
class MatrixView {
    typedef ndarray::Array<T,2,0> Base;
public:
    /// Ctor
    ///
    /// @param nx,ny: Number of elements in x and y
    /// @param sx,sy: Stride in x and y
    /// @param Dummy: put NonConj here
    MatrixView(T* data, size_t nx, size_t ny, size_t sx, size_t sy, Dummy) :
        _base(ndarray::external<T, 2>(data, ndarray::makeVector<int>(nx, ny),
                                      ndarray::makeVector<int>(sx, sy))) {}
    MatrixView(Base& base) : _base(base) {}

    /// Unimplemented methods
    ///
    /// These are only used in debugging prints
    std::string maxAbsElement() const { return "UNIMPLEMENTED"; }
    std::string subMatrix(int, int, int, int) const { return "UNIMPLEMENTED"; }

    friend std::ostream& operator<< <>(std::ostream&, MatrixView<T> const&);
private:
    friend class ConstMatrixView<T>;

    Base _base;
};

/// A constant view into a Matrix
template <typename T>
class ConstMatrixView {
    typedef ndarray::Array<T const,2,0> Base;
public:
    /// Ctor
    ///
    /// @param nx,ny: Number of elements in x and y
    /// @param sx,sy: Stride in x and y
    /// @param Dummy: put NonConj here
    ConstMatrixView(T const* data, size_t nx, size_t ny, size_t sx, size_t sy, Dummy) :
        _base(ndarray::external<T const, 2>(data, ndarray::makeVector<int>(nx, ny),
                                            ndarray::makeVector<int>(sx, sy))) {}
    ConstMatrixView(Base const& base) : _base(base) {}
    ConstMatrixView(MatrixView<T> const& other) : _base(other._base) {}

    /// Matrix multiplication
    Matrix<T> operator*(ConstMatrixView<T> const& other) const {
        // This does an allocation to hold the temporary, which may not be necessary with a bit of work
        ndarray::Array<T,2,0> result = ndarray::allocate(this->_base.getShape()[0], other._base.getShape()[1]);
        result.asEigen() = this->_base.asEigen() * other._base.asEigen();
        return result;
    }

    friend std::ostream& operator<< <>(std::ostream&, ConstMatrixView<T> const&);
private:
    friend class MatrixView<T>;
    Base const _base;
};

/// Stream output
template <typename T>
std::ostream & operator<<(std::ostream & os, Vector<T> const& vector) {
    return os << vector._base.getShape() << std::endl << vector._base.asEigen() << std::endl;
}
template <typename T>
std::ostream & operator<<(std::ostream & os, Matrix<T> const& matrix) {
    return os << matrix._base.getShape() << std::endl << matrix._base.asEigen() << std::endl;
}
template <typename T>
std::ostream & operator<<(std::ostream & os, MatrixView<T> const& matrix) {
    return os << matrix._base.getShape() << std::endl << matrix._base.asEigen() << std::endl;
}
template <typename T>
std::ostream & operator<<(std::ostream & os, ConstMatrixView<T> const& matrix) {
    return os << matrix._base.getShape() << std::endl << matrix._base.asEigen() << std::endl;
}

} // namespace tmv

#endif
