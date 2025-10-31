#ifndef GEOMETRYVECTOR_INCLUDED
#define GEOMETRYVECTOR_INCLUDED

#include <cstring>
#include <cassert>
#include <fstream>
#include <cmath>

#ifndef NDEBUG
#define GEOMETRYVECTOR_RECORDDIMENSION
#endif

const short MaxDimension=4;

//In debug mode, this class also records the dimension of the GeometryVector.
class GeometryVector
{
public:
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
	short Dimension;
#endif
	void SetDimension(short d)
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		this->Dimension=d;
#endif
	}

	double x[ ::MaxDimension];
	GeometryVector()
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		this->Dimension=0;
#endif
		for(int i=0; i< ::MaxDimension; i++)
			this->x[i]=0;
	}
	//do not change this "int" to "short"!
	//otherwize GeometryVector(2) would be ambiguous.
	GeometryVector(int dimension)
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		this->Dimension=dimension;
#endif
		for(int i=0; i< ::MaxDimension; i++)
			this->x[i]=0;
	}
	GeometryVector(double xx)
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		this->Dimension=1;
		assert(this->Dimension<=::MaxDimension);
#endif
		this->x[0]=xx;
		for(int i=1; i< ::MaxDimension; i++)
			this->x[i]=0;
	}
	GeometryVector(double xx, double yy)
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		this->Dimension=2;
		assert(this->Dimension<=::MaxDimension);
#endif
		this->x[0]=xx;
		this->x[1]=yy;
		for(int i=2; i< ::MaxDimension; i++)
			this->x[i]=0;
	}
	GeometryVector(double xx, double yy, double zz)
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		this->Dimension=3;
		assert(this->Dimension<=::MaxDimension);
#endif
		this->x[0]=xx;
		this->x[1]=yy;
		this->x[2]=zz;
		for(int i=3; i< ::MaxDimension; i++)
			this->x[i]=0;
	}
	GeometryVector(double xx, double yy, double zz, double aa)
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		this->Dimension=4;
		assert(this->Dimension<=::MaxDimension);
#endif
		this->x[0]=xx;
		this->x[1]=yy;
		this->x[2]=zz;
		this->x[3]=aa;
		for(int i=4; i< ::MaxDimension; i++)
			this->x[i]=0;
	}
	GeometryVector(const GeometryVector & src)
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		assert(src.Dimension<= ::MaxDimension);
		this->Dimension=src.Dimension;
#endif
		//std::memcpy(this->x, src.x, ::MaxDimension*sizeof(double));
		#pragma ivdep
		for(int i=0; i< ::MaxDimension; i++)
			this->x[i]=src.x[i];
	}
	void AddFrom(const GeometryVector & right)
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		assert(right.Dimension==this->Dimension);
		assert(right.Dimension<= ::MaxDimension);
#endif
		#pragma ivdep
		for(int i=0; i< ::MaxDimension; i++)
			this->x[i]+=right.x[i];
	}
	void MinusFrom(const GeometryVector & right)
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		assert(right.Dimension==this->Dimension);
		assert(right.Dimension<= ::MaxDimension);
#endif
		#pragma ivdep
		for(int i=0; i< ::MaxDimension; i++)
			this->x[i]-=right.x[i];
	}
	void MultiplyFrom(const double & right)
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		assert(this->Dimension<= ::MaxDimension);
#endif
		#pragma ivdep
		for(int i=0; i< ::MaxDimension; i++)
			this->x[i]*=right;
	}
	double Dot(const GeometryVector & right) const
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		assert(right.Dimension==this->Dimension);
		assert(right.Dimension<= ::MaxDimension);
#endif
		double result=0;
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		for(int i=0; i<this->Dimension; i++)
#else
		for(int i=0; i< ::MaxDimension; i++)
#endif
			result+=this->x[i]*right.x[i];
		return result;
	}
	double Modulus2(void) const
	{
		return this->Dot(*this);
	}
	bool IsEqual(const GeometryVector & right) const
	{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
		assert(right.Dimension==this->Dimension);
		assert(right.Dimension<= ::MaxDimension);
		for(int i=0; i<right.Dimension; i++)
			if(this->x[i]!=right.x[i])
				return false;
#else
		for(int i=0; i< ::MaxDimension; i++)
			if(this->x[i]!=right.x[i])
				return false;
#endif
		return true;
	}
	void OutputCoordinate(std::ostream & os, short dim) const
	{
		for(int i=0; i< dim; i++)
			os<<this->x[i]<<" \t";
	}
	void InputCoordinate(std::istream & is, short dim)
	{
		for(int i=0; i< dim; i++)
			is>>this->x[i];
	}
	void WriteBinary(std::ostream & ofile, short dimension) const;
	void ReadBinary(std::istream & ifile, short dimension);

	friend inline GeometryVector SameDimensionZeroVector(const GeometryVector & src);
};
//create another GeometryVector with same dimension as src, but the value is zero
inline GeometryVector SameDimensionZeroVector(const GeometryVector & src)
{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
	return GeometryVector(src.Dimension);
#else
	return GeometryVector(0);
#endif
}



inline std::ostream & operator << (std::ostream & os, const GeometryVector & a)
{
	//os<<a.Dimension<<" \t";
	a.OutputCoordinate(os, ::MaxDimension);
	os<<'\n';
	return os;
}
inline std::istream & operator >> (std::istream & is, GeometryVector & a)
{
	//is>>a.Dimension;
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
	assert(a.Dimension<= ::MaxDimension);
#endif
	a.InputCoordinate(is, ::MaxDimension);
	return is;
}
inline GeometryVector operator + (const GeometryVector & left, const GeometryVector & right)
{
	GeometryVector result(left);
	result.AddFrom(right);
	return result;
}
inline GeometryVector operator - (const GeometryVector & left, const GeometryVector & right)
{
	GeometryVector result(left);
	result.MinusFrom(right);
	return result;
}
inline GeometryVector operator * (const GeometryVector & left, const double & right)
{
	GeometryVector result(left);
	result.MultiplyFrom(right);
	return result;
}
inline GeometryVector operator * (const double & left, const GeometryVector & right)
{
	GeometryVector result(right);
	result.MultiplyFrom(left);
	return result;
}
inline bool operator == (const GeometryVector & left, const GeometryVector & right)
{
	return left.IsEqual(right);
}
inline bool operator != (const GeometryVector & left, const GeometryVector & right)
{
	return !left.IsEqual(right);
}

#endif
