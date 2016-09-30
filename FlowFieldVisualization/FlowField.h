#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <iostream>
#include <fstream>
#include <time.h>

class FlowField
{
public:
	/*struct of 2D double matrix*/
	struct mMat_F_2D
	{
		int dims0;	//width
		int dims1;	//height
		float *data;
		mMat_F_2D(){};
		~mMat_F_2D(){ free(data); }
		mMat_F_2D(int _dims0, int _dims1, float* _data)
		{
			dims0 = _dims0;
			dims1 = _dims1;
			data = (float*)malloc(dims0*dims1*2*sizeof(float));
			float* ptr_temp = data;
			for (size_t i = 0; i < dims1; i++)
				for (size_t j = 0; j < dims0; j++)
					*ptr_temp++ = *_data++;

		};
	};

private:
	static const double uu_D5[36];				//gradient projection unit vector (x) with the variation of Degree 5, range from +90 to -85
	static const double vv_D5[36];				//gradient projection unit vector (y) with the variation of Degree 5, range from +90 to -85
	static const double uu_D5_f[36];			//flow projection unit vector (x) with the variation of Degree 5, perpendicular to gradient projection unit vector (x)
	static const double vv_D5_f[36];			//flow projection unit vector (y) with the variation of Degree 5, perpendicular to gradient projection unit vector (y)

private:			//***Comparison***//

	/*minimal comparison*/
	inline double min(double _In1, double _In2) { return (_In1 <= _In2 ? _In1 : _In2); };
	/*minimal comparison*/
	inline int min(int _In1, int _In2) { return (_In1 <= _In2 ? _In1 : _In2); };
	/*maximal comparison*/
	inline double max(double _In1, double _In2) { return (_In1 <= _In2 ? _In2 : _In1); };
	/*maximal comparison*/
	inline int max(int _In1, int _In2) { return (_In1 <= _In2 ? _In2 : _In1); };

	/*Implement of round*/
	inline double round(double _In);

					//***Array Calculation***//
	/*
	turn the value of the input array into double type
	@_InMat: input mat(for opencv)
	@_flag: debug flag, return 1 when error
	*/
	inline double* array2double_3D(cv::Mat &_InMat, bool &_flag);		//if flag = 1 when error

	/*
	scaling the array, return 1 when error, otherwise 0
	@_InArr: input 3D array
	@_dims: the dimensions of the input array, should be width*height*depth or cols*rows*channels
	@_scale: scale factor
	*/
	inline int arrayScaling_3D(double* _InArr, const int* dims, double _scale);

	/*
	find the maximum element in 3D array, return 2 when can't find the maximum (maybe zero or not exist),
	return 1 when error, otherwise 0
	@_InArr: input 3D array
	@_dims: the dimensions of the input array, should be width*height*depth or cols*rows*channels
	@_Max: the output maximum
	@_sign: positive value or negative value. 0: will find the maximum of all the positive elements;
	1: will find the absolute maximum of all the negative elements
	*/
	inline int findMax_3D(double* _InArr, const int* _dims, double &_Max, bool _sign = 0);

	/*
	find the minimum element in 3D array, return 2 when can't find the minimum
	return 1 when error, otherwise 0
	@_InArr: input 3D array
	@_dims: the dimensions of the input array, should be width*height*depth or cols*rows*channels
	@_Max: the output minimum
	@_sign: positive value or negative value. 0: will find the minimum of all the positive elements;
	1: will find the absolute minimum of all the negative elements **(including zero)
	*/
	inline int findMin_3D(double* _InArr, const int* _dims, double &_Min, bool _sign = 0);

	/*
	find the value range of the input 3D array
	@_InArr: input 3D array
	@_dims: the dimensions of the input array, should be width*height*depth or cols*rows*channels
	@_Range: the output range, will be (min,max)
	@_sign: whether signed value or not. 0: will find the value regardless of sign, that is, find the
	absolute value range; 1: will find the value considering the sign
	*/
	inline int findRange_3D(double* _InArr, const int* _dims, double* _Range, bool _sign = 0);

	/*load the glyph model for default*/
	uchar* loadGlyph(std::string _filename, int _width, int _height, int _depth);

	/*load file*/
	std::vector<std::string> &split(const std::string &_s, char _delim, std::vector<std::string> &_elems);
	std::vector<std::string> split(const std::string &_s, char _delim);

	/*
	turn the value range to gray scale map
	http://www.programming-techniques.com/2013/01/contrast-stretching-using-c-and-opencv.html
	*/
	int grayscaling(cv::Mat& _InMat, cv::Mat& _OutMat, double _ValueUBound, double _ValueLBound);

	int GrayScaleCodec(double _In, double _ValueUBound, double _ValueLBound);


public:
	FlowField();
	~FlowField();

	/*
	Extract the gradient feature, return 1 when error, otherwise 0.
	**Data structure: xyxy...xy 
	@_SrcImg: the input image
	@_OutFeature: the output feature, should be GradientApp::mMat_D_3D
	*/
	int GradientFeature(cv::Mat& _SrcImg, mMat_F_2D& _OutFeature);

	/*
	Extract the flow feature which is perpendicular to gradient feature, return 1 when error, otherwise 0.
	**Data structure: xyxy...xy
	@_SrcImg: the input image
	@_OutFeature: the output feature, should be GradientApp::mMat_D_3D
	*/
	int FlowFeature(cv::Mat& _SrcImg, mMat_F_2D& _OutFeature);

	/*
	Create your own vector field manually
	@_rows: the rows of the field
	@_cols: the cols of the field
	@_numberMap: input orient index map
	@_flowfield: output the orient unit vector
	@_show: whether to show with arrow map
	**vector is reference to gradient projection unit vector
	*/
	int createVectorField(int _rows, int _cols, int* _numberMap, float* _flowfield, bool _show);

	/*
	Create a simple flow field by singularity
	@_Xrange: the range of x-coordination of the flow field. **should be 2-elements array,[lower_boundary, upper_boundary]
	@_Yrange: the range of y-coordination of the flow field. **should be 2-elements array,[lower_boundary, upper_boundary]
	@_Strength: the strength of the Jacobian matrix JV.
	@_Offset: the position of the sigularity. **should be 2-elements array,[x, y]
	@_Kernel: type of sigularity. **should be one of "sink", "saddle", "source", "CCW", "CW"
	@_OutField: the output flow field. Data structure: xyxy...xy
	reference: http://www.impa.br/opencms/pt/ensino/downloads/dissertacoes_de_mestrado/dissertacoes_2009/Ricardo_david_castaneda_marin.pdf
	*/
	int FlowFieldGenerator(int* _Xrange, int* _Yrange, int _Strength, int* _Offset, std::string _Kernel, mMat_F_2D& _OutField);

	/*
	Show the vector field with arrow map. 
	@_rows: the rows of the vector field
	@_cols: the cols of the vector field
	@_VectorField: the input vector field. **Data structure should be: xyxy...xy
	@_ArrowMap: the output arrow map image.
	@_show: whether to show the arrow map image or not.
	@_CoordSys: the coordinate system to reference. 0 for xy coordination, 1 for uv coordination.
	@_saveDir: the directory of the arrow map image to save. If you want save the result, please
	enter the whole directory name, like: "DIR/FILENAME.DATATYPE". The default is not to save.
	**we won't create the directory which you want to save if it not exist
	**if you want to show arrow map with FlowFieldGenerator(), you need to tansform by function CoorTrans_flowfield()
	*/
	int showArrowMap(int _rows, int _cols, float* _VectorField, cv::Mat& _ArrowMap, bool _show = 1, int _CoorSys = 0, std::string _saveDir = "");

	/*
	Show the vector field with arrow map.
	@_rows: the rows of the vector field
	@_cols: the cols of the vector field
	@_VectorField: the input vector field. **Data structure should be: xyxy...xy
	@_show: whether to show the arrow map image or not.
	@_CoordSys: the coordinate system to reference. 0 for xy coordination, 1 for uv coordination.
	@_saveDir: the directory of the arrow map image to save. If you want save the result, please
	enter the whole directory name, like: "DIR/FILENAME.DATATYPE". The default is not to save.
	**we won't create the directory which you want to save if it not exist
	**if you want to show arrow map with FlowFieldGenerator(), you need to tansform by function CoorTrans_flowfield()
	*/
	int showArrowMap(int _rows, int _cols, float* _VectorField, bool _show = 1, int _CoorSys = 0, std::string _saveDir = "");

	/*
	Because of data storage order, we need to restore the order for visualization
	*/
	int FlowField::CoorTrans_flowfield(mMat_F_2D& _InField, mMat_F_2D& _OutField);
};

