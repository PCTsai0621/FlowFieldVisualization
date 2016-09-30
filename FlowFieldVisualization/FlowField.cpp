#include "FlowField.h"

double const FlowField::uu_D5[36] = { 0.0000, 0.0872, 0.1736, 0.2588, 0.3420,
0.4226, 0.5, 0.5736, 0.6428, 0.7071, 0.7660, 0.8192, 0.8660, 0.9063,
0.9397, 0.9659, 0.9848, 0.9962, 1.0000, 0.9962, 0.9848, 0.9659, 0.9397,
0.9063, 0.8660, 0.8192, 0.7660, 0.7071, 0.6428, 0.5736, 0.5, 0.4226, 0.3420,
0.2588, 0.1736, 0.0872 };
double const FlowField::vv_D5[36] = { 1.0000, 0.9962, 0.9848, 0.9659, 0.9397,
0.9063, 0.8660, 0.8192, 0.7660, 0.7071, 0.6428, 0.5736, 0.5, 0.4226,
0.3420, 0.2588, 0.1736, 0.0872, 0.0000, -0.0872, -0.1736, -0.2588,
-0.3420, -0.4226, -0.5, -0.5736, -0.6428, -0.7071, -0.7660, -0.8192
, -0.8660, -0.9063, -0.9397, -0.9659, -0.9848, -0.9962 };

double const FlowField::uu_D5_f[36] = { 1.0000, 0.9962, 0.9848, 0.9659, 0.9397,
0.9063, 0.8660, 0.8192, 0.7660, 0.7071, 0.6428, 0.5736, 0.5, 0.4226,
0.3420, 0.2588, 0.1736, 0.0872, 0.0000, -0.0872, -0.1736, -0.2588, -0.3420,
-0.4226, -0.5, -0.5736, -0.6428, -0.7071, -0.7660, -0.8192, -0.8660, -0.9063,
-0.9397, -0.9659, -0.9848, -0.9962 };
double const FlowField::vv_D5_f[36] = { 0.0000, 0.0872, 0.1736, 0.2588, 0.3420,
0.4226, 0.5, 0.5736, 0.6428, 0.7071, 0.7660, 0.8192, 0.8660, 0.9063,
0.9397, 0.9659, 0.9848, 0.9962, 1.0000, 0.9962, 0.9848, 0.9659, 0.9737,
0.9063, 0.8660, 0.8192, 0.7660, 0.7071, 0.6428, 0.5736, 0.5, 0.4226,
0.3420, 0.2588, 0.1736, 0.0872 };


double FlowField::round(double _In)
{
	return static_cast<int>(_In + 0.5);
}

double* FlowField::array2double_3D(cv::Mat &_InMat, bool &_flag)
{
	if (_InMat.empty())
	{
		std::cout << "GradientApp::array2double_3D: input mat is empty" << std::endl;
		_flag = 1;
	}

	if (_InMat.type() == 16)		//8UC3
	{
		double *_OutArray = (double *)malloc(_InMat.rows* _InMat.cols * 3 * sizeof(double));
		uchar* ptr = _InMat.ptr<uchar>(0);

		for (size_t i = 0; i < _InMat.rows; i++)
		{
			for (size_t j = 0; j < _InMat.cols; j++)
			{
				*(_OutArray + i * _InMat.cols + j) = (double)*(ptr + i * _InMat.cols * 3 + 3 * j);											//B level 0
				*(_OutArray + _InMat.cols * _InMat.rows + i * _InMat.cols + j) = (double)*(ptr + i * _InMat.cols * 3 + 3 * j + 1);			//G level 1
				*(_OutArray + 2 * _InMat.cols * _InMat.rows + i * _InMat.cols + j) = (double)*(ptr + i * _InMat.cols * 3 + 3 * j + 2);		//R level 2
			}
		}
		_flag = 0;
		return _OutArray;
	}
	else
	{
		std::cout << "GradientApp::array2double_3D: input mat is not support" << std::endl;
		_flag = 1;
		double* unknowm = (double*)malloc(sizeof(double));
		return unknowm;
	}
}

int FlowField::arrayScaling_3D(double* _InArr, const int* dims, double _scale)
{
	if ((dims[0] == 0) || (dims[1] == 0) || (dims[2] == 0))
	{
		std::cout << "GradientApp::arrayScaling_3D: error dims in 3D" << std::endl;
		return 1;
	}
	if (_scale == 0.0)
	{
		std::cout << "GradientApp::arrayScaling_3D: scale can't be zero" << std::endl;
		return 1;
	}

	for (size_t i = 0; i < dims[1]; i++)
	{
		for (size_t j = 0; j < dims[0]; j++)
		{
			*(_InArr + i*dims[0] + j) = *(_InArr + i*dims[0] + j) / _scale;													//B
			*(_InArr + dims[0] * dims[1] + i*dims[0] + j) = *(_InArr + dims[0] * dims[1] + i*dims[0] + j) / _scale;			//G
			*(_InArr + 2 * dims[0] * dims[1] + i*dims[0] + j) = *(_InArr + 2 * dims[0] * dims[1] + i*dims[0] + j) / _scale;	//R
		}
	}
	return 0;
}

int FlowField::findMax_3D(double* _InArr, const int* _dims, double &_Max, bool _sign /*= 0*/)
{
	if ((_dims[0] == 0) || (_dims[1] == 0) || (_dims[2] == 0))
	{
		std::cout << "GradientApp::findMax_3D: error dims in 3D" << std::endl;
		return 1;
	}

	double maximum = -1000000;
	if (!_sign)
	{
		for (size_t k = 0; k < _dims[2]; k++)
		{
			for (size_t i = 0; i < _dims[1]; i++)
			{
				for (size_t j = 0; j < _dims[0]; j++)
				{
					maximum = *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j) > maximum ? *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j) : maximum;
				}
			}
		}
		_Max = maximum;
		if (maximum <= 0)
		{
			return 2;
		}
	}
	else
	{
		for (size_t k = 0; k < _dims[2]; k++)
		{
			for (size_t i = 0; i < _dims[1]; i++)
			{
				for (size_t j = 0; j < _dims[0]; j++)
				{
					maximum = *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j)*(-1) > maximum ? *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j)*(-1) : maximum;
				}
			}
		}
		_Max = maximum;
		if (maximum <= 0)
		{
			return 2;
		}
	}
	return 0;
}

int FlowField::findMin_3D(double* _InArr, const int* _dims, double &_Min, bool _sign /*= 0*/)
{
	if ((_dims[0] == 0) || (_dims[1] == 0) || (_dims[2] == 0))
	{
		std::cout << "GradientApp::findMin_3D: error dims in 3D" << std::endl;
		return 1;
	}

	double minimum = 1000000;
	if (!_sign)
	{
		for (size_t k = 0; k < _dims[2]; k++)
		{
			for (size_t i = 0; i < _dims[1]; i++)
			{
				for (size_t j = 0; j < _dims[0]; j++)
				{
					if (*(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j) >= 0)
						minimum = *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j) < minimum ? *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j) : minimum;

				}
			}
		}
		_Min = minimum;
		if (minimum == 1000000)
		{
			return 2;
		}
	}
	else
	{
		for (size_t k = 0; k < _dims[2]; k++)
		{
			for (size_t i = 0; i < _dims[1]; i++)
			{
				for (size_t j = 0; j < _dims[0]; j++)
				{
					if (*(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j)*(-1) >= 0)
						minimum = *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j)*(-1) < minimum ? *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j)*(-1) : minimum;
				}
			}
		}
		_Min = minimum;
		if (minimum == 1000000)
		{
			return 2;
		}
	}
	return 0;
}

int FlowField::findRange_3D(double* _InArr, const int* _dims, double* _Range, bool _sign /*= 0*/)
{
	if ((_dims[0] == 0) || (_dims[1] == 0) || (_dims[2] == 0))
	{
		std::cout << "GradientApp::findRange_3D: error dims in 3D" << std::endl;
		return 1;
	}

	double max_s, max_us, min_s, min_us;
	int Max1 = findMax_3D(_InArr, _dims, max_us, 0);		//positive maximum
	int Max2 = findMax_3D(_InArr, _dims, max_s, 1);		//negative maximum
	int Min1 = findMin_3D(_InArr, _dims, min_us, 0);		//positive minimum(include 0)
	int Min2 = findMin_3D(_InArr, _dims, min_s, 1);		//negative minimum(include 0)

	if (!_sign)
	{
		if ((Max1 == 1) || (Max2 == 1) || (Min1 == 1) || (Min2 == 1))
		{
			std::cout << "GradientApp::findRange_3D: ERROR" << std::endl;
			return 1;
		}

		/*Maximum*/
		if ((Max1 == 2) && (Max2 == 2))
			*(_Range + 1) = 0.0;
		else if ((Max1 == 0) && (Max2 == 2))
			*(_Range + 1) = max_us;
		else if ((Max1 == 2) && (Max2 == 0))
			*(_Range + 1) = max_s;
		else if ((Max1 == 0) && (Max2 == 0))
			*(_Range + 1) = max(max_us, max_s);

		/*Minimum*/
		if ((Min1 == 2) && (Min2 == 2))
		{
			std::cout << "GradientApp::findRange_3D: Something ERROR" << std::endl;
			return 1;
		}
		else if ((Min1 == 0) && (Min2 == 2))
			*_Range = min_us;
		else if ((Min1 == 2) && (Min2 == 0))
			*_Range = min_s;
		else if ((Min1 == 0) && (Min2 == 0))
			*_Range = min(min_us, min_s);
	}
	else
	{
		if ((Max1 == 1) || (Max2 == 1) || (Min1 == 1) || (Min2 == 1))
		{
			std::cout << "GradientApp::findRange_3D: ERROR" << std::endl;
			return 1;
		}

		/*Maximum*/
		if (Max1 == 0)
			*(_Range + 1) = max_us;
		else           //no positive elements
			*(_Range + 1) = min_s;


		/*Minimum*/
		if (Max2 == 0)
			*_Range = max_s;
		else           //no negative elements
			*_Range = min_us;
	}
	return 0;
}

uchar* FlowField::loadGlyph(std::string _filename, int _width, int _height, int _depth)
{
	std::ifstream fin(_filename);

	uchar* bim = (uchar*)malloc(_height * _width * _depth * sizeof(uchar));
	uchar* bimTemp = bim;
	std::string lineTemp;
	int count = 0;
	while (std::getline(fin, lineTemp))
	{
		count++;
		if ((count % (_height + 1)) == 0)
			continue;
		std::vector<std::string> lineElems = split(lineTemp, '\t');
		if (lineElems.size() != _width)
		{
			std::cout << "GradientApp::loadGlyph: data structure of the input file is error" << std::endl;
			break;
		}
		for (size_t i = 0; i < _width; i++)
		{
			*bimTemp++ = (uchar)std::stoi(lineElems.at(i));
		}
	}
	return bim;
}

std::vector<std::string> & FlowField::split(const std::string &_s, char _delim, std::vector<std::string> &_elems)
{
	std::stringstream ss(_s);
	std::string item;
	while (std::getline(ss, item, _delim))
		_elems.push_back(item);
	return _elems;
}

std::vector<std::string> FlowField::split(const std::string &_s, char _delim)
{
	std::vector<std::string> _elems;
	split(_s, _delim, _elems);
	return _elems;
}

int FlowField::grayscaling(cv::Mat& _InMat, cv::Mat& _OutMat, double _ValueUBound, double _ValueLBound)
{
	if (_InMat.empty())
	{
		std::cout << "GradientApp::grayscaling: input is empty" << std::endl;
		system("pause");
		return 1;
	}
	if (_ValueLBound >= _ValueUBound)
	{
		std::cout << "GradientApp::grayscaling: upper bound must be larger than lower bound" << std::endl;
		system("pause");
		return 1;
	}

	_OutMat = cv::Mat(_InMat.rows, _InMat.cols, CV_8UC1);

	double* ptrSRC = _InMat.ptr<double>(0);
	uchar* ptrDST = _OutMat.ptr<uchar>(0);
	for (size_t i = 0; i < _OutMat.rows; i++)
	{
		for (size_t j = 0; j < _OutMat.cols; j++)
		{
			*(ptrDST + i*_OutMat.cols + j) = (uchar)GrayScaleCodec(*(ptrSRC + i*_InMat.cols + j), _ValueUBound, _ValueLBound);
		}
	}
	return 0;
}

int FlowField::GrayScaleCodec(double _In, double _ValueUBound, double _ValueLBound)
{
	if (_ValueUBound <= _ValueLBound)
	{
		std::cout << "GradientApp::GrayScaleCodec: upper bound must be larger than lower bound" << std::endl;
		system("pause");
		return -1;
	}
	double m = (255 - 0) / (_ValueUBound - _ValueLBound);

	return m*(_In - _ValueLBound);
}

FlowField::FlowField()
{
}


FlowField::~FlowField()
{
}

int FlowField::GradientFeature(cv::Mat& _SrcImg, mMat_F_2D& _OutFeature)
{
	if (_SrcImg.empty())
	{
		std::cout << "GradientApp::GradientFeature: input image is empty" << std::endl;
		return 1;
	}
	const int *dims = _SrcImg.size.p;		//dims[0] = rows/height, dims[1] = cols/width
	if ((dims[0] < 3) || (dims[1] < 3))
	{
		std::cout << "GradientApp::GradientFeature: input size is too small" << std::endl;
		return 1;
	}

	bool flag;
	double *SrcImg = array2double_3D(_SrcImg, flag);
	if (flag == 1)
	{
		std::cout << "GradientApp::GradientFeature: ERROR" << std::endl;
		return 1;
	}


	//for (size_t i = 0; i < _SrcImg.rows; i++)
	//{
	//	for (size_t j = 0; j < _SrcImg.cols; j++)
	//	{
	//		std::cout << *(SrcImg + i*_SrcImg.cols + j) << ", ";
	//		std::cout << *(SrcImg + _SrcImg.cols*_SrcImg.rows + i*_SrcImg.cols + j ) << ", ";
	//		std::cout << *(SrcImg + 2*_SrcImg.cols*_SrcImg.rows + i*_SrcImg.cols + j) << "  ";
	//	}
	//	std::cout << std::endl;
	//}
	//std::cout << std::endl;


	double *range = (double*)malloc(2 * sizeof(double));
	int dimTemp[3];
	dimTemp[0] = dims[1];
	dimTemp[1] = dims[0];
	dimTemp[2] = 3;

	if (findRange_3D(SrcImg, dimTemp, range, 0))
	{
		std::cout << "GradientApp::GradientFeature: ERROR" << std::endl;
		return 1;
	}

	if (arrayScaling_3D(SrcImg, dimTemp, range[1]))
	{
		std::cout << "GradientApp::GradientFeature: ERROR" << std::endl;
		return 1;
	}

	/*memory for caching orientation histograms & their norms*/
	int outsize[2];							//output size
	outsize[0] = dims[1] - 2;				//size in cols
	outsize[1] = dims[0] - 2;				//size in rows

	int *ori = (int *)malloc(outsize[0] * outsize[1] * sizeof(int));

	/*memory for HOG features*/
	_OutFeature.data = (float *)malloc(outsize[0] * outsize[1] * 2 * sizeof(float));
	_OutFeature.dims0 = outsize[0];
	_OutFeature.dims1 = outsize[1];

	for (int i = 0; i < outsize[1]; i++) {
		for (int j = 0; j < outsize[0]; j++) {

			//-----------------illustration-------------------//
			//		x x x x x x x x x
			//		x o o o o o o o x
			//		x o o o o o o o x
			//		x o o o o o o o x
			//		x x x x x x x x x
			//		o: pixel which to cal
			//		x: pixel which not to cal (margin pixel)
			//------------------------------------------------//

			/* first color channel: B*/
			double *s = SrcImg + (i + 1)*dims[1] + (j + 1);
			double dx3 = *(s + 1) - *(s - 1);					//delta cols in blue
			double dy3 = *(s + dims[1]) - *(s - dims[1]);		//delta rows in blue
			double v3 = dx3*dx3 + dy3*dy3;

			/* second color channel: G*/
			s += dims[0] * dims[1];
			double dx2 = *(s + 1) - *(s - 1);
			double dy2 = *(s + dims[1]) - *(s - dims[1]);
			double v2 = dx2*dx2 + dy2*dy2;

			/* third color channel: R*/
			s += dims[0] * dims[1];
			double dx = *(s + 1) - *(s - 1);
			double dy = *(s + dims[1]) - *(s - dims[1]);
			double v = dx*dx + dy*dy;

			/* pick channel with strongest gradient*/
			if (v2 > v) {
				v = v2;
				dx = dx2;
				dy = dy2;
			}
			if (v3 > v) {
				v = v3;
				dx = dx3;
				dy = dy3;
			}

			*(_OutFeature.data + i*outsize[0] * 2 + 2 * j) = dx;
			*(_OutFeature.data + i*outsize[0] * 2 + 2 * j + 1) = dy;

			/* snap to one of 73 orientations, use vector projection*/
			double best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < 36; o++) {
				double dot = uu_D5[o] * dx + vv_D5[o] * dy;

				if (dot > best_dot) {				//positive orientation
					best_dot = dot;
					best_o = o;
				}
				else if (-dot > best_dot) {			//negative orientation
					best_dot = -dot;
					best_o = o + 36;
				}
			}
			if (best_dot == 0)
				best_o = 72;

			*(ori + i*outsize[0] + j) = best_o;
		}
	}

	bool _show = 0;				//temp
	if (_show)
	{
		uchar* bim = loadGlyph("Glyph_D5_signed.txt", 20, 20, 73);

		cv::Mat out = cv::Mat::zeros(outsize[1] * 20, outsize[0] * 20, CV_64FC1);
		double* ptr = out.ptr<double>(0);
		for (size_t i = 0; i < outsize[1]; i++)
		{
			int iis[2] = { i * 20, (i + 1) * 20 };
			for (size_t j = 0; j < outsize[0]; j++)
			{
				int jjs[2] = { j * 20, (j + 1) * 20 };
				int oriTemp = *(ori + i*outsize[0] + j);

				for (size_t l = iis[0]; l < iis[1]; l++)
				{
					for (size_t m = jjs[0]; m < jjs[1]; m++)
					{
						*(ptr + l*outsize[0] * 20 + m) = (double)*(bim + oriTemp * 20 * 20 + (l % 20) * 20 + (m % 20));
					}
				}
			}
		}
		cv::Mat gray;
		grayscaling(out, gray, 1, 0);
		cv::imwrite("test.png", gray);
		cv::imshow("show", out);
		cv::waitKey(1);
	}

	return 0;
}

int FlowField::FlowFeature(cv::Mat& _SrcImg, mMat_F_2D& _OutFeature)
{
	if (_SrcImg.empty())
	{
		std::cout << "GradientApp::FlowFeature: input image is empty" << std::endl;
		return 1;
	}
	const int *dims = _SrcImg.size.p;		//dims[0] = rows/height, dims[1] = cols/width
	if ((dims[0] < 3) || (dims[1] < 3))
	{
		std::cout << "GradientApp::FlowFeature: input size is too small" << std::endl;
		return 1;
	}

	bool flag;
	double *SrcImg = array2double_3D(_SrcImg, flag);
	if (flag == 1)
	{
		std::cout << "GradientApp::FlowFeature: ERROR" << std::endl;
		return 1;
	}


	//for (size_t i = 0; i < _SrcImg.rows; i++)
	//{
	//	for (size_t j = 0; j < _SrcImg.cols; j++)
	//	{
	//		std::cout << *(SrcImg + i*_SrcImg.cols + j) << ", ";
	//		std::cout << *(SrcImg + _SrcImg.cols*_SrcImg.rows + i*_SrcImg.cols + j ) << ", ";
	//		std::cout << *(SrcImg + 2*_SrcImg.cols*_SrcImg.rows + i*_SrcImg.cols + j) << "  ";
	//	}
	//	std::cout << std::endl;
	//}
	//std::cout << std::endl;


	double *range = (double*)malloc(2 * sizeof(double));
	int dimTemp[3];
	dimTemp[0] = dims[1];
	dimTemp[1] = dims[0];
	dimTemp[2] = 3;

	if (findRange_3D(SrcImg, dimTemp, range, 0))
	{
		std::cout << "GradientApp::FlowFeature: ERROR" << std::endl;
		return 1;
	}

	if (arrayScaling_3D(SrcImg, dimTemp, range[1]))
	{
		std::cout << "GradientApp::FlowFeature: ERROR" << std::endl;
		return 1;
	}

	/*memory for caching orientation histograms & their norms*/
	int outsize[2];							//output size
	outsize[0] = dims[1] - 2;				//size in cols
	outsize[1] = dims[0] - 2;				//size in rows

	int *ori = (int *)malloc(outsize[0] * outsize[1] * sizeof(int));

	/*memory for HOG features*/
	_OutFeature.data = (float *)malloc(outsize[0] * outsize[1] * 2 * sizeof(float));
	_OutFeature.dims0 = outsize[0];
	_OutFeature.dims1 = outsize[1];

	for (int i = 0; i < outsize[1]; i++) {
		for (int j = 0; j < outsize[0]; j++) {

			//-----------------illustration-------------------//
			//		x x x x x x x x x
			//		x o o o o o o o x
			//		x o o o o o o o x
			//		x o o o o o o o x
			//		x x x x x x x x x
			//		o: pixel which to cal
			//		x: pixel which not to cal (margin pixel)
			//------------------------------------------------//

			/* first color channel: B*/
			double *s = SrcImg + (i + 1)*dims[1] + (j + 1);
			double dx3 = *(s + 1) - *(s - 1);					//delta cols in blue
			double dy3 = *(s + dims[1]) - *(s - dims[1]);		//delta rows in blue
			double v3 = dx3*dx3 + dy3*dy3;

			/* second color channel: G*/
			s += dims[0] * dims[1];
			double dx2 = *(s + 1) - *(s - 1);
			double dy2 = *(s + dims[1]) - *(s - dims[1]);
			double v2 = dx2*dx2 + dy2*dy2;

			/* third color channel: R*/
			s += dims[0] * dims[1];
			double dx = *(s + 1) - *(s - 1);
			double dy = *(s + dims[1]) - *(s - dims[1]);
			double v = dx*dx + dy*dy;

			/* pick channel with strongest gradient*/
			if (v2 > v) {
				v = v2;
				dx = dx2;
				dy = dy2;
			}
			if (v3 > v) {
				v = v3;
				dx = dx3;
				dy = dy3;
			}

			/* snap to one of 18 orientations, use vector projection*/
			double best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < 36; o++) {
				double dot = uu_D5[o] * dx + vv_D5[o] * dy;

				if (dot > best_dot) {				//positive orientation
					best_dot = dot;
					best_o = o;
				}
				else if (-dot > best_dot) {			//negative orientation
					best_dot = -dot;
					best_o = o + 36;
				}
			}
			if (best_dot == 0)
				best_o = 72;

			*(ori + i*outsize[0] + j) = best_o;

			int sign[2] = { 1, -1 };
			int index = best_o % 36;
			int signn = best_o / 36;

			*(_OutFeature.data + i*outsize[0] * 2 + 2 * j) = uu_D5_f[index] * sign[signn];
			*(_OutFeature.data + i*outsize[0] * 2 + 2 * j + 1) = vv_D5_f[index] * sign[signn];

			if (best_dot == 0)
			{
				*(_OutFeature.data + i*outsize[0] * 2 + 2 * j) = 0;
				*(_OutFeature.data + i*outsize[0] * 2 + 2 * j + 1) = 0;
			}
		}
	}

	bool _show = 0;				//temp
	if (_show)
	{
		uchar* bim = loadGlyph("Glyph_D5_signed.txt", 20, 20, 73);

		cv::Mat out = cv::Mat::zeros(outsize[1] * 20, outsize[0] * 20, CV_64FC1);
		double* ptr = out.ptr<double>(0);
		for (size_t i = 0; i < outsize[1]; i++)
		{
			int iis[2] = { i * 20, (i + 1) * 20 };
			for (size_t j = 0; j < outsize[0]; j++)
			{
				int jjs[2] = { j * 20, (j + 1) * 20 };
				int oriTemp = *(ori + i*outsize[0] + j);

				for (size_t l = iis[0]; l < iis[1]; l++)
				{
					for (size_t m = jjs[0]; m < jjs[1]; m++)
					{
						*(ptr + l*outsize[0] * 20 + m) = (double)*(bim + oriTemp * 20 * 20 + (l % 20) * 20 + (m % 20));
					}
				}
			}
		}
		cv::Mat gray;
		grayscaling(out, gray, 1, 0);
		cv::imwrite("test.png", gray);
		cv::imshow("show", out);

		cv::waitKey(1);
	}

	return 0;
}

int FlowField::createVectorField(int _rows, int _cols, int* _numberMap, float* _flowfield, bool _show)
{
	if ((_rows == 0) || (_cols == 0))
	{
		std::cout << "GradientApp::createFlowField: input size should not be zero" << std::endl;
		return 1;
	}


	for (size_t i = 0; i < _rows; i++)
	{
		for (size_t j = 0; j < _cols; j++)
		{
			*(_flowfield + i*_cols * 2 + 2 * j) = (float)uu_D5[*(_numberMap + i*_cols + j)];
			*(_flowfield + i*_cols * 2 + 2 * j + 1) = (float)vv_D5[*(_numberMap + i*_cols + j)];
		}
	}

	if (_show)
	{
		uchar* bim = loadGlyph("Glyph_D5_signed.txt", 20, 20, 73);

		cv::Mat out = cv::Mat::zeros(_rows * 20, _cols * 20, CV_64FC1);
		double* ptr = out.ptr<double>(0);
		for (size_t i = 0; i < _rows; i++)
		{
			int iis[2] = { i * 20, (i + 1) * 20 };
			for (size_t j = 0; j < _cols; j++)
			{
				int jjs[2] = { j * 20, (j + 1) * 20 };
				int oriTemp = *(_numberMap + i*_cols + j);

				for (size_t l = iis[0]; l < iis[1]; l++)
				{
					for (size_t m = jjs[0]; m < jjs[1]; m++)
					{
						*(ptr + l*_cols * 20 + m) = (double)*(bim + oriTemp * 20 * 20 + (l % 20) * 20 + (m % 20));
					}
				}
			}
		}
		cv::imshow("show", out);
		cv::waitKey(1);
	}

	return 0;
}

int FlowField::FlowFieldGenerator(int* _Xrange, int* _Yrange, int _Strength, int* _Offset, std::string _Kernel, mMat_F_2D& _OutField)
{
	float decay = 0.0000005;
	float rows1[2], rows2[2];

	if (_Kernel == "CCW")
	{
		rows1[0] = 0;
		rows1[1] = -1 * _Strength;
		rows2[0] = 1 * _Strength;
		rows2[1] = 0;
	}
	else if (_Kernel == "CW")
	{
		rows1[0] = 0;
		rows1[1] = 1 * _Strength;
		rows2[0] = -1 * _Strength;
		rows2[1] = 0;
	}
	else if (_Kernel == "saddle")
	{
		rows1[0] = -1 * _Strength;
		rows1[1] = 0;
		rows2[0] = 0;
		rows2[1] = 1 * _Strength;
	}
	else if (_Kernel == "sink")
	{
		rows1[0] = -1 * _Strength;
		rows1[1] = 0;
		rows2[0] = 0;
		rows2[1] = -1 * _Strength;
	}
	else if (_Kernel == "source")
	{
		rows1[0] = 1 * _Strength;
		rows1[1] = 0;
		rows2[0] = 0;
		rows2[1] = 1 * _Strength;
	}
	else
	{
		std::cout << "GradientApp::FlowFieldGenerator: no such type or the wrong case of the word" << std::endl;
		return 1;
	}

	float dist, mm_x, mm_y;
	_OutField.data = (float*)malloc((_Xrange[1] - _Xrange[0] + 1) * (_Yrange[1] - _Yrange[0] + 1) * 2 * sizeof(float));
	_OutField.dims0 = (_Xrange[1] - _Xrange[0] + 1);
	_OutField.dims1 = (_Yrange[1] - _Yrange[0] + 1);

	for (int i = _Yrange[0]; i < (_Yrange[1] + 1); i++)
	{
		for (int j = _Xrange[0]; j < (_Xrange[1] + 1); j++)
		{
			dist = (j - _Offset[0])*(j - _Offset[0]) + (i - _Offset[1])*(i - _Offset[1]);
			mm_x = rows1[0] * (j - _Offset[0]) + rows1[1] * (i - _Offset[1]);
			mm_y = rows2[0] * (j - _Offset[0]) + rows2[1] * (i - _Offset[1]);
			*(_OutField.data + i*(_Xrange[1] - _Xrange[0] + 1) * 2 + 2 * j) = expf((-1)*decay*dist)*mm_x;
			*(_OutField.data + i*(_Xrange[1] - _Xrange[0] + 1) * 2 + 2 * j + 1) = expf((-1)*decay * dist)*mm_y;
		}
	}

	return 0;
}

int FlowField::showArrowMap(int _rows, int _cols, float* _VectorField, cv::Mat& _ArrowMap, bool _show /*= 1*/, int _CoorSys /*= 0*/, std::string _saveDir /*= ""*/)
{
	if ((_rows <= 0) || (_cols <= 0))
	{
		std::cout << "GradientApp::showArrowMap: input size is error" << std::endl;
		return 1;
	}

	int* orimap = (int*)malloc(_rows*_cols*sizeof(int));

	for (size_t i = 0; i < _rows; i++)
	{
		for (size_t j = 0; j < _cols; j++)
		{
			float x = *(_VectorField + i*_cols * 2 + 2 * j);
			float y;
			switch (_CoorSys)
			{
			case 0:			//for xy coordinate system
				y = *(_VectorField + i*_cols * 2 + 2 * j + 1);
				break;
			case 1:
				y = *(_VectorField + i*_cols * 2 + 2 * j + 1) * (-1);
				break;
			default:
				std::cout << "GradientApp::showArrowMap: no such coordinate system" << std::endl;
				break;
			}


			double best_dot = 0;
			int best_o = 0;
			for (size_t k = 0; k < 36; k++)
			{
				double dot = uu_D5[k] * x + vv_D5[k] * y;
				if (dot > best_dot)
				{
					best_dot = dot;
					best_o = k;
				}
				else if (-dot > best_dot)
				{
					best_dot = -dot;
					best_o = k + 36;
				}
			}
			if (best_dot == 0)
				best_o = 72;				//no orientation
			*(orimap + i*_cols + j) = best_o;
		}
	}

	uchar* bim = loadGlyph("Glyph_D5_signed.txt", 20, 20, 73);
	cv::Mat out = cv::Mat::zeros(_rows * 20, _cols * 20, CV_64FC1);
	double* ptr = out.ptr<double>(0);
	for (size_t i = 0; i < _rows; i++)
	{
		int iis[2] = { i * 20, (i + 1) * 20 };
		for (size_t j = 0; j < _cols; j++)
		{
			int jjs[2] = { j * 20, (j + 1) * 20 };
			int oriTemp = *(orimap + i*_cols + j);

			for (size_t l = iis[0]; l < iis[1]; l++)
			{
				for (size_t m = jjs[0]; m < jjs[1]; m++)
				{
					*(ptr + l*_cols * 20 + m) = (double)*(bim + oriTemp * 20 * 20 + (l % 20) * 20 + (m % 20));
				}
			}
		}
	}

	if (grayscaling(out, _ArrowMap, 1, 0))
	{
		std::cout << "GradientApp::ShowHOG: ERROR" << std::endl;
		system("pause");
		return 1;
	}
	if (_saveDir != "")
	{
		cv::imwrite(_saveDir, _ArrowMap);
	}
	if (_show)
	{
		cv::imshow("show", out);
		cv::waitKey(1);
	}

	return 0;
}

int FlowField::showArrowMap(int _rows, int _cols, float* _VectorField, bool _show /*= 1*/, int _CoorSys /*= 0*/, std::string _saveDir /*= ""*/)
{
	cv::Mat temp;
	return showArrowMap(_rows, _cols, _VectorField, temp, _show, _CoorSys, _saveDir);
}

int FlowField::CoorTrans_flowfield(mMat_F_2D& _InField, mMat_F_2D& _OutField)
{
	if ((_InField.dims0 == 0) || (_InField.dims1 == 0))
	{
		std::cout << "FlowField::CoorTrans_flowfield: empty input field" << std::endl;
		return 1;
	}

	_OutField.dims0 = _InField.dims0;
	_OutField.dims1 = _InField.dims1;

	_OutField.data = (float*)malloc(_InField.dims0 * _InField.dims1 * 2 * sizeof(float));		//this is for xy-coor to show

	for (size_t i = 0; i < _InField.dims1; i++)
	{
		for (size_t j = 0; j < _InField.dims0; j++)
		{
			*(_OutField.data + i*_InField.dims0 * 2 + 2 * j) = *(_InField.data + (_InField.dims1 - i - 1)*_InField.dims0 * 2 + 2 * j);
			*(_OutField.data + i*_InField.dims0 * 2 + 2 * j + 1) = *(_InField.data + (_InField.dims1 - i - 1)*_InField.dims0 * 2 + 2 * j + 1);  //stored order upside down
		}
	}

	return 0;
}
