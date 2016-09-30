#include "FlowField.h"

int main()
{
	//---------------Here show how to create a simple flow field and visualize it--------------------
	FlowField flowfield;
	int xr[2] = { 0, 15 };
	int yr[2] = { 0, 20 }; 
	int offset[2] = { 3, 8 };
	FlowField::mMat_F_2D field, field_tran;

	int res_gen = flowfield.FlowFieldGenerator(xr, yr, 1, offset, "CW", field);
	int res_tran = flowfield.CoorTrans_flowfield(field, field_tran);
	int res_show = flowfield.showArrowMap(field.dims1, field.dims0, field_tran.data);
	cv::waitKey(0);

	int res_a = res_gen || res_tran || res_show;

	//----------------Here show how to extract a flow feature and visualize it ----------------------
	cv::Mat src = cv::imread("test.jpg");

	FlowField::mMat_F_2D feat;
	int res_flow = flowfield.FlowFeature(src, feat);		//you can try GradientFeature, it would be more intuitive
	int res_show_b = flowfield.showArrowMap(feat.dims1, feat.dims0, feat.data, 1, 1);			//use uv-coord because of cv::Mat
	cv::waitKey(0);

	int res_b = res_flow || res_show_b;

	return res_a || res_b;
}