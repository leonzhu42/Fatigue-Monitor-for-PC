#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "asmfitting.h"
#include "asmlibrary.h"
#include "vjfacedetect.h"

#include <vector>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>
#include <strstream>
#include <time.h>

using namespace cv;
using namespace std;

const int n_iteration = 20;
const int asm_points = 68;
const size_t max_size = 100000;
const int value_line_num = 143;
const char *model_name = "my68-1d.amf";
const char *face_cascade_name = "haarcascade_frontalface_alt2.xml";
const char *asm_points_name = "asm_points.out";
const char *winname = "Fatigue Monitor";
CascadeClassifier face_cascade;
asmfitting fit_asm;
asm_shape shape, detshape;
vector<double> fatigue_values;
int stabilizer = 3, prev_stabilizer = stabilizer;
int interval = 20, prev_interval = interval;
int fatigue_threshold = 7;
Mat frame;
RNG rng(12345);

inline int max_num(int a, int b)
{
	return a > b ? a : b;
}

void onIntervalChanged(int, void*)
{
	interval = getTrackbarPos("Interval", winname);
	if(interval == 0)
	{
		interval = prev_interval;
		setTrackbarPos("Interval", winname, interval);
	}
	else
		prev_interval = interval;
}

void onStabilizerChanged(int, void*)
{
	stabilizer = getTrackbarPos("Stabilizer", winname);
	if(stabilizer == 0)
	{
		stabilizer = prev_stabilizer;
		setTrackbarPos("Stabilizer", winname, stabilizer);
	}
	else
		prev_stabilizer = stabilizer;
}

void onFatigueThresholdChanged(int, void*)
{
	fatigue_threshold = getTrackbarPos("Fatigue Threshold", winname);
}

void showValue(double value)
{
	ostringstream strs;
	strs << value;
	string str = strs.str();
	displayOverlay(winname, str.c_str(), 0);
	if(value >= fatigue_threshold)
		system("canberra-gtk-play -f rest.ogg");
}

inline double vector_mean(size_t begin, size_t end)
{
	double r = 0;
	for(size_t i = begin; i < end; ++i)
		r += fatigue_values[i];
	r /= (end - begin);
	return r;
}

double evaluateCurrentFatigueValue()
{
	system("java -classpath weka.jar weka.filters.supervised.attribute.AddClassification -serialized model_filtered.model -classification -remove-old-class -i value.arff -o value -c first");
	ifstream value("value");
	string s;
	for(int i = 0; i < value_line_num; ++i)
		getline(value, s);
	size_t c;
	for(c = s.length(); c > 0; --c)
		if(s[c - 1] == ',')
			break;

	s.erase(s.begin(), s.begin() + c);
	double v = atof(s.c_str());
	return v;
}

void DrawResult(IplImage* image, const asm_shape& shape)
{
	for(int i = 0; i < shape.NPoints(); ++i)
		cvCircle(image, cvPoint(shape[i].x, shape[i].y), 2, CV_RGB(255, 0, 0));
}

void ASM_genARFF_showFatigueValue()
{
	//ASM
	IplImage *image = new IplImage(frame);
	bool flag = detect_one_face(detshape, image);
	if(flag)
		InitShapeFromDetBox(shape, detshape, fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
	else
	{
		cout << "This frame doesn't contain any faces!" << endl;
		return;
	}
	fit_asm.Fitting(shape, image, n_iteration);
	DrawResult(image, shape);
	frame = Mat(image);

	//modify the image to fit the model
	for(int i = 0; i < shape.NPoints(); ++i)
	{
		shape[i].x -= shape[shape.NPoints() - 1].x;
		shape[i].y -= shape[shape.NPoints() - 1].y;
	}

	for(int i = 0; i < shape.NPoints(); ++i)
	{
		shape[i].x = shape[i].x * 200 / shape.GetWidth();
		shape[i].y = shape[i].y * 200 / shape.GetHeight();
	}


	//genARFF
	ofstream arff("value.arff");
	arff << "@relation fatigue" << endl;
	arff << endl;
	arff << "@attribute fatigue_value real" << endl;
	for(int i = 0; i < asm_points; ++i)
	{
		arff << "@attribute a" << i << "_x real" << endl;
		arff << "@attribute a" << i << "_y real" << endl;
	}
	arff << endl;
	arff << "@data" << endl;
	arff << "?";
	for(int i = 0; i < shape.NPoints(); ++i)
		arff << "," << shape[i].x << "," << shape[i].y;
	arff << endl;
	arff.close();


	//calculate and show fatigue value
	double current_fatigue_value = evaluateCurrentFatigueValue();
	fatigue_values.push_back(current_fatigue_value);
	size_t end = fatigue_values.size();
	size_t begin = end - stabilizer;
	if(begin < 0)
		begin = 0;
	double stabilized_fatigue_value = vector_mean(begin, end);
	showValue(stabilized_fatigue_value);

	if(fatigue_values.size() > max_size)
		fatigue_values.clear();
}

int main(int argc, char *argv[])
{
	//init
	if(fit_asm.Read(model_name) == false)
	{
		cerr << "Error loading model!" << endl;
		return -1;
	}
	if(init_detect_cascade(face_cascade_name) == false)
	{
		cerr << "Error initializing cascade!" << endl;
		return -1;
	}

	CvCapture* capture = cvCaptureFromCAM(-1);
	if(!capture)
	{
		cerr << "Error capturing from Webcam!" << endl;
		return -1;
	}

	//run
	namedWindow(winname);
	createTrackbar("Interval", winname, &interval, 5000, onIntervalChanged);
	createTrackbar("Stabilizer", winname, &stabilizer, 10, onStabilizerChanged);
	createTrackbar("Fatigue Threshold", winname, &fatigue_threshold, 10, onFatigueThresholdChanged);
	while(true)
	{
		frame = cvQueryFrame(capture);
		if(frame.empty())
		{
			cout << "No captured frame!" << endl;
			continue;
		}
		ASM_genARFF_showFatigueValue();
		if(frame.data)
			imshow(winname, frame);
		char c = waitKey(interval);
		if(c == 'c')
			break;
	}
	remove("value");
	remove("value.arff");
	return 0;
}

