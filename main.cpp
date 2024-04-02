#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace cv;
using namespace std;
const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;
static void on_low_H_thresh_trackbar(int, void*)
{
	low_H = min(high_H - 1, low_H);
	setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void*)
{
	high_H = max(high_H, low_H + 1);
	setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
	low_S = min(high_S - 1, low_S);
	setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void*)
{
	high_S = max(high_S, low_S + 1);
	setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void*)
{
	low_V = min(high_V - 1, low_V);
	setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void*)
{
	high_V = max(high_V, low_V + 1);
	setTrackbarPos("High V", window_detection_name, high_V);
}
int main()
{
	VideoCapture cap("test_video1.mp4");
	//VideoCapture cap("test_video2.mp4");
	if (!cap.isOpened()) {
		cout << "Error opening video" << endl;
		return -1;
	}

	int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

	VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));

	namedWindow(window_capture_name);
	namedWindow(window_detection_name);
	createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
	createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
	createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
	createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
	createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
	createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
	Mat frame, frame_HSV, frame_threshold;
	while (true) {
		cap >> frame;
		if (frame.empty())
		{
			break;
		}
		
		cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
		//inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
		inRange(frame_HSV, Scalar(95, 12, 138), Scalar(255, 50, 255), frame_threshold);

		cv::Mat image1;
		GaussianBlur(frame_threshold, image1, Size(3, 3), 0);

		cv::Mat edges1;
		Canny(image1, edges1, 50, 100);

		std::vector<std::vector<Point>> contours;
		std::vector<Vec4i> hierarchy;

		findContours(edges1, contours, hierarchy, RETR_EXTERNAL,
			CHAIN_APPROX_SIMPLE);
		vector<Moments> mu(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			mu[i] = moments(contours[i], false);
		}
		vector<Point2f> mc(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}
		vector<vector<Point>> contourPoly(contours.size());
		for (size_t i = 0; i < contours.size(); i++)
		{
			int c_area = contourArea(contours[i]);
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], contourPoly[i], 0.02 * peri, true);
			drawContours(frame, contours, (int)i, Scalar(0, 255, 0), 2, LINE_8, hierarchy, 0);

			int obj_corners = (int)contourPoly[i].size();
			string obj_name;

			if (obj_corners == 4)
				obj_name = "Rectangle";

			putText(frame, obj_name,
				mc[i],
				FONT_HERSHEY_PLAIN,
				1,
				Scalar(255, 255, 255),
				2);
		}

		video.write(edges1);

		imshow(window_capture_name, frame);
		imshow(window_detection_name, frame_threshold);
		char key = (char)waitKey(30);
		if (key == 'q' || key == 27)
		{
			break;
		}
	}
	cap.release();
	video.release();

	cv::destroyAllWindows();

	return 0;
}