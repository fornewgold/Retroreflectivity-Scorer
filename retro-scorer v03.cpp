#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;

const int step_sec = 10; // Seconds skipped when pressing arrows
/*
TM_SQDIFF
TM_SQDIFF_NORMED
TM_CCORR
TM_CCORR_NORMED
TM_CCOEFF
TM_CCOEFF_NORMED
*/
const int MATCH_METHOD = TM_CCORR_NORMED;

VideoCapture cap, cap_cache;
bool dragging = false;
bool selected = false;
float display_scale = 0.5;
Point leftdown(-1, -1), leftup(-1, -1);


// Print version and help information
static void help()
{
	cout << "Traffic Sign Retroreflectivity Scorer [Verison 0.3 Beta]" << endl;
	cout << "Copyright <c> 2017 Xi Zhao at Clemson University. All rights reserved." << endl << endl;

	cout << "Hot keys:" << endl;
	cout << "\tESC                     - quit the program" << endl;
	cout << "\tP                       - pause the player" << endl;
	cout << "\tleft/right arrow        - move backwad/forwad" << endl;
	cout << "\tdrag a box when pausing - select region of interest" << endl << endl;
}


// Show timer
void addTime(Mat* img)
{
	double front_scale = 0.5;
	int thickness = 1;
	int baseline = 0;

	int timeInSec = (int)(cap.get(CAP_PROP_POS_FRAMES) / cap.get(CAP_PROP_FPS));
	int sec = timeInSec % 60;
	int min = timeInSec / 60;

	string timeString = format("%d:%d", min, sec);
	Size text_size = getTextSize(timeString, FONT_HERSHEY_SIMPLEX, front_scale, thickness, &baseline);
	putText(*img, timeString, Size(0, text_size.height), FONT_HERSHEY_SIMPLEX, front_scale, Scalar(255, 255, 255), thickness, LINE_8, false);
}


// Get mask using Otsu's method
static void getMask(Mat* img, Mat* th)
{
	Mat blur;
	GaussianBlur(*img, blur, Size(25, 75), 0);
	threshold(blur, *th, 0, 255, THRESH_BINARY | THRESH_OTSU);
}


// Add edge of target into crop
static void addEdge(Mat* img, Mat* edge)
{
	for (int j = 0; j < img->rows; ++j)
	for (int i = 0; i < img->cols; ++i)
	if (edge->at<uchar>(j, i) != 0)
		img->at<Vec3b>(j, i) = Vec3b(0, 255, 255);
}


// Compute average intensity of target
static int computeROIIntensity(Mat* img, bool show)
{
	Mat grayscale, mask, edge, marked = img->clone();

	cvtColor(*img, grayscale, COLOR_RGB2GRAY);
	//imshow("Grayscale", grayscale);
	getMask(&grayscale, &mask);
	//imshow("Mask", mask);
	Canny(mask, edge, 0.25, 0.75);
	//imshow("Edge", edge);
	addEdge(&marked, &edge);
	if (show)
		imshow("Target", marked);
	return (int)mean(grayscale, mask)[0];
}


// Compute score for each following frame
int computeNextScores(int i, vector<Mat>* temp)
{
	// Set cap_cache at the sample frame as cap and read next frame
	cap_cache.set(CAP_PROP_POS_FRAMES, cap.get(CAP_PROP_POS_FRAMES) + i);

	Point matchLoc;
	Mat img;
	if (cap_cache.read(img))
	{
		//cout << "Current frame number: " << cap.get(CAP_PROP_POS_FRAMES) << ", " << cap_cache.get(CAP_PROP_POS_FRAMES) << endl;
		
		int result_cols = img.cols - (*temp)[i].cols + 1;
		int result_rows = img.rows - (*temp)[i].rows + 1;
		Mat result;
		result.create(result_cols, result_rows, CV_32FC1);
		matchTemplate(img, (*temp)[i], result, MATCH_METHOD);
		normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		if (MATCH_METHOD == TM_SQDIFF || MATCH_METHOD == TM_SQDIFF_NORMED)
			matchLoc = minLoc;
		else
			matchLoc = maxLoc;

		Mat img_display = img.clone();
		rectangle(img_display, matchLoc, Point(matchLoc.x + (*temp)[i].cols, matchLoc.y + (*temp)[i].rows), Scalar(0, 255, 0));

		/*
		if (i == 29)
		{
			string matchTitle = to_string(i + 1);
			imshow("Match" + matchTitle, img_display);
		}
		*/
	}
	else
		cout << "Fail to grab the next frame" << endl;

	Mat newROI = Mat(img, Rect(matchLoc, Point(matchLoc.x + (*temp)[i].cols, matchLoc.y + (*temp)[i].rows)));
	int score = computeROIIntensity(&newROI, false);
	(*temp).push_back(newROI);
	
	return score;
}


// Keep the point within the image
static Point restrainCoordinates(Point* p, const Mat* img)
{
	p->x = max(0, p->x);
	p->y = max(0, p->y);
	p->x = min(img->size().width - 1, p->x);
	p->y = min(img->size().height - 1, p->y);
	return *p;
}


// Estimate overall retro-score
static void getRetroScore(Mat* crop)
{
	int current = computeROIIntensity(crop, true);

	int maxAfterScore = 0;
	vector<int> after;
	vector<Mat> *afterTemplates = new vector<Mat>();
	(*afterTemplates).push_back(*crop);
	for (int i = 0; i < 30; ++i)
	{
		/*
		if (i == 29)
		{
		string templateTitle = to_string(i + 1);
		imshow("Template " + templateTitle, (*afterTemplates)[i]);
		}
		*/
		
		int aScore = computeNextScores(i, afterTemplates);
		//cout << "aScore = " << aScore << endl;
		after.push_back(aScore);

		if (maxAfterScore < aScore)
			maxAfterScore = aScore;

	}
	
	int maxScore = max(current, maxAfterScore);
	cout << "Instant Retro-score: " << current << endl;
	cout << "Max Retro-score: " << maxScore << endl << endl;
}


// Crop ROI
static void getROI()
{
	selected = false;

	// Set cap_cache at the same frame as cap
	cap_cache.set(CAP_PROP_POS_FRAMES, cap.get(CAP_PROP_POS_FRAMES) - 1);

	// Get ROI with original size in cache and its retro-score
	Mat img;
	if (cap_cache.read(img))
	{
		cout << "Current frame number: " << cap.get(CAP_PROP_POS_FRAMES) << ", " << cap_cache.get(CAP_PROP_POS_FRAMES) << endl;
		Mat roi = Mat(img, Rect(leftdown * (1 / display_scale), leftup * (1 / display_scale)));
		getRetroScore(&roi);
	}
	else
		cout << "Fail to grab the current frame" << endl;
}


// Mouse events
static void mouseEvent(const int event, const int x, const int y, const int flags, void* data)
{
	Mat* f = (Mat*)data;

	// Initialize selection when left button down
	if (event == EVENT_LBUTTONDOWN)
	{
		dragging = true;
		selected = false;
		leftdown = Point(x, y);
		leftup = Point(x, y);
	}
	// Draw ROI when dragging
	if (event == EVENT_MOUSEMOVE)
	{
		if (dragging == true)
		{
			leftup = Point(x, y);
			rectangle(*f, leftdown, leftup, Scalar(0, 255, 0));
		}
	}
	// Get ROI when left button up
	if (event == EVENT_LBUTTONUP)
	{
		dragging = false;
		selected = true;
		leftup = Point(x, y);
		leftdown = restrainCoordinates(&leftdown, f);
		leftup = restrainCoordinates(&leftup, f);
		if ((leftdown != leftup))
			cout << "ROI: " << Rect(leftdown, leftup) << endl;
	}
}


// Trackbar operations
static void onTrackbar(const int slider, void*)
{
	int f = (int)(slider * cap.get(CAP_PROP_FRAME_COUNT) / 100);
	cap.set(CAP_PROP_POS_FRAMES, f);
}


int main(int argc, const char *argv[])
{
	help();

	// Parse command-line arguments
	if (argc != 2)
	{
		cerr << "Usage: exe_file input_file" << endl;
		return 0;
	}

	// Get the input file name
	string filename;
	filename = argv[1];
	cout << "Input file name: " << filename << endl;

	// Open the input file
	cap.open(filename);
	cap_cache.open(filename);

	// Check if the input file opened successfully
	if (!cap.isOpened() || !cap_cache.isOpened())
	{
		cerr << "Fail to opening the file" << endl;
		return -1;
	}
	else
		cout << "File opened" << endl;

	// Get and print basic information of the video file
	int len_sec = (int)(cap.get(CAP_PROP_FRAME_COUNT) / cap.get(CAP_PROP_FPS));
	cout << "Video length : " << len_sec / 3600 << ":" << (len_sec / 60) % 60 << ":" << len_sec % 60 << endl;
	cout << "Original resolution: " << cap.get(CAP_PROP_FRAME_WIDTH) << " x " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
	cout << "Frame rate: " << cap.get(CAP_PROP_FPS) << endl;
	cout << endl;

	// Initialize variables
	bool paused = false;
	Mat frame, frame_display;
	namedWindow("Video", 1);

	// Create a trackbar
	int slider = 0;
	const int slider_max = 100;
	String trackbarname = "Played(%)";

	// Looply process frames from the video file
	while (1)
	{
		// End loop when reach the end of the video file
		if (cap.get(CAP_PROP_POS_FRAMES) >= (cap.get(CAP_PROP_FRAME_COUNT) - 1))
		{
			cout << "Reached the end of video file" << endl;
			waitKey();
			break;
		}

		// Get a new frame when playing
		if (!paused)
		{
			cap >> frame;
			// Initialize ROI
			leftdown = Point(-1, -1);
			leftup = Point(-1, -1);

			// Check if the new frame is empty
			if (frame.empty())
			{
				cout << "Reached an empty frame at frame " << cap.get(CAP_PROP_POS_FRAMES) << "/" << cap.get(CAP_PROP_FRAME_COUNT) << endl << endl;
				continue;
			}
		}

		// Restrict the size of displayed frames
		if ((frame.size().height > 1080 * 0.75) || (frame.size().height > 1920 * 0.75))
			resize(frame, frame_display, Size(), display_scale, display_scale);

		// Show the frame
		addTime(&frame_display);
		imshow("Video", frame_display);

		// Create the trackbar
		slider = (int)(100 * cap.get(CAP_PROP_POS_FRAMES) / cap.get(CAP_PROP_FRAME_COUNT));
		createTrackbar(trackbarname, "Video", &slider, slider_max, onTrackbar);

		// Set the callback function for any mouse event
		if (paused)
		{
			// Select ROI
			setMouseCallback("Video", mouseEvent, &frame_display);
			// Draw ROI
			rectangle(frame_display, leftdown, leftup, Scalar(0, 255, 0));
			// Show frame with ROI
			imshow("Video", frame_display);

			if (selected)
			{
				// Get crop if the are of ROI is not zero
				if ((leftdown.x == leftup.x) || (leftdown.y == leftup.y))
				{
					cout << "Invalid crop:" << Rect(leftdown, leftup) << endl;
					selected = false;
				}
				else
				{
					destroyWindow("Target");
					getROI();
				}
			}
		}

		// Keyboard Operations
		int key = waitKey(10);
	
		if (key == 27)
			break;
		if ((key == 'P') || (key == 'p'))
			paused = !paused;
		if (key == 2424832)
			cap.set(CAP_PROP_POS_FRAMES, cap.get(CAP_PROP_POS_FRAMES) - step_sec * cap.get(CAP_PROP_FPS));
		if (key == 2555904)
			cap.set(CAP_PROP_POS_FRAMES, cap.get(CAP_PROP_POS_FRAMES) + step_sec * cap.get(CAP_PROP_FPS));
	}

	cap.release();
	cap_cache.release();
	destroyAllWindows();

	return 0;
}