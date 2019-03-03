#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "LeastSquare.h"


using namespace ofxCv;
//int MAX = 10000;


class ofApp : public ofBaseApp{

	public:

    
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
        void detectAndDisplay( cv::Mat frame, cv::Mat& outFrame);
        cv::Mat findEyes(cv::Mat frame_gray, cv::Rect face);
        void calculateWeight();
        ofxLeastSquares ls;
        bool bBeenFit;
        void callibrate();
        bool isCallibrating;
    
        float avg_x;
        float avg_y;
        int counter;
        ofVec2f Dot[20];
        bool firstRun[21];
    int inputCount;
    int outputCount;
        vector<float> makeInput(float x, float y);
        vector<float> makeOutput(float x, float y);
    ofVec2f getCalibratedPoint(float x, float y);
    vector <ofPoint> trackedPoints;// Points BEFORE mapping
    vector <ofPoint> knownPoints; // This vector consists of points AFTER mapping
        //void drawResult();
    
    ofImage readImg;
    ofImage resultImg;
    
    //void setCap(cv::VideoCapture& cap_);
		
    ofVideoGrabber cam;
    cv::Mat frame;
    cv::Mat ooo;

    cv::CascadeClassifier face_cascade;

    ofImage Img;
    std::string face_window_name = "Capture - Face";
  //cv::RNG rng(12345);
    cv::Mat debugImage;
    cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
    
    int MA_window;
    ofVec2f *pointArray;
    ofVec2f eyepoints[2];
    ofVec2f MidPoint;
    int pointFlag;
    int pointIndex;
    ofVec2f tmpPoint;
    double smoothRate;
    double tmp_x;
    double tmp_y;
    bool startFlag;
    double vFactor;
    double hFactor;
    
    ofImage screenshot;
    bool startRead;
    
    ofTrueTypeFont georgia;
    int levelMark;
    
    float callibrationTimer;
    float stareTime;
    
};
