#include "ofApp.h"

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include "ofxCv.h"

vector<float> ofApp::makeInput(float x, float y){
    // Ax + Bx^2 + Cy + Dy^2 + Exy + F
    vector<float> in;
    in.resize(inputCount);
    in[0] = x;
    in[1] = x * x;
    in[2] = y;
    in[3] = y * y;
    in[4] = x * y;
    in[5] = 1;
    return in;
}

vector<float> ofApp::makeOutput(float x, float y){
    vector<float> out;
    out.resize(outputCount);
    out[0] = x;
    out[1] = y;
    return out;
}

void ofApp::calculateWeight(){
    // calculate the weights for an equation that goes from tracked points to known points.
    
    int length = trackedPoints.size();
    
    ls.clear();
    
    for(int i = 0; i < length; i++) {
        ofPoint& ipt = trackedPoints[i];
        ofPoint& opt = knownPoints[i];
        ls.add(makeInput(ipt.x, ipt.y), makeOutput(opt.x, opt.y));
    }
    
    bBeenFit = true;
}

ofVec2f ofApp::getCalibratedPoint (float x, float y){
    
    if (bBeenFit == true){
        
        vector<float> out = ls.map(makeInput(x, y));
        
        float calibratedx = out[0];
        float calibratedy =    out[1];
        
        // ---------------------------------------------------
        // let's fix "offscreen" or very bad values, since we are smoothing them in....
        // and nans, etc will screw us up bad.
        
        if (calibratedx < -1000)    calibratedx = -1000;
        if (calibratedx > 2000)        calibratedx = 2000;
        if (calibratedy < -1000)    calibratedy = -1000;
        if (calibratedy > 2000)        calibratedy = 2000;
        
        if (isnan(calibratedx))        calibratedx = 0;
        if (isnan(calibratedy))        calibratedy = 0;
        // ---------------------------------------------------
        
        return ofVec2f(calibratedx, calibratedy);
    }
    return ofVec2f(0,0);
}

void ofApp::detectAndDisplay( cv::Mat frame, cv::Mat& outFrame) {
    std::vector<cv::Rect> faces;
    //cv::Mat frame_gray;
    // cv::Mat outFrame;
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(frame, rgbChannels);
    cv::Mat frame_gray = rgbChannels[2];
    
    //cvtColor( frame, frame_gray, CV_BGR2GRAY );
    //equalizeHist( frame_gray, frame_gray );
    //cv::pow(frame_gray, CV_64F, frame_gray);
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
    //  findSkin(debugImage);
    
    for( int i = 0; i < faces.size(); i++ )
    {
        rectangle(debugImage, faces[i], 1234);
    }
    //-- Show what you got
//    std::cout<<faces.size()<<std::endl;
    if (faces.size() > 0) {
        
        outFrame = findEyes(frame_gray, faces[0]);
//        std::cout<<outFrame.size()<<std::endl;
    }
    else{
        outFrame = frame_gray;
//        std::cout<<outFrame.size()<<std::endl;
    }
}

cv::Mat ofApp::findEyes(cv::Mat frame_gray, cv::Rect face) {
    cv::Mat faceROI = frame_gray(face);
    cv::Mat debugFace = faceROI;
    
    if (kSmoothFaceImage) {
        double sigma = kSmoothFaceFactor * face.width;
        GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
    }
    //-- Find eye regions and draw them
    int eye_region_width = face.width * (kEyePercentWidth/100.0);
    int eye_region_height = face.width * (kEyePercentHeight/100.0);
    int eye_region_top = face.height * (kEyePercentTop/100.0);
    cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                           eye_region_top,eye_region_width,eye_region_height);
    cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                            eye_region_top,eye_region_width,eye_region_height);
    
    //-- Find Eye Centers
    cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
    cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
   
    
    // get corner regions
    cv::Rect leftRightCornerRegion(leftEyeRegion);
    leftRightCornerRegion.width -= leftPupil.x;
    leftRightCornerRegion.x += leftPupil.x;
    leftRightCornerRegion.height /= 2;
    leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
    
    cv::Rect leftLeftCornerRegion(leftEyeRegion);
    leftLeftCornerRegion.width = leftPupil.x;
    leftLeftCornerRegion.height /= 2;
    leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
    
    cv::Rect rightLeftCornerRegion(rightEyeRegion);
    rightLeftCornerRegion.width = rightPupil.x;
    rightLeftCornerRegion.height /= 2;
    rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
    
    cv::Rect rightRightCornerRegion(rightEyeRegion);
    rightRightCornerRegion.width -= rightPupil.x;
    rightRightCornerRegion.x += rightPupil.x;
    rightRightCornerRegion.height /= 2;
    rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
    
    leftPupil.x += leftEyeRegion.x;
    leftPupil.y += leftEyeRegion.y;
    
    rightPupil.x += rightEyeRegion.x;
    rightPupil.y += rightEyeRegion.y;
    
    cv::Point midPoint(float(leftPupil.x+rightPupil.x)/2.0, float(leftPupil.y+rightPupil.y)/2.0);
    
    midPoint.x -= leftLeftCornerRegion.x;
    midPoint.y -= leftLeftCornerRegion.y;
    rectangle(debugFace,rightRightCornerRegion,200);
   
    // change eye centers to face coordinates
    rightPupil.x += rightEyeRegion.x;
    rightPupil.y += rightEyeRegion.y;
    leftPupil.x += leftEyeRegion.x;
    leftPupil.y += leftEyeRegion.y;
   
    double width = rightRightCornerRegion.x - leftLeftCornerRegion.x + rightRightCornerRegion.width;
    double height = leftLeftCornerRegion.height;
    
    double x_perc = midPoint.x/width;
    double y_perc = midPoint.y/height;
    x_perc = (x_perc-0.5+(1.0/hFactor))/(2.0/hFactor);
    y_perc = (y_perc-0.5+(1.0/vFactor))/(2.0/vFactor);
    //@@@@@@@@@@@
    //    std::cout<<"("<<x_perc<<","<<y_perc<<")"<<std::endl;
    
    midPoint.x = ofGetWindowWidth()*x_perc;
    midPoint.y = ofGetWindowHeight()*y_perc;
    // draw eye centers
//    circle(debugFace, rightPupil, 3, 1234);
//    circle(debugFace, leftPupil, 3, 1234);
    
    // //-- Find Eye Corners
    // if (/*kEnableEyeCorner*/0) {
    //   cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
    //   leftRightCorner.x += leftRightCornerRegion.x;
    //   leftRightCorner.y += leftRightCornerRegion.y;
    //   cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
    //   leftLeftCorner.x += leftLeftCornerRegion.x;
    //   leftLeftCorner.y += leftLeftCornerRegion.y;
    //   cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
    //   rightLeftCorner.x += rightLeftCornerRegion.x;
    //   rightLeftCorner.y += rightLeftCornerRegion.y;
    //   cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
    //   rightRightCorner.x += rightRightCornerRegion.x;
    //   rightRightCorner.y += rightRightCornerRegion.y;
    //   circle(faceROI, leftRightCorner, 3, 200);
    //   circle(faceROI, leftLeftCorner, 3, 200);
    //   circle(faceROI, rightLeftCorner, 3, 200);
    //   circle(faceROI, rightRightCorner, 3, 200);
    // }
//    cv::imwrite("tmp.jpg", debugFace);
    //@@@@@@@@@@@
    //std::cout<<"("<<midPoint.x<<","<<midPoint.y<<")"<<std::endl;


//    eyepoints[pointFlag].x = (leftPupil.x + rightPupil.x)/2 * mappingRate;
//    eyepoints[pointFlag].y = (leftPupil.y + rightPupil.y)/2 * mappingRate;
//    tmpPoint.set(pointArray[pointIndex].x, pointArray[pointIndex].y);
//    pointArray[pointIndex].set(midPoint.x, midPoint.y);
    MidPoint.set(midPoint.x, midPoint.y);
    
    tmpPoint.set((1-smoothRate)*midPoint.x + smoothRate*tmpPoint.x, (1-smoothRate)*midPoint.y + smoothRate*tmpPoint.y);
    std::cout<< tmpPoint.x << " " << tmpPoint.y << std::endl;
//    tmp_x = 0.0;
//    tmp_y = 0.0;
//    for(int i = 0; i < MA_window ; i++)
//    {
//        tmp_x += pointArray[i].x;
//        tmp_y += pointArray[i].y;
//    }
//    tmp_x = tmp_x - tmpPoint.x + pointArray[pointIndex].x;
//    tmp_y = tmp_y - tmpPoint.y + pointArray[pointIndex].y;
    ofVec2f ptAfterMap = getCalibratedPoint(tmpPoint.x, tmpPoint.y);
    
    std::cout<<ptAfterMap<<std::endl;
    eyepoints[pointFlag].set(ptAfterMap.x, ptAfterMap.y);
    //@@@@@@@@@@@@
    //std::cout<<eyepoints[pointFlag].x<<"   "<<eyepoints[pointFlag].y<<" eyepoints "<<std::endl;
    
    pointFlag = (pointFlag==0)?1:0;

//    imshow(face_window_name, faceROI);
    //  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
    //  cv::Mat destinationROI = debugImage( roi );
    //  faceROI.copyTo( destinationROI );
    pointIndex = (pointIndex>(MA_window-2)?0:(pointIndex+1)) ;
    
    if(!startFlag && pointIndex == 0){
        startFlag = true;
    }
    return debugFace;
}

//string ExePath() {
//    char buffer[MAX_PATH];
//    GetModuleFileName( NULL, buffer, MAX_PATH );
//    string::size_type pos = string( buffer ).find_last_of( "\\/" );
//    return string( buffer ).substr( 0, pos);
//}


//---------------------------------------
void ofApp::setup(){

    if( !face_cascade.load( "/Volumes/MyFiles/Work/of_v0.10.1_osx_release/apps/myApps/eye_tracking_try5/bin/data/haarcascade_frontalface_alt.xml" )){
        
        printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
        return -1;
        
    };

    cam.setup(1280, 720);
    
    startFlag = 0;
    readImg.load("images/article_0.jpg");
    ofBackground(0,0,0);
    ofSetBackgroundAuto(false);
    
    pointFlag = 0;
    pointIndex = 0;
//    ofVec2f tmp[MA_window];
    
    pointArray = new ofVec2f[MA_window];
    for(int i = 0; i < MA_window; i++)
    {
        pointArray[i].set(0.0,0.0);
    }
    tmpPoint.set(0.0, 0.0);
    tmp_x = 0.0;
    tmp_y = 0.0;
    vFactor = 2.0;
    hFactor = 10.0;
    smoothRate = 0.96;
    MA_window = 2/(1-smoothRate);
    georgia.load("Georgia.ttf", 14);
    georgia.setLineHeight(30.0f);
    georgia.setLetterSpacing(1.037);
    
    levelMark = 0;
    
    stareTime = 2;

    isCallibrating = false;
    
    Dot[0] = ofVec2f(ofGetWindowWidth()/6*1, ofGetWindowHeight()/5);
    Dot[1] = ofVec2f(ofGetWindowWidth()/6*2, ofGetWindowHeight()/5);
    Dot[2] = ofVec2f(ofGetWindowWidth()/6*3, ofGetWindowHeight()/5);
    Dot[3] = ofVec2f(ofGetWindowWidth()/6*4, ofGetWindowHeight()/5);
    Dot[4] = ofVec2f(ofGetWindowWidth()/6*5, ofGetWindowHeight()/5);
    
    Dot[5] = ofVec2f(ofGetWindowWidth()/6*1, ofGetWindowHeight()/5*2);
    Dot[6] = ofVec2f(ofGetWindowWidth()/6*2, ofGetWindowHeight()/5*2);
    Dot[7] = ofVec2f(ofGetWindowWidth()/6*3, ofGetWindowHeight()/5*2);
    Dot[8] = ofVec2f(ofGetWindowWidth()/6*4, ofGetWindowHeight()/5*2);
    Dot[9] = ofVec2f(ofGetWindowWidth()/6*5, ofGetWindowHeight()/5*2);
    
    Dot[10] = ofVec2f(ofGetWindowWidth()/6*1, ofGetWindowHeight()/5*3);
    Dot[11] = ofVec2f(ofGetWindowWidth()/6*2, ofGetWindowHeight()/5*3);
    Dot[12] = ofVec2f(ofGetWindowWidth()/6*3, ofGetWindowHeight()/5*3);
    Dot[13] = ofVec2f(ofGetWindowWidth()/6*4, ofGetWindowHeight()/5*3);
    Dot[14] = ofVec2f(ofGetWindowWidth()/6*5, ofGetWindowHeight()/5*3);
    
    Dot[15] = ofVec2f(ofGetWindowWidth()/6*1, ofGetWindowHeight()/5*4);
    Dot[16] = ofVec2f(ofGetWindowWidth()/6*2, ofGetWindowHeight()/5*4);
    Dot[17] = ofVec2f(ofGetWindowWidth()/6*3, ofGetWindowHeight()/5*4);
    Dot[18] = ofVec2f(ofGetWindowWidth()/6*4, ofGetWindowHeight()/5*4);
    Dot[19] = ofVec2f(ofGetWindowWidth()/6*5, ofGetWindowHeight()/5*4);
    
    for (int i = 0; i < 20; i++){
        firstRun[i] = true;
    }
    inputCount = 6;
    outputCount = 2;
    ls.setup(inputCount, outputCount);
}

//-------------------------------------------------
void ofApp::update(){

    cam.update();
    if(cam.isFrameNew()){
        frame = toCv(cam.getPixels());
        
        // mirror it
        cv::flip(frame, frame, 1);
        frame.copyTo(debugImage);
        detectAndDisplay(frame, ooo);

        toOf(ooo, Img);

        Img.update();
    }
    
    if(isCallibrating){
        
        avg_x += MidPoint.x;
        avg_y += MidPoint.y;
        counter ++;
        
    }
}

//--------------------------------------------------------------

void ofApp::draw(){
    
    //initial text
    if(levelMark == 0){
        
    ofSetColor(0,255,0);
    georgia.drawString("Please press ENTER to start.", 50, ofGetWindowHeight()/2);
        
    }else if(levelMark == 1){
        
        ofSetColor(0);
        ofDrawRectangle(0, 0, ofGetWindowWidth(),ofGetWindowHeight());
        callibrationTimer = ofGetElapsedTimef();
       // std::cout<<callibrationTimer<<endl;
        
        
        //draw H points
        //--------------------line 1----------------------
        ofSetColor(255);
        
        if(callibrationTimer > 0 && callibrationTimer <= stareTime){
            
            ofDrawCircle(Dot[0].x, Dot[0].y, 10);

            
        }else if (callibrationTimer < stareTime*2){
            ofDrawCircle(Dot[1].x, Dot[1].y, 10);
            
            if(firstRun[1]){
                callibrate();
                firstRun[1] = false;
            }
            
        }else if (callibrationTimer < stareTime*3){
           ofDrawCircle(Dot[2].x, Dot[2].y, 10);
            
            if(firstRun[2]){
                callibrate();
                firstRun[2] = false;
            }
            
        }else if (callibrationTimer < stareTime*4){
            ofDrawCircle(Dot[3].x, Dot[3].y, 10);
            
            if(firstRun[3]){
                callibrate();
                firstRun[3] = false;
            }
            
        }
        else if (callibrationTimer < stareTime*5){
            ofDrawCircle(Dot[4].x, Dot[4].y, 10);
            
            if(firstRun[4]){
                callibrate();
                firstRun[4] = false;
            }
            
        }//--------------------line 2----------------------
       
        else if (callibrationTimer < stareTime*6){
            ofDrawCircle(Dot[5].x, Dot[5].y, 10);
            
            if(firstRun[5]){
                callibrate();
                firstRun[5] = false;
            }
            
        }else if (callibrationTimer < stareTime*7){
            ofDrawCircle(Dot[6].x, Dot[6].y, 10);
            
            if(firstRun[6]){
                callibrate();
                firstRun[6] = false;
            }
            
        }else if (callibrationTimer < stareTime*8){
            ofDrawCircle(Dot[7].x, Dot[7].y, 10);
            
            if(firstRun[7]){
                callibrate();
                firstRun[7] = false;
            }
            
        }else if (callibrationTimer < stareTime*9){
            ofDrawCircle(Dot[8].x, Dot[8].y, 10);
            
            if(firstRun[8]){
                callibrate();
                firstRun[8] = false;
            }
            
        }else if (callibrationTimer < stareTime*10){
            ofDrawCircle(Dot[9].x, Dot[9].y, 10);
            
            if(firstRun[9]){
                callibrate();
                firstRun[9] = false;
            }
        }//--------------------line 3----------------------
       
        else if (callibrationTimer < stareTime*11){
            ofDrawCircle(Dot[10].x, Dot[10].y, 10);
            
            if(firstRun[10]){
                callibrate();
                firstRun[10] = false;
            }
        }else if (callibrationTimer < stareTime*12){
            ofDrawCircle(Dot[11].x, Dot[11].y, 10);
            
            if(firstRun[11]){
                callibrate();
                firstRun[11] = false;
            }
        }else if (callibrationTimer < stareTime*13){
            ofDrawCircle(Dot[12].x, Dot[12].y, 10);
            
            if(firstRun[12]){
                callibrate();
                firstRun[12] = false;
            }
        }else if (callibrationTimer < stareTime*14){
            ofDrawCircle(Dot[13].x, Dot[13].y, 10);
            
            if(firstRun[13]){
                callibrate();
                firstRun[13] = false;
            }
        }else if (callibrationTimer < stareTime*15){
            ofDrawCircle(Dot[14].x, Dot[14].y, 10);
            
            if(firstRun[14]){
                callibrate();
                firstRun[14] = false;
            }
        }//--------------------line 4---------------------
        
        else if (callibrationTimer < stareTime*16){
           ofDrawCircle(Dot[15].x, Dot[15].y, 10);
            
            if(firstRun[15]){
                callibrate();
                firstRun[15] = false;
            }
        }else if (callibrationTimer < stareTime*17){
            ofDrawCircle(Dot[16].x, Dot[16].y, 10);
            
            if(firstRun[16]){
                callibrate();
                firstRun[16] = false;
            }
        }else if (callibrationTimer < stareTime*18){
            ofDrawCircle(Dot[17].x, Dot[17].y, 10);
            
            if(firstRun[17]){
                callibrate();
                firstRun[17] = false;
            }
        }else if (callibrationTimer < stareTime*19){
            ofDrawCircle(Dot[18].x, Dot[18].y, 10);
            
            if(firstRun[18]){
                callibrate();
                firstRun[18] = false;
            }
        }else if (callibrationTimer < stareTime*20){
            ofDrawCircle(Dot[19].x, Dot[19].y, 10);
            
            if(firstRun[19]){
                callibrate();
                firstRun[19] = false;
            }
        }else{
            
            }
        
    }else if(levelMark == 2){
        
        if(startFlag){
            
            //            ofSetColor(255);
            //            if(Img.isAllocated()){
            //                Img.draw(10, 500);
            //            }
            
            ofSetColor(255, 0, 0);
            ofSetLineWidth(3.0f);
            ofDrawLine(eyepoints[0].x, eyepoints[0].y, eyepoints[1].x, eyepoints[1].y);
        }
        
        
    }
    
    
    
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    
    //level 1: calibrate
    if(key == OF_KEY_RETURN){
        levelMark = 1;
        isCallibrating = true;
        ofResetElapsedTimeCounter();
    }
    
    //save screenshot
    if(key == 's'){
        screenshot.grabScreen(0, 0 , ofGetWidth(), ofGetHeight());
        screenshot.save("screenshot.png");
        startFlag = 0;
    }
    
    //draw rectangles
    if(key == 'r'){
        calculateWeight();
        levelMark = 2;
        
        //draw initial article
        ofSetColor(0, 0, 0);
        ofDrawRectangle(0, 0, ofGetWindowWidth(),ofGetWindowHeight());
        
        ofSetColor(255);
        georgia.drawString("What we read, how we read, and why we read change \n how we think, changes that are continuing now at a \n faster pace. In a span of only six millennia reading \n became the transformative catalyst for intellectual \n development within individuals and within literate \n cultures. The quality of our reading is not only an index \n of the quality of our thought, it is our bestknown path \n to developing whole new pathways in the cerebral \n evolution of our species. There is much at stake in the \n development of the reading brain and in the quickening \n changes that now characterize its current, evolving \n iterations.", 30, 50);
        
        georgia.drawString("You need only examine yourself. Perhaps you have \nalready noticed how the quality of your attention has \nchanged the more you read on screens and digital \ndevices. Perhaps you have felt a pang of something \nsubtle that is missing when you seek to immerse \nyourself in a once favorite book. Like a phantom limb, \nyou remember who you were as a reader, but cannot \nsummon that “attentive ghost” with the joy you once felt \nin being transported somewhere outside the self to \nthat interior space. ", 30, 450);
    }
}



//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}

void ofApp::callibrate(){
    
    int dotIndex = knownPoints.size();
    trackedPoints.push_back(ofPoint(avg_x/counter,avg_y/counter));
    knownPoints.push_back(Dot[dotIndex]);
    
    avg_y = 0;
    avg_x = 0;
    counter = 0;

    
}


