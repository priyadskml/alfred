#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include <unistd.h>
#include <fstream>

#define ORIGCOL2COL COLOR_BGR2HLS
#define COL2ORIGCOL COLOR_HLS2BGR
#define NSAMPLES 7
#define PI 3.14159

bool check ;

using namespace cv;
using namespace std;

//*********************** roi.hpp*****************
class My_ROI{
	public:
		My_ROI();
		My_ROI(Point upper_corner, Point lower_corner,Mat src);
		Point upper_corner, lower_corner;
		Mat roi_ptr;
		Scalar color;
		int border_thickness;
		void draw_rectangle(Mat src);
};
//********************roi.cpp***********************
My_ROI::My_ROI(){
	upper_corner=Point(0,0);
	lower_corner=Point(0,0);

}

My_ROI::My_ROI(Point u_corner, Point l_corner, Mat src){
	upper_corner=u_corner;
	lower_corner=l_corner;
	color=Scalar(0,255,0);
	border_thickness=2;
	roi_ptr=src(Rect(u_corner.x, u_corner.y, l_corner.x-u_corner.x,l_corner.y-u_corner.y));
}

void My_ROI::draw_rectangle(Mat src){
	rectangle(src,upper_corner,lower_corner,color,border_thickness);

}

//*****************myImage.hpp*************************

class MyImage{
	public:
		MyImage(int webCamera);
		MyImage();
		Mat srcLR;
		Mat src;
		Mat bw;
		vector<Mat> bwList;
		VideoCapture cap;		
		int cameraSrc; 
		void initWebCamera(int i);
};
//*********************myImage.cpp*****************

MyImage::MyImage(){
}

MyImage::MyImage(int webCamera){
	cameraSrc=webCamera;
	cap=VideoCapture(webCamera);
}
#include "handgesture.hpp"

// //motiondetect class
// class motiondetect
// {
// public:
// 	Mat back_sub(Mat frame){
// 	Mat back;
//     Mat fore;
//     //BackgroundSubtractorMOG2 bg(5,3,true) ;
//     //bg = new cv::BackgroundSubtractorMOG2(10, 16, false);
//     namedWindow("Background");
//     vector<std::vector<cv::Point> > contours;
//     cv::Ptr<cv::BackgroundSubtractor> pMOG2;
//     pMOG2 = cv::createBackgroundSubtractorMOG2();
//     pMOG2->apply(frame, fore);
//     //bg.operator ()(frame,fore);
//     pMOG2->getBackgroundImage(back);
//     erode(fore,fore,cv::Mat());
//     dilate(fore,fore,cv::Mat());
//     findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
//     drawContours(frame,contours,-1,cv::Scalar(0,0,255),2);
//     return frame;

// 	}
// 	/* data */
// };

//->global variables need to check again
//*************************main.cpp***************************
int fontFace = FONT_HERSHEY_PLAIN;
int square_len;
int avgColor[NSAMPLES][3] ;
int c_lower[NSAMPLES][3];
int c_upper[NSAMPLES][3];
int avgBGR[3];
int nrOfDefects;
int iSinceKFInit;
struct dim{int w; int h;}boundingDim;
	VideoWriter out;
Mat edges;
My_ROI roi1, roi2,roi3,roi4,roi5,roi6;//add number of rectangles
vector <My_ROI> roi;
vector <KalmanFilter> kf;
vector <Mat_<float> > measurement;

//createdvar
int calib_val_h=600;
int calib_val_w=400;

float ration=1.5;
//createdvar

/* end global variables */

void init(MyImage *m){
	square_len=10;		//box size for palm 
	iSinceKFInit=0;		//useless variable not used anywhere
}

// change a color from one space to another
void col2origCol(int hsv[3], int bgr[3], Mat src){
	Mat avgBGRMat=src.clone();	
	for(int i=0;i<3;i++){
		avgBGRMat.data[i]=hsv[i];	
	}
	cvtColor(avgBGRMat,avgBGRMat,COL2ORIGCOL);
	for(int i=0;i<3;i++){
		bgr[i]=avgBGRMat.data[i];	
	}
}

void printText(Mat src, string text){
	int fontFace = FONT_HERSHEY_PLAIN;
	putText(src,text,Point(src.cols/2, src.rows/10),fontFace, 1.2f,Scalar(200,0,0),2);
}

void waitForPalmCover(MyImage* m){
    m->cap >> m->src;
	flip(m->src,m->src,1);
	roi.push_back(My_ROI(Point(m->src.cols/3, m->src.rows/6),Point(m->src.cols/3+square_len,m->src.rows/6+square_len),m->src));
	roi.push_back(My_ROI(Point(m->src.cols/4, m->src.rows/2),Point(m->src.cols/4+square_len,m->src.rows/2+square_len),m->src));
	roi.push_back(My_ROI(Point(m->src.cols/3, m->src.rows/1.5),Point(m->src.cols/3+square_len,m->src.rows/1.5+square_len),m->src));
	roi.push_back(My_ROI(Point(m->src.cols/2, m->src.rows/2),Point(m->src.cols/2+square_len,m->src.rows/2+square_len),m->src));
	roi.push_back(My_ROI(Point(m->src.cols/2.5, m->src.rows/2.5),Point(m->src.cols/2.5+square_len,m->src.rows/2.5+square_len),m->src));
	roi.push_back(My_ROI(Point(m->src.cols/2, m->src.rows/1.5),Point(m->src.cols/2+square_len,m->src.rows/1.5+square_len),m->src));
	roi.push_back(My_ROI(Point(m->src.cols/2.5, m->src.rows/1.8),Point(m->src.cols/2.5+square_len,m->src.rows/1.8+square_len),m->src));
	//roi.push_back(My_ROI(Point(m->src.cols/2, m->src.rows/3.3),Point(m->src.cols/2.5+square_len,m->src.rows/1.8+square_len),m->src));
	
	
	for(int i =0;i<50;i++){
    	m->cap >> m->src;
		flip(m->src,m->src,1);
		for(int j=0;j<NSAMPLES;j++){
			roi[j].draw_rectangle(m->src);
		}
		string imgText=string("Cover rectangles with palm");
		printText(m->src,imgText);	
		
		if(i==30){
		//	imwrite("./images/waitforpalm1.jpg",m->src);
		}

		imshow("img1", m->src);
		out << m->src;
        if(cv::waitKey(30) >= 0) break;
	}
}

int getMedian(vector<int> val){
  int median;
  size_t size = val.size();
  sort(val.begin(), val.end());
  if (size  % 2 == 0)  {
      median = val[size / 2 - 1] ;
  } else{
      median = val[size / 2];
  }
  return median;
}


void getAvgColor(MyImage *m,My_ROI roi,int avg[3]){
	Mat r;
	roi.roi_ptr.copyTo(r);
	vector<int>hm;
	vector<int>sm;
	vector<int>lm;
	// generate vectors
	for(int i=2; i<r.rows-2; i++){
    	for(int j=2; j<r.cols-2; j++){
    		hm.push_back(r.data[r.channels()*(r.cols*i + j) + 0]) ;
        	sm.push_back(r.data[r.channels()*(r.cols*i + j) + 1]) ;
        	lm.push_back(r.data[r.channels()*(r.cols*i + j) + 2]) ;
   		}
	}
	avg[0]=getMedian(hm);
	avg[1]=getMedian(sm);
	avg[2]=getMedian(lm);
}

void average(MyImage *m){
	m->cap >> m->src;
	flip(m->src,m->src,1);
	for(int i=0;i<30;i++){
		m->cap >> m->src;
		flip(m->src,m->src,1);
		cvtColor(m->src,m->src,ORIGCOL2COL);
		for(int j=0;j<NSAMPLES;j++){
			getAvgColor(m,roi[j],avgColor[j]);
			roi[j].draw_rectangle(m->src);
		}	
		cvtColor(m->src,m->src,COL2ORIGCOL);
		string imgText=string("Finding average color of hand");
		printText(m->src,imgText);	
		imshow("img1", m->src);
        if(cv::waitKey(30) >= 0) break;
	}
}

void initTrackbars(){
	for(int i=0;i<NSAMPLES;i++){
		c_lower[i][0]=12;
		c_upper[i][0]=7;
		c_lower[i][1]=30;
		c_upper[i][1]=40;
		c_lower[i][2]=80;
		c_upper[i][2]=80;
	}
	createTrackbar("lower1","trackbars",&c_lower[0][0],255);
	createTrackbar("lower2","trackbars",&c_lower[0][1],255);
	createTrackbar("lower3","trackbars",&c_lower[0][2],255);
	createTrackbar("upper1","trackbars",&c_upper[0][0],255);
	createTrackbar("upper2","trackbars",&c_upper[0][1],255);
	createTrackbar("upper3","trackbars",&c_upper[0][2],255);
}


void normalizeColors(MyImage * myImage){
	// copy all boundries read from trackbar
	// to all of the different boundries
	for(int i=1;i<NSAMPLES;i++){
		for(int j=0;j<3;j++){
			c_lower[i][j]=c_lower[0][j];	
			c_upper[i][j]=c_upper[0][j];	
		}	
	}
	// normalize all boundries so that 
	// threshold is whithin 0-255
	for(int i=0;i<NSAMPLES;i++){
		if((avgColor[i][0]-c_lower[i][0]) <0){
			c_lower[i][0] = avgColor[i][0] ;
		}if((avgColor[i][1]-c_lower[i][1]) <0){
			c_lower[i][1] = avgColor[i][1] ;
		}if((avgColor[i][2]-c_lower[i][2]) <0){
			c_lower[i][2] = avgColor[i][2] ;
		}if((avgColor[i][0]+c_upper[i][0]) >255){ 
			c_upper[i][0] = 255-avgColor[i][0] ;
		}if((avgColor[i][1]+c_upper[i][1]) >255){
			c_upper[i][1] = 255-avgColor[i][1] ;
		}if((avgColor[i][2]+c_upper[i][2]) >255){
			c_upper[i][2] = 255-avgColor[i][2] ;
		}
	}
}

void produceBinaries(MyImage *m){
	//here could be size defect	
	Scalar lowerBound;
	Scalar upperBound;
	Mat foo;
	for(int i=0;i<NSAMPLES;i++){
		normalizeColors(m);
		lowerBound=Scalar( avgColor[i][0] - c_lower[i][0] , avgColor[i][1] - c_lower[i][1], avgColor[i][2] - c_lower[i][2] );
		upperBound=Scalar( avgColor[i][0] + c_upper[i][0] , avgColor[i][1] + c_upper[i][1], avgColor[i][2] + c_upper[i][2] );
		m->bwList.push_back(Mat(m->srcLR.rows,m->srcLR.cols,CV_8U));	
		inRange(m->srcLR,lowerBound,upperBound,m->bwList[i]);	
	}
	m->bwList[0].copyTo(m->bw);
	for(int i=1;i<NSAMPLES;i++){
		m->bw+=m->bwList[i];	
	}
	medianBlur(m->bw, m->bw,7);
}

void initWindows(MyImage m){
    namedWindow("trackbars",WINDOW_FULLSCREEN);
    namedWindow("img1",WINDOW_FULLSCREEN);
}

void showWindows(MyImage m){
	pyrDown(m.bw,m.bw);
	pyrDown(m.bw,m.bw);
	Rect roi( Point( 3*m.src.cols/4,0 ), m.bw.size());
	vector<Mat> channels;
	Mat result;
	for(int i=0;i<3;i++)
		channels.push_back(m.bw);
	merge(channels,result);
	result.copyTo( m.src(roi));
	imshow("img1",m.src);	
}

int findBiggestContour(vector<vector<Point> > contours){
    int indexOfBiggestContour = -1;
    int sizeOfBiggestContour = 0;
    for (int i = 0; i < contours.size(); i++){
        if(contours[i].size() > sizeOfBiggestContour){
            sizeOfBiggestContour = contours[i].size();
            indexOfBiggestContour = i;
        }
    }
    return indexOfBiggestContour;
}

void myDrawContours(MyImage *m,HandGesture *hg){
	drawContours(m->src,hg->hullP,hg->cIdx,cv::Scalar(200,0,0),2, 8, vector<Vec4i>(), 0, Point());




	rectangle(m->src,hg->bRect.tl(),hg->bRect.br(),Scalar(0,0,200));
	vector<Vec4i>::iterator d=hg->defects[hg->cIdx].begin();
	int fontFace = FONT_HERSHEY_PLAIN;
		
	
	vector<Mat> channels;
		Mat result;
		for(int i=0;i<3;i++)
			channels.push_back(m->bw);
		merge(channels,result);
	//	drawContours(result,hg->contours,hg->cIdx,cv::Scalar(0,200,0),6, 8, vector<Vec4i>(), 0, Point());
		drawContours(result,hg->hullP,hg->cIdx,cv::Scalar(0,0,250),10, 8, vector<Vec4i>(), 0, Point());

		
	while( d!=hg->defects[hg->cIdx].end() ) {
   	    Vec4i& v=(*d);
	    int startidx=v[0]; Point ptStart(hg->contours[hg->cIdx][startidx] );
   		int endidx=v[1]; Point ptEnd(hg->contours[hg->cIdx][endidx] );
  	    int faridx=v[2]; Point ptFar(hg->contours[hg->cIdx][faridx] );
	    float depth = v[3] / 256;
   /*	
		line( m->src, ptStart, ptFar, Scalar(0,255,0), 1 );
	    line( m->src, ptEnd, ptFar, Scalar(0,255,0), 1 );
   		circle( m->src, ptFar,   4, Scalar(0,255,0), 2 );
   		circle( m->src, ptEnd,   4, Scalar(0,0,255), 2 );
   		circle( m->src, ptStart,   4, Scalar(255,0,0), 2 );
*/
   		circle( result, ptFar,   9, Scalar(0,205,0), 5 );
		
		
	    d++;

   	 }
//	imwrite("./images/contour_defects_before_eliminate.jpg",result);

}

void makeContours(MyImage *m, HandGesture* hg){
	Mat aBw;
	pyrUp(m->bw,m->bw);
	m->bw.copyTo(aBw);
	findContours(aBw,hg->contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);
	hg->initVectors(); 
	hg->cIdx=findBiggestContour(hg->contours);
	if(hg->cIdx!=-1){
//		approxPolyDP( Mat(hg->contours[hg->cIdx]), hg->contours[hg->cIdx], 11, true );
		hg->bRect=boundingRect(Mat(hg->contours[hg->cIdx]));		
		convexHull(Mat(hg->contours[hg->cIdx]),hg->hullP[hg->cIdx],false,true);
		convexHull(Mat(hg->contours[hg->cIdx]),hg->hullI[hg->cIdx],false,false);
		approxPolyDP( Mat(hg->hullP[hg->cIdx]), hg->hullP[hg->cIdx], 18, true );
		if(hg->contours[hg->cIdx].size()>3 ){
			convexityDefects(hg->contours[hg->cIdx],hg->hullI[hg->cIdx],hg->defects[hg->cIdx]);
			//convexityDefects(hg->contours[hg->cIdx],hg->hullI[hg->cIdx],hg->defects1[hg->cIdx]);
			check=hg->find_pointing_finger(m);
			int k = hg->directions(m);
			if(check)
			{
				if(hg->fingerTips.size()>=3)
					check =true;//change 1
				else
				{
					ofstream myfile ("serial.txt");
					if(myfile.is_open())
					{
						myfile << k ;
						myfile << "\n";
					}
					myfile.close();

				}
			}

			hg->eleminateDefects(m);
		}
		bool isHand=hg->detectIfHand();
		hg->printGestureInfo(m->src);
		if(isHand){	
			hg->getFingerTips(m);
			hg->drawFingerTips(m);
			myDrawContours(m,hg);
		}
	}
}
//make contours with defects
void makeContoursWithDefects(MyImage *m, HandGesture* hg){
	Mat aBw;
	pyrUp(m->bw,m->bw);
	m->bw.copyTo(aBw);
	findContours(aBw,hg->contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);
	hg->initVectors(); 
	hg->cIdx=findBiggestContour(hg->contours);
	if(hg->cIdx!=-1){
//		approxPolyDP( Mat(hg->contours[hg->cIdx]), hg->contours[hg->cIdx], 11, true );
		hg->bRect=boundingRect(Mat(hg->contours[hg->cIdx]));		
		convexHull(Mat(hg->contours[hg->cIdx]),hg->hullP[hg->cIdx],false,true);
		convexHull(Mat(hg->contours[hg->cIdx]),hg->hullI[hg->cIdx],false,false);
		approxPolyDP( Mat(hg->hullP[hg->cIdx]), hg->hullP[hg->cIdx], 18, true );
		if(hg->contours[hg->cIdx].size()>3 ){
			convexityDefects(hg->contours[hg->cIdx],hg->hullI[hg->cIdx],hg->defects[hg->cIdx]);
			hg->eleminateDefects(m);
		}
		bool isHand=hg->detectIfHand();
		hg->printGestureInfo(m->src);
		if(isHand){	
			hg->getFingerTips(m);
			hg->drawFingerTips(m);
			myDrawContours(m,hg);
		}
		else
			myDrawContours(m,hg);
	}
}

int angleInRadians(MyImage X)
{
	return 0;
}

//boolean functon to return the fist HandGesture

int  FistGesture(MyImage * m , HandGesture * hg)
{
	bool isHand = hg->detectIfHand_sm();
	//cout<<
	if(hg->fingerTips.size()<3){
		//cout<<"111111"<<endl;
		
		
		Rect fist_Rect=boundingRect(Mat(hg->contours[hg->cIdx]));
		if(fist_Rect.height>0)
		{
			if (fist_Rect.height < calib_val_h/1.55)
			{
				//cout<<"333333"<<endl;
				if (fist_Rect.width < calib_val_w)
				{
					/* code */
					return 1;
				}
				//return 1;
			}
		}
		
		
	}
	else if (hg->fingerTips.size()>3 && hg->fingerTips.size()<7)
	{
		cout << "chandura" <<endl<<endl;
		return 2;
	}
	else 
	{
	return -1;

	}
		return -1;
}

int niranjanBest(MyImage * m , HandGesture * hg)
{
	//usleep(8 * 1000);
	//write for 40 frames
	for(int i=0;i<50;i++){
		cout << i <<endl <<endl;
		hg->frameNumber++;
		m->cap >> m->src;
		flip(m->src,m->src,1);
		pyrDown(m->src,m->srcLR);
		blur(m->srcLR,m->srcLR,Size(3,3));
		cvtColor(m->srcLR,m->srcLR,ORIGCOL2COL);
		produceBinaries(m);
		cvtColor(m->srcLR,m->srcLR,COL2ORIGCOL);
		//cout<<"9999999"<<endl;
		//cout<<endl;
		makeContoursWithDefects(m, hg);
		//cout<<"kbjhds"<<endl;
		hg->getFingerNumber(m);
		int t = FistGesture(m,hg);
		if(t!=-1)
		{
			check =false;
			return t;
		}
		showWindows(*m);
		out << m->src;
        if(cv::waitKey(30) >= 0) break;

	}
	check = false;
	return -1;	 
}

int main(){
	check=false;
	MyImage m(0);		
	HandGesture hg;
	init(&m);		
	m.cap >>m.src;
    namedWindow("img1",WINDOW_FULLSCREEN);
    Mat imageFirst;
    HandGesture g;
    //make a new handgesture and store it as first one then compare the hand gestures so that 
    //it can be used for comparision
	//out.open("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, m.src.size(), true);
	waitForPalmCover(&m);
	average(&m);
	destroyWindow("img1");
	initWindows(m);
	initTrackbars();
	for(;;){
		hg.frameNumber++;
		m.cap >> m.src;
		flip(m.src,m.src,1);
		pyrDown(m.src,m.srcLR);
		blur(m.srcLR,m.srcLR,Size(3,3));
		cvtColor(m.srcLR,m.srcLR,ORIGCOL2COL);
		produceBinaries(&m);
		cvtColor(m.srcLR,m.srcLR,COL2ORIGCOL);
		//cout<<"9999999"<<endl;
		//cout<<endl;

		makeContours(&m, &hg);
		cout<<"after make counter"<<endl;
		// hg.getFingerNumber(&m);
		// int t = FistGesture(&m,&hg);
		// //cout<<"------->>>>>>"<<hg.bRect.height<<endl <<endl;;
		// if(t==1)
		// {
		// 	cout  << "your hand is detected" << endl;
		// }
		//int k = hg.find_pointing_finger(&m);
		if(check == true)
		{
			cout << "its pointing" <<endl;
			int k = niranjanBest(&m,&hg);
			ofstream myfile("serial.txt",std::fstream::app);
					
			if(k==1)
			{
				if(myfile.is_open())
				{
					myfile << "1" << endl;
				}
				myfile.close();


				cout << "smFuncWritten" <<endl;
			}
			else if (k==2)
			{
				cout << "chanduFunc" <<endl;

				if(myfile.is_open())
				{
					myfile << "0" << endl;
				}
				myfile.close();

			}
			else
				cout << "na tera na mera" <<endl;

			
		}
		showWindows(m);
		out << m.src;
		//imwrite("./images/final_result.jpg",m.src);
    	if(cv::waitKey(30) == char('q')) break;
	 }
	destroyAllWindows();
	out.release();
	m.cap.release();
    return 0;
}