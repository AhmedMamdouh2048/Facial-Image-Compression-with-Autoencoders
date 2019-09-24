#pragma once
#ifndef DATASET_H_INCLUDED
#define DATASET_H_INCLUDED
#include <fstream>
#include <vector>
#include <conio.h>
#include <thread>
#include <dirent.h>
#include "Matrix.h"
#include "AE_Tools.h"
#include "opencv2\opencv.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
using namespace cv;
//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////  GET DATASET FROM HARD DISK   ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void Get_TrainSet(Arguments& Arg, DatasetParam& DP);     // Reads the TrainSet from the hard disk and puts them in matrix X and matrix Y
void Get_TestSet(Arguments& Arg, DatasetParam& DP);      // Reads the TestSet from the hard disk and puts them in matrix X and matrix Y
void Shuffle(Matrix* X);		                       // Randomly Shuffles matrix X and maatrix Y
void Shuffle(U_IntMatrix* X);                          // Randomly Shuffles matrix X and maatrix Y
void SWAP(U_IntMatrix* MAT, int i, int k);             // Swaps the ith column with the kth column in MAT
void SWAP(Matrix* MAT, int i, int k);                  // Swaps the ith column with the kth column in MAT
void DisplayData(Arguments& Arg, DatasetParam& DP);      // Displays data in the previously specified DP.Disp_dir
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////  Gaussian Blur   ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* pad(Matrix* img, int p, float value);
Matrix* FilterToMatrix(Matrix* filter, int nh, int nw, int s);
Matrix* to_SquareMat(Matrix* X);
Matrix* convolve(Matrix* Aprev, Matrix* Filter, int s);
Matrix* gausianFilter(int x, int y, float sigma);                                       // Returns a gaussian filter (x,y) with standard deviation sigma
Matrix* gaussianBlur(Matrix* img, int filterSize, float sigma);                         // Performs a gaussian blur to a 2D img
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////  Visualization   //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void visualize(Matrix* img);
void visualize(Matrix* img1, Matrix* img2);
void to_JPG(U_IntMatrix* X, string PATH);
//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////  Face feature extracting and editing   ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* add_SP_Noise(U_IntMatrix* X, DatasetParam& DP, bool Out_Stream);
Matrix* add_SP_Noise(Matrix* X, DatasetParam& DP, bool Out_Stream);
U_IntMatrix* add_Gaussian_Noise(U_IntMatrix* X, double Mean, double StdDev, Arguments& Arg, DatasetParam& DP);
Matrix* add_Gaussian_Noise(Matrix* X,  double Mean, double StdDev, Arguments& Arg, DatasetParam& DP);
U_IntMatrix* AddGaussianNoise_Opencv(U_IntMatrix* X, int Imgdim, double Mean , double StdDev, bool displayNoisyImgs);
U_IntMatrix* Normalize(U_IntMatrix* X, DatasetParam& DP);
Matrix* DeNormalize(Matrix* X, DatasetParam& DP);
U_IntMatrix* Face_Detection(U_IntMatrix* X, int LargeDim, int ImageDim, bool Display_Data);
U_IntMatrix* Eyes_Detection(U_IntMatrix* X, int LargeDim, int ImageDim, bool Display_Data);
U_IntMatrix* Nose_Detection(U_IntMatrix* X, int LargeDim, int ImageDim, bool Display_Data);
U_IntMatrix* Mouth_Detection(U_IntMatrix* X, int LargeDim, int ImageDim, bool Display_Data);
U_IntMatrix* FaceCropper(string ImagePath, int ImgDim);
U_IntMatrix* ReadImageFolder(string FolderPath, int numOfImages, int ImgDim, Choice Crop);
#endif // DATASET_H_INCLUDED

/*Get_TrainSet(Arg, DP);
	U_IntMatrix* X1 = Arg.X;
	U_IntMatrix* Y1 = X1;
	DP.curFile++;
	

	Get_TrainSet(Arg, DP);
	U_IntMatrix* X2 = Arg.X;
	U_IntMatrix* Y2 = X2;
	DP.curFile++;

	Get_TrainSet(Arg, DP);
	U_IntMatrix* X3 = Arg.X;
	U_IntMatrix* Y3 = X3;
	DP.curFile++;

	Get_TrainSet(Arg, DP);
	U_IntMatrix* X4 = Arg.X;
	U_IntMatrix* Y4 = X4;
	DP.curFile++;

	Get_TrainSet(Arg, DP);
	U_IntMatrix* X5 = Arg.X;
	U_IntMatrix* Y5 = X5;
	DP.curFile++;

	Get_TrainSet(Arg, DP);
	U_IntMatrix* X6 = Arg.X;
	U_IntMatrix* Y6 = X6;
	DP.curFile++;

	Get_TrainSet(Arg, DP);
	U_IntMatrix* X7 = Arg.X;
	U_IntMatrix* Y7 = X7;
	DP.curFile++;

	Get_TrainSet(Arg, DP);
	U_IntMatrix* X8 = Arg.X;
	U_IntMatrix* Y8 = X8;
	DP.curFile++;

	U_IntMatrix* XX = new U_IntMatrix(10000, 80000);

	for (int i = 0; i < 10000; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			XX->access(i, j + 0) = X1->access(i, j);
		}
	}
	cout << "X1 is done" << endl;

	for (int i = 0; i < 10000; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			XX->access(i, j + 10000) = X2->access(i, j);
		}
	}
	cout << "X2 is done" << endl;


	for (int i = 0; i < 10000; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			XX->access(i, j + 20000) = X3->access(i, j);
		}
	}
	cout << "X3 is done" << endl;

	for (int i = 0; i < 10000; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			XX->access(i, j + 30000) = X4->access(i, j);
		}
	}
	cout << "X4 is done" << endl;

	for (int i = 0; i < 10000; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			XX->access(i, j + 40000) = X4->access(i, j);
		}
	}
	cout << "X5 is done" << endl;

	for (int i = 0; i < 10000; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			XX->access(i, j + 50000) = X4->access(i, j);
		}
	}
	cout << "X6 is done" << endl;

	for (int i = 0; i < 10000; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			XX->access(i, j + 60000) = X4->access(i, j);
		}
	}
	cout << "X7 is done" << endl;

	for (int i = 0; i < 10000; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			XX->access(i, j + 70000) = X4->access(i, j);
		}
	}
	cout << "X8 is done" << endl;


	Shuffle(XX);

	delete X1;
	delete X2;
	delete X3;
	delete X4;
	delete X5;
	delete X6;
	delete X7;
	delete X8;

	X1 = XX->SubMat(0, 0, 9999, 9999);
	X2 = XX->SubMat(0, 10000, 9999, 19999);
	X3 = XX->SubMat(0, 20000, 9999, 29999);
	X4 = XX->SubMat(0, 30000, 9999, 39999);

	X5 = XX->SubMat(0, 40000, 9999, 49999);
	X6 = XX->SubMat(0, 50000, 9999, 59999);
	X7 = XX->SubMat(0, 60000, 9999, 69999);
	X8 = XX->SubMat(0, 70000, 9999, 79999);

	cout << "X1 " << X1->Rows() << " " << X1->Columns() << endl;
	cout << "X2 " << X2->Rows() << " " << X1->Columns() << endl;
	cout << "X3 " << X3->Rows() << " " << X1->Columns() << endl;
	cout << "X4 " << X4->Rows() << " " << X1->Columns() << endl;
	cout << "X5 " << X5->Rows() << " " << X1->Columns() << endl;
	cout << "X6 " << X6->Rows() << " " << X1->Columns() << endl;
	cout << "X7 " << X7->Rows() << " " << X1->Columns() << endl;
	cout << "X8 " << X8->Rows() << " " << X1->Columns() << endl;




	X1->WriteDataSet("F:\\Final Project\\X_1");
	X2->WriteDataSet("F:\\Final Project\\X_2");
	X3->WriteDataSet("F:\\Final Project\\X_3");
	X4->WriteDataSet("F:\\Final Project\\X_4");
	X5->WriteDataSet("F:\\Final Project\\X_5");
	X6->WriteDataSet("F:\\Final Project\\X_6");
	X7->WriteDataSet("F:\\Final Project\\X_7");
	X8->WriteDataSet("F:\\Final Project\\X_8");

	exit(0);

*/
