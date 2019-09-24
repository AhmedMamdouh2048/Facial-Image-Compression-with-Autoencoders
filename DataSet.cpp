#include "DataSet.h"
//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////  GET DATASET FROM HARD DISK   ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void Get_TrainSet(Arguments& Arg, DatasetParam& DP)
{
	if (DP.Get_NewData)
	{
		Arg.X = new U_IntMatrix(DP.ImageSize * (DP.Resize_Fact* DP.Resize_Fact), DP.Train_Examples);
		Arg.Y = Arg.X;
		ifstream pixels(DP.TextData_dir);
		int NewImgDim = DP.ImageDim * DP.Resize_Fact;
		int select = 1.0 / DP.Resize_Fact;
		for (uint32_t i = 0; i < DP.Train_Examples; i++)
		{
			int r1 = 0; int c1 = 0; int r = 0; int c = 0;   //r1,c1 for original image, r,c for resized one
			for (uint32_t j = 0; j < DP.ImageSize; j++)
			{
				int val;
				char ch;
				pixels >> val;
				if (j % DP.ImageDim == 0 && j != 0)
				{
					r1++;
					c1 = 0;
				}
				else
				{
					c1++;
				}

				if (j % select == 0 && r1 % select == 0)
				{
					Arg.X->access(r * NewImgDim + c, i) = val;        //resizing X to 100*100 image
					c++;
					if (c % NewImgDim == 0)
					{
						c = 0;
						r++;
					}
				}
				if (j != DP.ImageSize - 1)
					pixels >> ch;           // to get rid of ','
			}
		}
		Arg.X->WriteDataSet(DP.X_dir[DP.curFile]);
	}
	else
	{

		matrix<unsigned char>* X_Read = new matrix<unsigned char>(DP.Train_Examples, DP.ImageSize * (DP.Resize_Fact* DP.Resize_Fact));
		X_Read->Read(DP.X_dir[DP.curFile]);
		matrix<unsigned char>* temp = X_Read;
		X_Read = X_Read->TRANSPOSE();
		delete temp;


		Arg.X = new U_IntMatrix(DP.ImageSize * (DP.Resize_Fact* DP.Resize_Fact), DP.Train_Examples);
		for (int i = 0; i<DP.ImageSize * (DP.Resize_Fact* DP.Resize_Fact); i++)
			for (int j = 0; j<DP.Train_Examples; j++)
			{
				Arg.X->access(i, j) = X_Read->access(i, j);
			}
		Arg.Y = Arg.X;
		delete X_Read;

		if (DP.ReScale)
		{
			U_IntMatrix* X_mul_2 = Arg.X->mul(2);
			U_IntMatrix* temp1 = Arg.X;
			Arg.X = X_mul_2->sub(255);
			Arg.Y = Arg.X;
			delete temp1;
			delete X_mul_2;
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Get_TestSet(Arguments& Arg, DatasetParam& DP)
{
	if (DP.Get_NewData)
	{
		Arg.X_test = new U_IntMatrix(DP.ImageSize * (DP.Resize_Fact* DP.Resize_Fact), DP.Test_Examples);
		Arg.Y_test = Arg.X_test;
		ifstream pixels(DP.TextData_dir);
		int NewImgDim = DP.ImageDim * DP.Resize_Fact;
		int select = 1.0 / DP.Resize_Fact;
		for (uint32_t i = 0; i < DP.Test_Examples; i++)
		{
			int r1 = 0; int c1 = 0; int r = 0; int c = 0;   //r1,c1 for original image, r,c for resized one
			for (uint32_t j = 0; j < DP.ImageSize; j++)
			{
				int val;
				char ch;
				pixels >> val;
				if (j % DP.ImageDim == 0 && j != 0)
				{
					r1++;
					c1 = 0;
				}
				else
				{
					c1++;
				}

				if (j % select == 0 && r1 % select == 0)
				{
					Arg.X_test->access(r * NewImgDim + c, i) = val;        //resizing X to 100*100 image
					c++;
					if (c % NewImgDim == 0)
					{
						c = 0;
						r++;
					}
				}
				if (j != DP.ImageSize - 1)
					pixels >> ch;           // to get rid of ','
			}
		}
		Arg.X_test->WriteDataSet(DP.Xtest_dir);
	}
	else
	{
		matrix<unsigned char>* X_Read = new matrix<unsigned char>(DP.Train_Examples, DP.ImageSize * (DP.Resize_Fact* DP.Resize_Fact));
		X_Read->Read(DP.Xtest_dir);
		matrix<unsigned char>* temp = X_Read;
		X_Read = X_Read->TRANSPOSE();
		delete temp;


		Arg.X_test = new U_IntMatrix(DP.ImageSize * (DP.Resize_Fact* DP.Resize_Fact), DP.Test_Examples);
		for (int i = 0; i<DP.ImageSize * (DP.Resize_Fact* DP.Resize_Fact); i++)
			for (int j = 0; j<DP.Test_Examples; j++)
			{
				Arg.X_test->access(i, j) = X_Read->access(i, j);
			}
		Arg.Y_test = Arg.X_test;
		delete X_Read;

		if (DP.ReScale)
		{
			U_IntMatrix* X_mul_2 = Arg.X_test->mul(2);
			U_IntMatrix* temp1 = Arg.X_test;
			Arg.X_test = X_mul_2->sub(255);
			Arg.Y_test = Arg.X_test;
			delete temp1;
			delete X_mul_2;
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Shuffle(Matrix* X)
{
	for (int i = 0; i < X->Columns(); i++)
	{
		int s = rand() % X->Columns();
		SWAP(X, i, s);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Shuffle(U_IntMatrix* X)
{
	for (int i = 0; i < X->Columns(); i++)
	{
		int s = rand() % X->Columns();
		SWAP(X, i, s);
	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SWAP(Matrix* MAT, int i, int k)
{
	Matrix* temp = new Matrix(MAT->Rows(), 1);
	for (int j = 0; j < MAT->Rows(); j++)
	{
		temp->access(j, 0) = MAT->access(j, i);
		MAT->access(j, i) = MAT->access(j, k);
		MAT->access(j, k) = temp->access(j, 0);
	}
	delete temp;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SWAP(U_IntMatrix* MAT, int i, int k)
{
	U_IntMatrix* temp = new U_IntMatrix(MAT->Rows(), 1);
	for (int j = 0; j < MAT->Rows(); j++)
	{
		temp->access(j, 0) = MAT->access(j, i);
		MAT->access(j, i) = MAT->access(j, k);
		MAT->access(j, k) = temp->access(j, 0);
	}
	delete temp;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DisplayData(Arguments& Arg, DatasetParam& DP)
{
	if (DP.Display_Data)
	{
		U_IntMatrix* src = new U_IntMatrix(DP.ImageDim * DP.Resize_Fact, DP.ImageDim * DP.Resize_Fact);
		for (int k = 0; k < Arg.X_disp->Columns(); k++)
		{
			for (int n = 0; n < DP.ImageDim * DP.Resize_Fact; n++)
				for (int m = 0; m < DP.ImageDim * DP.Resize_Fact; m++)
				{
					src->access(n, m) = Arg.X_disp->access(n * DP.ImageDim * DP.Resize_Fact + m, k);
				}
			visualize(ConvertMat_U(src, UC_F, NO));
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////  Gaussian Blur   ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* pad(Matrix* img, int p, float value)
{
	if (img->Rows() != img->Columns())
		cout << "this is not square matrix" << endl;

	int n = img->Rows();
	int m = n + 2 * p;

	Matrix* newImg = new Matrix(m, m);

	for (int i = 0; i < m; i++)
		for (int j = 0; j < m; j++)
		{
			if (i < (m - n - p) || j<(m - n - p) || i>(p + n - 1) || j >(p + n - 1))
				newImg->access(i, j) = 0;
			else
				newImg->access(i, j) = img->access(i - p, j - p);
		}

	return newImg;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* FilterToMatrix(Matrix* filter, int nh, int nw, int s)
{
	int f = filter->Rows();                 // Filter size
	int n = nh * nw;                        // 2D filter row size
	int p = (nh - f + 1) * (nw - f + 1);    // 2D filter column size
	Matrix* Result = new Matrix(p, n, 0);

	int count3 = 0;                         // Counter for the number of shifts from the diagonal
	for (int i = 0; i < p; i++)
	{
		int count1 = f * f;                 // Counter for the number of elements in the filter
		int count2 = 0;                     // Counter for the number of elements in a single row in the filter
		int ii = 0;                         // Row index for the filter
		int jj = 0;                         // Column index for the filter
		for (int j = i + count3; j < n; j++)
		{
			if (i != 0 && j == i + count3 && i % (nh - f + 1) == 0)
			{
				// Shift by f-1 for every down movement of the filter
				j += f - 1;
				count3 += f - 1;
			}
			if (count2 != 0 && count2 % f == 0)
			{
				// Shift by s for every right movement of the filter
				j += nh - f;
				ii++;
				count2 = 0;
			}

			Result->access(i, j) = filter->access(ii, jj);

			count2++;

			// If all elements of the filter are placed in the row
			count1--;
			if (count1 == 0)
				break;

			// If the row of the filter is placed
			jj++;
			if (jj == f)
				jj = 0;
		}

	}
	return Result;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* to_SquareMat(Matrix* X)
{
	int dim = sqrt(X->Rows());
	Matrix* X_2D = new Matrix(dim, dim);
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
		{
			X_2D->access(i, j) = X->access(i*dim + j, 0);
		}
	return X_2D;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* convolve(Matrix* Aprev, Matrix* Filter, int s)
{

	Matrix* convTemp = nullptr;
	Matrix* Matptr1 = nullptr;
	Matrix* filter = nullptr;

	int nh = Aprev->Rows();
	int nw = Aprev->Columns();

	filter = FilterToMatrix(Filter, nh, nw, s);
	Matptr1 = filter->dot_T(Aprev);
	delete filter;
	convTemp = to_SquareMat(Matptr1);
	delete Matptr1;

	return convTemp;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* gaussianBlur(Matrix* img, int filterSize, float sigma)
{
	int p = (filterSize - 1) / 2;

	Matrix* paddedImg = pad(img, p, 0);

	Matrix* filter = gausianFilter(filterSize, filterSize, sigma);

	Matrix* newImg = convolve(paddedImg, filter, 1);

	delete filter;
	delete paddedImg;
	return newImg;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* gausianFilter(int x, int y, float sigma)
{
	Matrix* GaussianFilter = new Matrix(x, y);
	float sum = 0.0;
	int xx = (x - 1) / 2;
	int yy = (y - 1) / 2;

	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			int ii = (i - xx);
			int jj = (j - yy);
			float r = sqrt(ii*ii + jj * jj);
			GaussianFilter->access(i, j) = exp(-(r*r) / (2 * sigma*sigma)) / (2 * 3.14159265358979323846*sigma*sigma);
			sum += GaussianFilter->access(i, j);
		}
	}

	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			GaussianFilter->access(i, j) = GaussianFilter->access(i, j) / sum;
		}
	}

	return GaussianFilter;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////  Visualization   //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void visualize(Matrix* img)
{
	Mat mat1 = Mat::ones(img->Rows(), img->Columns(), CV_32FC1);
	for (int i = 0; i < img->Rows(); i++)
		for (int j = 0; j < img->Columns(); j++)
		{
			mat1.at<float>(i, j) = img->access(i, j);
		}

	imshow("test", mat1);
	waitKey(0);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void visualize(Matrix* img1, Matrix* img2)
{
	Mat mat1 = Mat::ones(img1->Rows(), img1->Columns() + img2->Columns(), CV_32FC1);
	for (int i = 0; i < img1->Rows(); i++)
	{
		for (int j = 0; j < img1->Columns(); j++)
		{
			mat1.at<float>(i, j) = img1->access(i, j);
		}
		int count = 0;
		for (int j = img1->Columns(); j < img1->Columns() + img2->Columns(); j++)
		{
			mat1.at<float>(i, j) = img2->access(i, count);
			count++;
		}
	}

	
	imshow("test", mat1);
	imwrite("F:\\TEST_IMAAAAGES\\1.jpg", mat1);
	//imwrite("F:\\TEST_IMAAAAGES\\1.JPG", mat1);
	//imwrite("F:\\TEST_IMAAAAGES\\", mat1);
	waitKey(0);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void to_JPG(U_IntMatrix* X, string PATH)
{
	U_IntMatrix* img = new U_IntMatrix(100, 100);
	cout << X->Columns() << endl;
	for (int k = 0; k < X->Columns(); k++)
	{
		for (int n = 0; n < 100; n++)
			for (int m = 0; m < 100; m++)
			{
				img->access(n, m) = X->access(n * 100 + m, k);
			}
		Mat mat1 = Mat::ones(img->Rows(), img->Columns(), CV_32FC1);
		for (int i = 0; i < img->Rows(); i++)
			for (int j = 0; j < img->Columns(); j++)
			{
				mat1.at<float>(i, j) = img->access(i, j);
			}
		string s = PATH;
		s.append(std::to_string(k));
		s = s.append(".jpg");
		imwrite(s, mat1);
	}
	delete img;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////  Face feature extracting and editing   ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* add_SP_Noise(U_IntMatrix* X, DatasetParam& DP, bool Out_Stream)
{
	U_IntMatrix* noisyX = new U_IntMatrix(X->Rows(), X->Columns());
	for (int i = 0; i < X->Rows(); i++)
	{
		for (int j = 0; j < X->Columns(); j++)
		{
			noisyX->access(i, j) = X->access(i, j);
		}
	}
	matrix<uint32_t>* indices = new matrix<uint32_t>(DP.Noise_Fact* X->Rows(), 1, Random);
	for (int i = 0; i < indices->Rows(); i++)
	{
		for (int j = 0; j < X->Columns(); j++)
		{
			int k = int(indices->access(i, 0)) % X->Rows();
			if (i < indices->Rows() / 2)
				noisyX->access(k, j) = 1;
			else
				noisyX->access(k, j) = 0;
		}
	}
	delete indices;

	if (Out_Stream)
	{
		ofstream file(DP.Noisy_dir);
		for (int i = 0; i < DP.Train_Examples; i++)
		{
			for (int j = 0; j < X->Rows(); j++)
			{
				file << noisyX->access(j, i);
				if (j != DP.Train_Examples - 1)
					file << ",";
			}
			file << endl;
		}
	}
	return noisyX;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* add_SP_Noise(Matrix* X, DatasetParam& DP, bool Out_Stream)
{
	Matrix* noisyX = new Matrix(X->Rows(), X->Columns());
	for (int i = 0; i < X->Rows(); i++)
	{
		for (int j = 0; j < X->Columns(); j++)
		{
			noisyX->access(i, j) = X->access(i, j);
		}
	}
	Matrix* indices = new Matrix(DP.Noise_Fact* X->Rows(), 1, Random);
	for (int i = 0; i < indices->Rows(); i++)
	{
		for (int j = 0; j < X->Columns(); j++)
		{
			int k = int(indices->access(i, 0)) % X->Rows();
			if (i < indices->Rows() / 2)
				noisyX->access(k, j) = 1;
			else
				noisyX->access(k, j) = 0;
		}
	}
	delete indices;

	if (Out_Stream)
	{
		ofstream file(DP.Noisy_dir);
		for (int i = 0; i < DP.Train_Examples; i++)
		{
			for (int j = 0; j < X->Rows(); j++)
			{
				file << noisyX->access(j, i);
				if (j != DP.Train_Examples - 1)
					file << ",";
			}
			file << endl;
		}
	}
	return noisyX;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* add_Gaussian_Noise(Matrix* X, DatasetParam& DP)
{
	Matrix* noisyX = new Matrix(X->Rows(), X->Columns());
	for (int k = 0; k < X->Columns(); k++)
	{
		Matrix* noise = new Matrix(DP.ImageDim * DP.Resize_Fact, DP.ImageDim * DP.Resize_Fact, NormalDist, 0, DP.Noise_Mean, DP.Noise_Var);
		for (int n = 0; n < DP.ImageDim * DP.Resize_Fact; n++)
			for (int m = 0; m < DP.ImageDim * DP.Resize_Fact; m++)
			{
				float newPixel = (noise->access(n, m) + X->access(n * DP.ImageDim * DP.Resize_Fact + m, k) * 255.0) / 255.0;
				if (newPixel > 1.0)
					newPixel = 1.0;
				else if (newPixel < 0)
					newPixel = 0;
				noisyX->access(n * DP.ImageDim * DP.Resize_Fact + m, k) = newPixel;
			}
		delete noise;
	}
	return noisyX;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* add_Gaussian_Noise(U_IntMatrix* X, DatasetParam& DP)
{
	Matrix* float_X = ConvertMat_U(X, UC_F, NO);
	U_IntMatrix* noisy_X = ConvertMat_U(add_Gaussian_Noise(float_X, DP), F_UC, YES);
	delete float_X;
	return noisy_X;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* add_Noise(Matrix* X, NoiseType NT, DatasetParam& DP)
{
	if (NT == SP)
	{
		return add_SP_Noise(X, DP, "");
	}
	else if (NT == Gauss)
	{
		return add_Gaussian_Noise(X, DP);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* add_Noise(U_IntMatrix* X, NoiseType NT, DatasetParam& DP)
{
	if (NT == SP)
	{
		return add_SP_Noise(X, DP, "");
	}
	else if (NT == Gauss)
	{
		return add_Gaussian_Noise(X, DP);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*U_IntMatrix* AddGaussianNoise_Opencv(U_IntMatrix* X,int Imgdim, double Mean, double StdDev, bool displayNoisyImgs)
{
int Rows = X->Rows();
int Columns = X->Columns();
Mat mGaussian_noise = Mat(Imgdim, Imgdim, CV_8U);
randn(mGaussian_noise, Scalar::all(Mean), Scalar::all(StdDev));
U_IntMatrix* noisyX = new U_IntMatrix(Rows, Columns);
for (int i = 0; i < Columns; i++)
{
int r = 0; int c = 0;
for (int j = 0; j < Rows; j++)
{
noisyX->access(j, i) = X->access(j, i) + mGaussian_noise.at<uint8_t>(r, c);
if (noisyX->access(j, i) > 255)
noisyX->access(j, i) = 255;
c++;
if (c == Imgdim)
{
c = 0;
r++;
}
}
}
if (displayNoisyImgs)
{

for (int k = 0; k < noisyX->Columns(); k++)
{
U_IntMatrix* src = new U_IntMatrix(Imgdim, Imgdim);
for (int n = 0; n < Imgdim; n++)
for (int m = 0; m < Imgdim; m++)
{
src->access(n, m) = noisyX->access(n * Imgdim + m, k);

}
visualize(ConvertMat_U(src, UC_F));
}

}
return noisyX;

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* Face_Detection(U_IntMatrix* X, int LargeDim, int ImgDim, bool Display_Data)
{
U_IntMatrix* Faces = new U_IntMatrix(ImgDim*ImgDim , X->Columns());
CascadeClassifier face_cascade;
face_cascade.load("C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml");
vector<Rect> faces;
Mat matImage = Mat::ones(LargeDim, LargeDim, CV_8U);
Mat matCropped = Mat::ones(LargeDim, LargeDim, CV_8U);
Mat matFace = Mat::ones(ImgDim, ImgDim, CV_8U);

for (int k = 0; k < X->Columns(); k++)
{
for (int n = 0; n < LargeDim; n++)
for (int m = 0; m < LargeDim; m++)
{
matImage.at<uint8_t>(n, m) = X->access(n * LargeDim + m, k);
}
face_cascade.detectMultiScale(matImage, faces, 1.05, 3, 0 , Size(30, 30)); //Face is detected
for (size_t i = 0; i < faces.size(); i++)
{
Rect r = faces[i];
matCropped = matImage(r);
resize(matCropped, matFace, Size(ImgDim, ImgDim), 0, 0, 1);
}

for (int n = 0; n < ImgDim; n++)
for (int m = 0; m < ImgDim; m++)
{
Faces->access(n * ImgDim + m, k) = matFace.at<uint8_t>(n, m);
}
}

if (Display_Data)
{
U_IntMatrix* src = new U_IntMatrix(ImgDim, ImgDim);
for (int k = 0; k < Faces->Columns(); k++)
{
for (int n = 0; n < ImgDim; n++)
for (int m = 0; m < ImgDim; m++)
{
src->access(n, m) = Faces->access(n * ImgDim + m, k);

}
visualize(ConvertMat_U(src,UC_F));
}
}

return Faces;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* Eyes_Detection(U_IntMatrix* X, int LargeDim, int ImgDim, bool Display_Data)
{
U_IntMatrix* Eyes = new U_IntMatrix(ImgDim*ImgDim, X->Columns());
CascadeClassifier Eyes_cascade;
Eyes_cascade.load("C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\eyepair_big.xml");
vector<Rect> eyes;
Mat matImage = Mat::ones(LargeDim, LargeDim, CV_8U);
Mat matCropped = Mat::ones(LargeDim, LargeDim, CV_8U);
Mat matEyes = Mat::ones(ImgDim, ImgDim, CV_8U);

for (int k = 0; k < X->Columns(); k++)
{
for (int n = 0; n < LargeDim; n++)
for (int m = 0; m < LargeDim; m++)
{
matImage.at<uint8_t>(n, m) = X->access(n * LargeDim + m, k);
}
Eyes_cascade.detectMultiScale(matImage, eyes, 1.05, 3, 0, Size(5, 5)); //Eyes is detected
for (size_t i = 0; i < eyes.size(); i++)
{
Rect r = eyes[i];
matCropped = matImage(r);
resize(matCropped, matEyes, Size(ImgDim, ImgDim), 0, 0, 1);
}

for (int n = 0; n < ImgDim; n++)
for (int m = 0; m < ImgDim; m++)
{
Eyes->access(n * ImgDim + m, k) = matEyes.at<uint8_t>(n, m);
}
}

if (Display_Data)
{
U_IntMatrix* src = new U_IntMatrix(ImgDim, ImgDim);
for (int k = 0; k < Eyes->Columns(); k++)
{
for (int n = 0; n < ImgDim; n++)
for (int m = 0; m < ImgDim; m++)
{
src->access(n, m) = Eyes->access(n * ImgDim + m, k);

}
visualize(ConvertMat_U(src, UC_F));
}
}

return Eyes;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* Nose_Detection(U_IntMatrix* X, int LargeDim, int ImgDim, bool Display_Data)
{
U_IntMatrix* Noses = new U_IntMatrix(ImgDim*ImgDim, X->Columns());
CascadeClassifier nose_cascade;
nose_cascade.load("C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\nose.xml");
vector<Rect> noses;
Mat matImage = Mat::ones(LargeDim, LargeDim, CV_8U);
Mat matCropped = Mat::ones(LargeDim, LargeDim, CV_8U);
Mat matNose = Mat::ones(ImgDim, ImgDim, CV_8U);

for (int k = 0; k < X->Columns(); k++)
{
for (int n = 0; n < LargeDim; n++)
for (int m = 0; m < LargeDim; m++)
{
matImage.at<uint8_t>(n, m) = X->access(n * LargeDim + m, k) ;
}
nose_cascade.detectMultiScale(matImage, noses, 1.05, 3, 0, Size(5, 5), Size(30, 30)); //nose is detected
for (size_t i = 0; i < noses.size(); i++)
{
Rect r = noses[i];
matCropped = matImage(r);
resize(matCropped, matNose, Size(ImgDim, ImgDim), 0, 0, 1);
}

for (int n = 0; n < ImgDim; n++)
for (int m = 0; m < ImgDim; m++)
{
Noses->access(n * ImgDim + m, k) = matNose.at<uint8_t>(n, m);
}
}

if (Display_Data)
{
U_IntMatrix* src = new U_IntMatrix(ImgDim, ImgDim);
for (int k = 0; k < Noses->Columns(); k++)
{
for (int n = 0; n < ImgDim; n++)
for (int m = 0; m < ImgDim; m++)
{
src->access(n, m) = Noses->access(n * ImgDim + m, k);
}
visualize(ConvertMat_U(src, UC_F));
}
}

return Noses;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* Mouth_Detection(U_IntMatrix* X, int LargeDim, int ImgDim, bool Display_Data)
{
U_IntMatrix* Mouths = new U_IntMatrix(ImgDim*ImgDim, X->Columns());
CascadeClassifier mouth_cascade;
mouth_cascade.load("C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\mouth.xml");
vector<Rect> mouths;
Mat matImage = Mat::ones(LargeDim, LargeDim, CV_8U);
Mat matCropped = Mat::ones(LargeDim, LargeDim, CV_8U);
Mat matMouth = Mat::ones(ImgDim, ImgDim, CV_8U);

for (int k = 0; k < X->Columns(); k++)
{
for (int n = 0; n < LargeDim; n++)
for (int m = 0; m < LargeDim; m++)
{
matImage.at<uint8_t>(n, m) = X->access(n * LargeDim + m, k);
}
mouth_cascade.detectMultiScale(matImage, mouths, 1.05, 3, 0, Size(15, 15), Size(25, 25)); //mouth is detected
for (size_t i = 0; i < mouths.size(); i++)
{
Rect r = mouths[i];
matCropped = matImage(r);
resize(matCropped, matMouth, Size(ImgDim, ImgDim), 0, 0, 1);
}

for (int n = 0; n < ImgDim; n++)
for (int m = 0; m < ImgDim; m++)
{
Mouths->access(n * ImgDim + m, k) = matMouth.at<uint8_t>(n, m);
}
}

if (Display_Data)
{
U_IntMatrix* src = new U_IntMatrix(ImgDim, ImgDim);
for (int k = 0; k < Mouths->Columns(); k++)
{
for (int n = 0; n < ImgDim; n++)
for (int m = 0; m < ImgDim; m++)
{
src->access(n, m) = Mouths->access(n * ImgDim + m, k);

}
visualize(ConvertMat_U(src, UC_F));
}
}

return Mouths;
}
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* Normalize(U_IntMatrix* X, DatasetParam& DP)
{
	DeNorm* X_DeNorm = &(DP.X_DeNorm[DP.curFile]);

	Matrix* X_Float = ConvertMat_U(X, UC_F, NO);
	Matrix* X_SumCol = X_Float->SUM("column");
	Matrix* X_Mean = X_SumCol->div((X_Float->Columns()));
	
	Matrix* X_Mue = X_Float->sub(X_Mean);
	Matrix* X_Mue_SQUARE = X_Mue->SQUARE();
	Matrix* X_Mue_SQUARE_SumCol = X_Mue_SQUARE->SUM("column");
	Matrix* X_Var = X_Mue_SQUARE_SumCol->div(X_Float->Columns());
	
	Matrix* X_Var_Eps = X_Var->add(1e-7);
	Matrix* X_Var_Eps_SQRT = X_Var_Eps->SQRT();
	Matrix* X_Telda = X_Mue->div(X_Var_Eps_SQRT);
	
	Matrix* X_Telda_Min = X_Telda->sub(X_Telda->MinElement());
	Matrix* X_Normalized = X_Telda_Min->div(X_Telda->MaxElement() - X_Telda->MinElement());
	
	if (DP.Fill_DeNorm)
	{
		DP.X_DeNorm[DP.curFile].mean = X_Mean;
		X_DeNorm->var_sqrt = X_Var_Eps_SQRT;
		X_DeNorm->max = X_Telda->MaxElement();
		X_DeNorm->min = X_Telda->MinElement();
		if (DP.curFile == DP.numFiles - 1)
			DP.Fill_DeNorm = false;
	}
	else
	{
		delete X_Mean;
		delete X_Var_Eps_SQRT;
	}

	delete X_SumCol;
	delete X_Mue_SQUARE;
	delete X_Mue_SQUARE_SumCol;
	delete X_Float;
	delete X_Mue;
	delete X_Var_Eps;
	delete X_Telda;
	delete X_Telda_Min;


	//X_Telda = (X_Telda_max - X_Telda_min) * X_Normlized + X_Telda_min;
	//X_float = X_Telda * X_Var_Eps_SQRT + X_mean

	return ConvertMat_U(X_Normalized, F_UC, YES);
}

Matrix* DeNormalize(Matrix* X_Norm, DatasetParam& DP)
{
	DeNorm* X_DeNorm = &(DP.X_DeNorm[DP.curFile]);

	Matrix* temp = X_Norm->mul(X_DeNorm->max - X_DeNorm->min);
	Matrix* X_Telda = temp->add(X_DeNorm->min);
	delete temp;

	temp = X_Telda->mul(X_DeNorm->var_sqrt);
	Matrix* X_float = temp->add(X_DeNorm->mean);
	delete temp;
	delete X_Telda;

	return X_float;
}

U_IntMatrix* FaceCropper(string ImagePath, int ImgDim)
{
	U_IntMatrix* CroppedFace = new U_IntMatrix(ImgDim, ImgDim);
	//CascadeClassifier face_cascade;
	//face_cascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml");
	//vector<Rect> faces;
	Mat matImage = imread(ImagePath, IMREAD_GRAYSCALE);
	//face_cascade.detectMultiScale(matImage, faces, 1.05, 3, 0, Size(80,80)); //Face is detected
	Mat matFace = Mat::ones(ImgDim, ImgDim, CV_8U);
	for (size_t i = 0; i < 1/*faces.size()*/; i++)
	{
		//Rect r = faces[i];
		//Mat matCropped = matImage(r);
		resize(matImage, matFace, Size(ImgDim, ImgDim), 0, 0, 1);
	}

	for (int n = 0; n < ImgDim; n++)
		for (int m = 0; m < ImgDim; m++)
		{
			CroppedFace->access(n, m) = matFace.at<uint8_t>(n, m);
		}

	return CroppedFace;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* ReadImageFolder(string FolderPath, int numOfImages, int ImgDim, Choice Crop)
{
	U_IntMatrix* Faces = new U_IntMatrix(ImgDim*ImgDim, numOfImages);

	const char* FolderPathChar = FolderPath.data();

	DIR* DirPtr = NULL;						// Pointer to directory
	struct dirent* pent = NULL;				// Pointer to an object of dirent which is used to list files out of directory
	DirPtr = opendir(FolderPathChar);

	if (DirPtr == NULL)						// if DirPtr wasn't initialised correctly
	{
		cout << "\nERROR! There is no such directoryx!" << endl;
		_getche();
		exit(1);							// exit the program, using 1 as the status (most common for a failed execution)
	}

	int i = 0;
	while (pent = readdir(DirPtr))			// while there is still something in the directory to list
	{
		
		if (pent == NULL)					// if pent has not been initialised correctly
		{
			cout << "\nERROR! Directory can't be readxx" << endl;
			_getche();
			exit(1);
		}

		string ImageName;
		ImageName.assign(pent->d_name);
		cout << ImageName << endl;

		// Checks if this file is VM or not
		if (ImageName.size() <= 4 || ".jpg" != ImageName.substr(ImageName.size() - 4, 4))
			continue;
		Mat matImage;
		string ImagePath = FolderPath + "\\" + ImageName;
		if (Crop == YES)
		{

			U_IntMatrix* face = FaceCropper(ImagePath, ImgDim);
			for (int n = 0; n < ImgDim; n++)
				for (int m = 0; m < ImgDim; m++)
				{
					Faces->access(n * ImgDim + m, i) = face->access(n, m);
				}
			delete face;
		}
		else if (Crop == NO)
		{
			matImage = imread(ImagePath, IMREAD_GRAYSCALE);
			for (int n = 0; n < ImgDim; n++)
				for (int m = 0; m < ImgDim; m++)
				{
					Faces->access(n * ImgDim + m, i) = matImage.at<uint8_t>(n, m);
				}
		}
		i++;

		if (i == 20000)
			break;
	}

	return Faces;
}