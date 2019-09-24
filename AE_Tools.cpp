#include "AE_Tools.h"
#include "DataSet.h"
#include <conio.h>
using namespace std;

string CharGen(string name, int i)
{
	int temp = i;
	int counter1;   //number of decimal digits in i

	if (temp == 0)
		counter1 = 1;
	else
	{
		for (counter1 = 0; temp != 0; counter1++)
			temp = temp / 10;
	}


	int counter2 = name.size();   //number of chars in name

	string result;
	if (counter2 == 1) { result = "W0"; }
	if (counter2 == 2) { result = "dW0"; }
	if (counter2 == 3) { result = "Sdw0"; }
	if (counter2 == 4) { result = "dACP0"; }
	if (counter2 == 5) { result = "dACP01"; }
	if (counter2 == 6) { result = "dACP012"; }
	if (counter2 == 7) { result = "dACP0123"; }
	if (counter2 == 8) { result = "dACP01234"; }
	if (counter2 == 9) { result = "dACP012345"; }
	if (counter2 == 10) { result = "dACP0123456"; }
	if (counter2 == 11) { result = "dACP01234567"; }
	if (counter2 == 12) { result = "dACP012345678"; }


	for (unsigned int j = 0; j < name.size(); j++) //copy the name into result
		result[j] = name[j];

	int j = counter1 + counter2 - 1;      //copy the number into result
	temp = i;
	do
	{
		result[j] = '0' + (temp % 10);
		temp = temp / 10;
		j--;
	} while (temp != 0);

	return result;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float AccuracyTest(Matrix* Y, Matrix* Y_hat, Arguments& Arg, DatasetParam& DP)
{
	float thrsh = (DP.ReScale) ? Arg.threshold * 2 : Arg.threshold;
	float cost = 0;
	float m = Y_hat->Columns();
	Matrix* diff = Y_hat->sub(Y);
	float result = diff->norm_L1();
	result = result / (Y->Rows()*Y->Columns());

	if (DP.ReScale)
	{
		result = result / 2.0;
	}

	cout << "Avg Errs = " << setw(3) << setiosflags(ios::fixed) << setprecision(3) << result * 100 << '%' << endl;
	Arg.numErrors = 0;
	for (int i = 0; i < diff->Columns(); i++)
	{
		Matrix* MatPtr = diff->SubMat(0, i, -1, i);
		float err = (MatPtr->norm_L1() / diff->Rows()) * 100.0;
		if (err > thrsh)
			Arg.numErrors++;
		delete MatPtr;
	}
	cout << "Errs Higher Than " << int(Arg.threshold) << '%' << " = " << int(Arg.numErrors) << endl;
	delete diff;


	int ImgDim = 100;
	Matrix* src = new Matrix(ImgDim, ImgDim);
	Matrix* dst = new Matrix(ImgDim, ImgDim);
	for (int k = 0; k < Y_hat->Columns(); k++)
	{
		for (int n = 0; n < ImgDim; n++)
			for (int m = 0; m < ImgDim; m++)
			{
				if (DP.ReScale)
				{
					src->access(n, m) = (Y->access(n * ImgDim + m, k) + 1 ) / 2;
					dst->access(n, m) = (Y_hat->access(n * ImgDim + m, k )+ 1 ) / 2;
				}
				else
				{
					src->access(n, m) = Y->access(n * ImgDim + m, k);
					dst->access(n, m) = Y_hat->access(n * ImgDim + m, k);
				}
			}

		if (Arg.TestParameters)
			visualize(src, dst);

	}


	return result;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix** cluster(U_IntMatrix* X_test, Matrix* A, float CompValue)
{
	/*IMPORTANT VARIABLES*/
	int m = A->Columns();				// Number of test examples
	int neurons = A->Rows();			// Number of activations
	Matrix* temp = nullptr;				// Temporary matrix
	int* IsSimilar = new int[m];		// Array of the same number of test examples. The content of all similar examples is the same.
	for (int i = 0; i < m; i++)
	{
		IsSimilar[i] = 0;
	}
	int count = 0;						// The number inside the content of IsSimilar. If this number is the same for 2 or more images, they are similar.
	bool FirstTime = true;				// Indicator to more than 2 similar images
										/*END OF IMPORTANT VARIABLES*/

										/*PERFORMING NORM TO DETERMINE SIMILAR IMAGES*/
	for (int i = 0; i < m - 1; i++)		// The 2 for loops for comparing all examples in bubble sort fashion
	{
		if (IsSimilar[i] != 0)
			continue;

		for (int j = i + 1; j < m; j++)
		{
			if (IsSimilar[j] != 0)
				continue;

			temp = new Matrix(neurons, 1);
			for (int k = 0; k < neurons; k++)
			{
				temp->access(k, 0) = A->access(k, i) - A->access(k, j);
			}
			float distance = CosineSim(A->SubMat(0, i, A->Rows() - 1, i), A->SubMat(0, j, A->Rows() - 1, j));
			cout << distance << endl;
			delete temp;

			if (distance < CompValue)
			{
				if (FirstTime)
				{
					count++;
					FirstTime = false;
					IsSimilar[i] = count;
				}
				IsSimilar[j] = count;
			}
		}

		FirstTime = true;
	}
	/*END OF PERFORMING NORM TO DETERMINE SIMILAR IMAGES*/

	/*GROUPING SIMILAR IMAGES INTO CLUSTERS*/
	// SimilarImages is a group of matrices each holding a cluster of similar images
	U_IntMatrix** SimilarImages = new U_IntMatrix*[count + 1];
	for (int i = 0; i < count + 1; i++)
	{
		if (i == 0)
			cout << "The distinct Images: " << endl;
		else
			cout << "Cluster No." << i << ": " << endl;

		int NumOfCount = 0;
		for (int j = 0; j < m; j++)
		{
			if (IsSimilar[j] == i)
				NumOfCount++;
		}
		SimilarImages[i] = new U_IntMatrix(X_test->Rows(), NumOfCount);
		int ii = 0;
		cout << "Images";
		for (int j = 0; j < m; j++)
		{
			if (IsSimilar[j] == i)
			{
				cout << ' ' << j << ',';
				for (int k = 0; k < X_test->Rows(); k++)
				{
					SimilarImages[i]->access(k, ii) = X_test->access(k, j);
				}
				ii++;
			}
		}
		cout << "\b.\b" << endl;
	}
	/*END OF GROUPING SIMILAR IMAGES INTO CLUSTERS*/

	/*VISUALISING THE CLUSTERED IMAGES*/
	cout << endl << "Visualizing the clusters.." << endl;
	int ImgDim = 100;
	for (int i = 1; i < count + 1; i++)
	{
		cout << "Cluster No." << i << ".." << endl;
		for (int j = 0; j < SimilarImages[i]->Columns(); j++)
		{
			Matrix* DisplayedImage_float = new Matrix(ImgDim, ImgDim);
			U_IntMatrix* DisplayedImage = new U_IntMatrix(ImgDim, ImgDim);

			for (int k = 0; k < ImgDim; k++)
				for (int n = 0; n < ImgDim; n++)
					DisplayedImage->access(k, n) = SimilarImages[i]->access(k * ImgDim + n, j);

			DisplayedImage_float = ConvertMat_U(DisplayedImage, UC_F, YES);
			visualize(DisplayedImage_float);
			delete DisplayedImage_float;
		}
		cout << "Press any key to continue.." << endl;
		_getch();
	}

	cout << "The distinct images: " << endl;
	for (int j = 0; j < SimilarImages[0]->Columns(); j++)
	{
		Matrix* DisplayedImage_float = new Matrix(ImgDim, ImgDim);
		U_IntMatrix* DisplayedImage = new U_IntMatrix(ImgDim, ImgDim);

		for (int k = 0; k < ImgDim; k++)
			for (int n = 0; n < ImgDim; n++)
				DisplayedImage->access(k, n) = SimilarImages[0]->access(k * ImgDim + n, j);

		DisplayedImage_float = ConvertMat_U(DisplayedImage, UC_F, YES);
		visualize(DisplayedImage_float);
		delete DisplayedImage_float;
	}
	cout << "Visualization complete!" << endl;
	/*END OF VISUALISING THE CLUSTERED IMAGES*/

	return SimilarImages;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* DOT(Matrix* X, Matrix* Y)
{
	Matrix* result = new Matrix(X->Rows(), Y->Columns());
	int CORES = thread::hardware_concurrency();
	thread** Threads = new  thread*[CORES];
	Y = Y->TRANSPOSE();
	float** Y_data = Y->ptr();
	float** X_data = X->ptr();
	float** result_data = result->ptr();
	for (int i = 0; i < CORES; i++)
	{
		Threads[i] = new thread(DotPart, i + 1, result, X, Y);
	}

	for (int i = 0; i < CORES; i++)
	{
		Threads[i]->join();
		delete Threads[i];
	}
	delete Threads;

	if (X->Rows() % CORES != 0)
	{
		int numOfRows = X->Rows() % CORES;
		int limit = X->Rows();
		int start = limit - numOfRows;
		for (int i = start; i < limit; i++)
			for (int j = 0; j < Y->Rows(); j++)
				for (int k = 0; k < X->Columns(); k++)
					result_data[i][j] += X_data[i][k] * Y_data[j][k];
	}
	delete Y;
	return result;
}
///////////////////////////////////////////////////////////////////////
void DotPart(int part, Matrix* result, Matrix* X, Matrix* Y)
{
	float** Y_data = Y->ptr();
	float** X_data = X->ptr();
	float** result_data = result->ptr();
	int numOfRows = X->Rows() / thread::hardware_concurrency();
	int limit = part * numOfRows;
	int start = limit - numOfRows;
	for (int i = start; i < limit; i++)
		for (int j = 0; j < Y->Rows(); j++)
			for (int k = 0; k < X->Columns(); k++)
				result_data[i][j] += X_data[i][k] * Y_data[j][k];
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MIX(Matrix*& X, Matrix*& Y, Matrix* X_, Matrix* Y_)
{
	Matrix* XX = new Matrix(X->Rows(), X->Columns() + X_->Columns());
	Matrix* YY = new Matrix(Y->Rows(), Y->Columns() + Y_->Columns());
	int i = 0;
	int j = 0;
	int k = 0;
	int m = 0;

	for (i = 0; i < X->Rows(); i++)
		for (j = 0; j < X->Columns(); j++)
			XX->access(i, j) = X->access(i, j);


	for (i = 0; i < XX->Rows(); i++)
		for (k = j, m = 0; k < XX->Columns(); k++, m++)
			XX->access(i, k) = X_->access(i, m);




	for (i = 0; i < Y->Rows(); i++)
		for (j = 0; j < Y->Columns(); j++)
			YY->access(i, j) = Y->access(i, j);


	for (i = 0; i < YY->Rows(); i++)
		for (k = j, m = 0; k < YY->Columns(); k++, m++)
			YY->access(i, k) = Y_->access(i, m);


	delete X;
	delete Y;
	delete X_;
	delete Y_;
	X = XX;
	Y = YY;
}
//////////////////////////////////////////////////////////////////////////////////////////////
Matrix* ConvertMat_U(U_IntMatrix* src, TypeOfConversion TYPE, Choice DeleteSrc)
{
	Matrix* dst = new Matrix(src->Rows(), src->Columns());
	if (src->Columns() != dst->Columns() || src->Rows() != src->Rows())
	{
		cout << "ERROR in ConvertMat!" << endl;
		_getche();
		exit(0);
	}

	if (TYPE == UC_F)
	{
		for (int i = 0; i < src->Rows(); i++)
		{
			for (int j = 0; j < src->Columns(); j++)
			{
				dst->access(i, j) = src->access(i, j) / 255.0;
			}
		}
	}
	else if (TYPE == UI16_F)
	{
		for (int i = 0; i < src->Rows(); i++)
		{
			for (int j = 0; j < src->Columns(); j++)
			{
				dst->access(i, j) = src->access(i, j) / 65535.0;
			}
		}
	}
	else
	{
		cout << "ERROR in ConvertMat!" << endl;
		_getche();
		exit(0);
	}

	if (DeleteSrc == YES)
	{
		delete src;
	}

	return dst;
}
//////////////////////////////////////////////////////////////////////////////////////////////
U_IntMatrix* ConvertMat_U(Matrix* src, TypeOfConversion TYPE, Choice DeleteSrc)
{
	U_IntMatrix* dst = new U_IntMatrix(src->Rows(), src->Columns());
	if (src->Columns() != dst->Columns() || src->Rows() != src->Rows())
	{
		cout << "ERROR in ConvertMat!" << endl;
		_getche();
		exit(0);
	}

	if (TYPE == F_UC)
	{
		for (int i = 0; i < src->Rows(); i++)
		{
			for (int j = 0; j < src->Columns(); j++)
			{
				dst->access(i, j) = src->access(i, j) * 255.0;
			}
		}
	}
	else if (TYPE == F_UI16)
	{
		for (int i = 0; i < src->Rows(); i++)
		{
			for (int j = 0; j < src->Columns(); j++)
			{
				dst->access(i, j) = src->access(i, j) * 65535.0;

			}
		}
	}
	else
	{
		cout << "ERROR in ConvertMat!" << endl;
		_getche();
		exit(0);
	}

	if (DeleteSrc == YES)
	{
		delete src;
	}

	return dst;
}

//////////////////////////////////////////////////////////////////////////////////////////////
Matrix* ConvertMat_S(IntMatrix* src, TypeOfConversion TYPE, Choice DeleteSrc)
{
	Matrix* dst = new Matrix(src->Rows(), src->Columns());
	if (src->Columns() != dst->Columns() || src->Rows() != src->Rows())
	{
		cout << "ERROR in ConvertMat!" << endl;
		_getche();
		exit(0);
	}

	if (TYPE == C_F)
	{
		for (int i = 0; i < src->Rows(); i++)
		{
			for (int j = 0; j < src->Columns(); j++)
			{
				dst->access(i, j) = src->access(i, j) / 127.0;
			}
		}
	}
	else if (TYPE == I16_F)
	{
		for (int i = 0; i < src->Rows(); i++)
		{
			for (int j = 0; j < src->Columns(); j++)
			{
				dst->access(i, j) = src->access(i, j) / 32767.0;

			}
		}
	}
	else
	{
		cout << "ERROR in ConvertMat!" << endl;
		_getche();
		exit(0);
	}

	if (DeleteSrc == YES)
	{
		delete src;
	}

	return dst;
}
//////////////////////////////////////////////////////////////////////////////////////////////
IntMatrix* ConvertMat_S(Matrix* src, TypeOfConversion TYPE, Choice DeleteSrc)
{
	IntMatrix* dst = new IntMatrix(src->Rows(), src->Columns());
	if (src->Columns() != dst->Columns() || src->Rows() != src->Rows())
	{
		cout << "ERROR in ConvertMat!" << endl;
		_getche();
		exit(0);
	}

	if (TYPE == F_C)
	{
		for (int i = 0; i < src->Rows(); i++)
		{
			for (int j = 0; j < src->Columns(); j++)
			{
				dst->access(i, j) = src->access(i, j) * 127.0;

			}
		}
	}
	else if (TYPE == F_I16)
	{
		for (int i = 0; i < src->Rows(); i++)
		{
			for (int j = 0; j < src->Columns(); j++)
			{
				dst->access(i, j) = src->access(i, j) * 32767.0;
			}
		}
	}
	else
	{
		cout << "ERROR in ConvertMat!" << endl;
		_getche();
		exit(0);
	}

	if (DeleteSrc == YES)
	{
		delete src;
	}

	return dst;
}
//////////////////////////////////////////////////////////////////////////////////////////////
void PrintLayout(Arguments& Arg, DatasetParam& DP)
{
	cout << ">> DataSet Information: " << endl;
	cout << "Size Of Image = " << DP.ImageSize << " Pixels" << endl;
	cout << "Total No. Images In a Training File = " << DP.BIG_FILE << endl;
	cout << "Total No. Training Files = " << DP.numFiles << endl;
	cout << "NO. Training Images = " << DP.Train_Examples << endl;
	cout << "NO. Test Images = " << DP.Test_Examples << endl << endl;



	//------------------------------------------------------------------//
	//---------------------- Print Network Layout ----------------------//
	//------------------------------------------------------------------//
	cout << ">> Training Information: " << endl;
	cout << "Type Of Network: ";
	switch (Arg.NetType)
	{
	case FC: cout << "Fully Connected" << endl; break;
	case LENET1: cout << "LENET1" << endl; break;
	case CUSTOM: cout << "Convolution";
	}
	cout << "Optimization Algorithm: ";
	switch (Arg.optimizer)
	{
	case ADAM: cout << "ADAM" << endl; break;
	case GRADIENT_DESCENT: cout << "Gradient Descent" << endl;
	}
	cout << "Cost Function: ";
	switch (Arg.ErrType)
	{
	case CROSS_ENTROPY: cout << "Cross Entropy" << endl; break;
	case SQAURE_ERROR: cout << "Square Error" << endl;
	}
	cout << "Learining Rate = " << Arg.learingRate << endl;
	cout << "Batch Size = " << Arg.batchSize << endl;
	cout << "No. Layers = " << Arg.numOfLayers << endl;
	for (int i = 0; i < Arg.numOfLayers; i++)
	{
		cout << "Layer " << i + 1 << " = " << Arg.layers[i].neurons << " Neurons" << endl;
	}
	cout << endl;

	cout << ">> Parameters Retrieving and Saving:" << endl;
	cout << "Retrieve Parameters (y/n)? ";
	char ans1 = _getche();
	if (ans1 = 'y')
		Arg.RetrieveParameters = true;
	else
		Arg.RetrieveParameters = false;
	cout << endl;

	cout << "Save Parameters (y/n)? ";
	char ans2 = _getche();
	if (ans2 = 'y')
		Arg.SaveParameters = true;
	else
		Arg.SaveParameters = false;
	cout << endl << endl;
}

float CosineSim(Matrix* Vect1, Matrix* Vect2)
{
	Matrix* VectMul = Vect1->mul(Vect2);
	float DotProd = VectMul->sumall();
	float Vect1_norm = Vect1->norm_L2();
	float Vect2_norm = Vect2->norm_L2();
	float CosineTheta = DotProd / (Vect1_norm * Vect2_norm);
	delete VectMul;
	return CosineTheta;
}


/*float AccuracyTest(Matrix* Y, Matrix* Y_hat, Arguments& Arg, bool Visualize)
{
	float cost = 0;
	float m = Y_hat->Columns();
	Matrix* diff = Y_hat->sub(Y);
	float result = diff->norm_L1();
	result = result / (Y->Rows()*Y->Columns());
	cout << "Average Erros = " << result * 100 << '%' << endl;
	Arg.numErrors = 0;
	for (int i = 0; i < diff->Columns(); i++)
	{
		Matrix* kk = diff->SubMat(0, i, -1, i);
		float err = (kk->norm_L1() / diff->Rows()) * 100.0;
		int ImgDim = 100;
		Matrix* src = new Matrix(ImgDim, ImgDim);
		Matrix* dst = new Matrix(ImgDim, ImgDim);
		if (err > Arg.threshold)
		{
			Arg.numErrors++;
			for (int n = 0; n < ImgDim; n++)
				for (int m = 0; m < ImgDim; m++)
				{
					src->access(n, m) = Y->access(n * ImgDim + m, i);
					dst->access(n, m) = Y_hat->access(n * ImgDim + m, i);
				}
			if (Visualize)
				visualize(src, dst);
		}
			
		delete kk;
	}
	cout << "No. Erros Larger Than " << Arg.threshold << '%' <<" = "<< Arg.numErrors << endl;
	delete diff;
	
	

	return result;*/