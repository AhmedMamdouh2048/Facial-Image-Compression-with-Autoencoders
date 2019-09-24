#include "AutoEncoder.h"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AutoEncoder::init_FC()
{
	/*ARGUMENT LIST*/
	Optimizer optimizer = Arg->optimizer;
	bool batchNorm = Arg->batchNorm;
	bool dropout = Arg->dropout;
	bool dropConnect = Arg->dropConnect;
	int numOfLayers = Arg->numOfLayers;
	layer* layers = Arg->layers;
	/*END OF ARGUMENT LIST*/

	// Temporary pointer
	Matrix*  MatPtr = nullptr;

	// Drop-out initalization
	if (dropout)
	{
		D = new Matrix**[Cores + 1];
		for (int i = 0; i < Cores + 1; i++)
			D[i] = new Matrix*[numOfLayers];
	}

	// Drop-connection initalization
	if (dropConnect)
	{
		D2 = new matrix<bool>***[Cores + 1];
		for (int i = 0; i < Cores + 1; i++)
			D2[i] = new matrix<bool>**[numOfLayers - 2];    //The weights in the last layer are not dropped out
	}

	// Weigths and biases initalization
	Matrix** Mw = new Matrix*[numOfLayers - 1];
	Matrix** Mb = new Matrix*[numOfLayers - 1];
	for (int i = 0; i < numOfLayers - 1; i++)  // L-1 = number of hidden layers + output layer
	{
		if (Arg->tiedWeights && i >= ((numOfLayers - 1) / 2))
			Mw[i] = Mw[numOfLayers - 2 - i]->TRANSPOSE();
		else
		{
			Mw[i] = new Matrix(layers[i + 1].neurons, layers[i].neurons, Random); // Mw[0] holds W1 and so on
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->div(float(RAND_MAX));
			delete MatPtr;

			/*To make the standard deviation of weights = 1 and mean = 0*/
			if (Mw[i]->Rows() != 1 || Mw[i]->Columns() != 1) // Don't calculate if dimensions are 1x1
			{
				float Wmean = Mw[i]->sumall() / (Mw[i]->Rows() * Mw[i]->Columns());
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->sub(Wmean);
				delete MatPtr;

				float Wstd = sqrt((Mw[i]->square()).sumall() / (Mw[i]->Rows() * Mw[i]->Columns()));
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->div(Wstd);
				delete MatPtr;
			}
			/*Normalize the weights with respect to the number of neurons*/
			switch (layers[i + 1].activation)
			{
			case SIGMOID:
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->mul(sqrt(2.0 / layers[i].neurons));
				delete MatPtr;
				break;

			case SOFTMAX:
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->mul(sqrt(2.0 / layers[i].neurons));
				delete MatPtr;
				break;

			case TANH:
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->mul(sqrt(1.0 / layers[i].neurons));
				delete MatPtr;
				break;

			case RELU:
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->mul(sqrt(2.0 / layers[i].neurons));
				delete MatPtr;
				break;

			case LEAKYRELU:
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->mul(sqrt(2.0 / layers[i].neurons));
				delete MatPtr;
				break;

			case SATLINEAR:
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->mul(sqrt(1.0 / layers[i].neurons));
				delete MatPtr;
				break;

			case LINEAR:
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->mul(sqrt(2.0 / layers[i].neurons));
				delete MatPtr;
				break;

			case SATLINEAR2:
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->mul(sqrt(1.0 / layers[i].neurons));
				delete MatPtr;
				break;

			case SATLINEAR3:
				MatPtr = Mw[i];
				Mw[i] = Mw[i]->mul(sqrt(1.0 / layers[i].neurons));
				delete MatPtr;
				break;

			case NONE:
				break;
			}
		}
		Mb[i] = new Matrix(layers[i + 1].neurons, 1, Random);
		MatPtr = Mb[i];
		Mb[i] = Mb[i]->div(float(RAND_MAX));
		delete MatPtr;

		/*To make the standard deviation of biases = 1 and mean = 0*/
		if (Mb[i]->Rows() != 1 || Mb[i]->Columns() != 1)         // Don't calculate if dimensions are 1x1
		{
			float bmean = Mb[i]->sumall() / (Mb[i]->Rows() * Mb[i]->Columns());
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->sub(bmean);
			delete MatPtr;

			float bstd = sqrt((Mb[i]->square()).sumall() / (Mb[i]->Rows() * Mb[i]->Columns()));
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->div(bstd);
			delete MatPtr;
		}

		/*Normalize the biases with respect to the number of neurons*/
		switch (layers[i + 1].activation)
		{
		case SIGMOID:
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->mul(sqrt(2.0 / layers[i].neurons));
			delete MatPtr;
			break;

		case SOFTMAX:
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->mul(sqrt(2.0 / layers[i].neurons));
			delete MatPtr;
			break;

		case TANH:
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->mul(sqrt(1.0 / layers[i].neurons));
			delete MatPtr;
			break;

		case RELU:
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->mul(sqrt(2.0 / layers[i].neurons));
			delete MatPtr;
			break;

		case LEAKYRELU:
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->mul(sqrt(2.0 / layers[i].neurons));
			delete MatPtr;
			break;

		case SATLINEAR:
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->mul(1.0 / layers[i].neurons);
			delete MatPtr;
			break;

		case LINEAR:
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->mul(sqrt(2.0 / layers[i].neurons));
			delete MatPtr;
			break;

		case SATLINEAR2:
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->mul(1.0 / layers[i].neurons);
			delete MatPtr;
			break;

		case SATLINEAR3:
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->mul(1.0 / layers[i].neurons);
			delete MatPtr;
			break;

		case NONE:
			break;
		}

		FC_Parameters.put(CharGen("W", i + 1), Mw[i]);
		FC_Parameters.put(CharGen("b", i + 1), Mb[i]);
	}

	/*BatchNorm Initialization*/
	if (batchNorm)
	{
		// Initialization for batchnorm at training time
		Matrix** g1 = new Matrix*[numOfLayers - 1];   //gamma1
		Matrix** g2 = new Matrix*[numOfLayers - 1];   //gamma2
		for (int ii = 0; ii < numOfLayers - 1; ii++)
		{
			g1[ii] = new Matrix(layers[ii + 1].neurons, 1, 1);
			FC_Parameters.put(CharGen("g1", ii + 1), g1[ii]);
			g2[ii] = new Matrix(layers[ii + 1].neurons, 1);
			FC_Parameters.put(CharGen("g2", ii + 1), g2[ii]);
		}

		//Intialization For batch Norm at test time
		Matrix** running_mean = new Matrix*[numOfLayers - 1];      //mean of z for each layer
		Matrix** running_var = new Matrix*[numOfLayers - 1];       //standard deviation of z for each layer
		for (int ii = 0; ii < numOfLayers - 1; ii++)
		{
			running_mean[ii] = new Matrix(layers[ii + 1].neurons, 1);
			FC_Parameters.put(CharGen("rm", ii + 1), running_mean[ii]);
			running_var[ii] = new Matrix(layers[ii + 1].neurons, 1);
			FC_Parameters.put(CharGen("rv", ii + 1), running_var[ii]);
		}
	}
	/*End Of BatchNorm Initialization*/

	/*ADAM INITIALIZATION*/
	if (optimizer == ADAM)
	{
		Matrix** Msdw = new Matrix*[numOfLayers - 1];
		Matrix** Mvdw = new Matrix*[numOfLayers - 1];
		Matrix** Msdb = new Matrix*[numOfLayers - 1];
		Matrix** Mvdb = new Matrix*[numOfLayers - 1];

		//For batchnorm
		Matrix** sdg1 = new Matrix*[numOfLayers - 1];
		Matrix** vdg1 = new Matrix*[numOfLayers - 1];
		Matrix** sdg2 = new Matrix*[numOfLayers - 1];
		Matrix** vdg2 = new Matrix*[numOfLayers - 1];

		for (int i = 0; i < numOfLayers - 1; i++)   // L-1 = number of hidden layers + output layer
		{
			Msdw[i] = new Matrix(layers[i + 1].neurons, layers[i].neurons);
			FC_ADAM.put(CharGen("Sdw", i + 1), Msdw[i]);

			Mvdw[i] = new Matrix(layers[i + 1].neurons, layers[i].neurons);
			FC_ADAM.put(CharGen("Vdw", i + 1), Mvdw[i]);

			Msdb[i] = new Matrix(layers[i + 1].neurons, 1);
			FC_ADAM.put(CharGen("Sdb", i + 1), Msdb[i]);

			Mvdb[i] = new Matrix(layers[i + 1].neurons, 1);
			FC_ADAM.put(CharGen("Vdb", i + 1), Mvdb[i]);

			if (batchNorm)
			{
				sdg1[i] = new Matrix(layers[i + 1].neurons, 1);
				FC_ADAM.put(CharGen("sg1", i + 1), sdg1[i]);
				vdg1[i] = new Matrix(layers[i + 1].neurons, 1);
				FC_ADAM.put(CharGen("vg1", i + 1), vdg1[i]);

				sdg2[i] = new Matrix(layers[i + 1].neurons, 1);
				FC_ADAM.put(CharGen("sg2", i + 1), sdg2[i]);
				vdg2[i] = new Matrix(layers[i + 1].neurons, 1);
				FC_ADAM.put(CharGen("vg2", i + 1), vdg2[i]);
			}
		}
	}
	/*END OF ADAM INITIALIZATION*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AutoEncoder::train_FC()
{
	/*ARGUMENT LIST*/
	U_IntMatrix* X = Arg->X;
	U_IntMatrix* Y = Arg->Y;
	Matrix* A = (DP->Noisy_Data) ? Arg->A_Noisy : Arg->A;
	Matrix* A_hat = Arg->A;
	int minibatchSize = Arg->batchSize;
	int numOfLayers = Arg->numOfLayers;
	int numOfEpochs = Arg->numOfEpochs;
	layer* layers = Arg->layers;
	float* keep_prob = Arg->keep_prob;
	bool dropout = Arg->dropout;
	bool dropConnect = Arg->dropConnect;
	/*END OF ARGUMENT LIST*/

	/*ESSENTIAL VARIABLE INITIALIZATION*/
	Matrix* cur_X = nullptr;    //Holds current batch examples
	Matrix* cur_Y = nullptr;    //Holds current batch labels
	isLastepoch = false;        //Indicator to last epoch
	momentum = true;            //Enable momentum in Adam
	int t = 0;                  //Counter used for Adam optimizer
	int m;                      //Number of training examples
								/*END OF ESSENTIAL VARIABLE INITIALIZATION*/



    /*BEGINNING OF EPOCHS ITERATIONS*/
	for (int i = 0; i < numOfEpochs; i++)
	{
		std::cout << "Training[" << DP->curFile + 1 << "] ";
		m = (Arg->Stack) ? A->Columns() : X->Columns();
		int numOfMiniBatches = m / minibatchSize;
		int LastBatchSize = m - minibatchSize * numOfMiniBatches;
		int j;
		if (i == numOfEpochs - 1)
			isLastepoch = true;

		if (Arg->SPARSE)
		{
			Matrix* All_X = (Arg->Stack) ? A : ConvertMat_U(X, UC_F, NO);
			FC_Cache[Cores].put("A0", All_X);
			Bottleneck_FeedForward(Cores);
			if (Arg->Stack)
				FC_Cache[Cores].DeleteThenClearTEST();
			else
				FC_Cache[Cores].DeleteThenClear();
		}

		for (j = 0; j < numOfMiniBatches; j++)
		{

			if (dropout)
			{
				for (int k = 0; k < numOfLayers - 2; k++)
					D[Cores][k + 1] = new Matrix(layers[k + 1].neurons, minibatchSize, Bernoulli, keep_prob[k + 1]);
			}

			if (dropConnect)
			{
				for (int jj = 0; jj < numOfLayers - 2; jj++)
				{
					D2[Cores][jj] = new matrix<bool>*[minibatchSize];
					for (int kk = 0; kk < minibatchSize; kk++)
					{
						D2[Cores][jj][kk] = new matrix<bool>(layers[jj + 1].neurons, layers[jj].neurons, Bernoulli, keep_prob[jj + 1]);
					}
				}
			}


			if (Arg->Stack)
			{

				cur_X = A->SubMat(0, j*minibatchSize, Arg->A->Rows() - 1, ((j + 1)*(minibatchSize)-1));
				cur_Y = A_hat->SubMat(0, j*minibatchSize, Arg->A->Rows() - 1, ((j + 1)*(minibatchSize)-1));
			}
			else
			{
				cur_X = ConvertMat_U(X->SubMat(0, j*minibatchSize, X->Rows() - 1, ((j + 1)*(minibatchSize)-1)), UC_F);
				cur_Y = ConvertMat_U(Y->SubMat(0, j*minibatchSize, Y->Rows() - 1, ((j + 1)*(minibatchSize)-1)), UC_F);
			}

			FC_Cache[Cores].put("A0", cur_X);
			Matrix* Y_hat = FC_FeedForward(TRAIN, Cores);

			FC_CalGrads(cur_Y, Y_hat, Cores);
			FC_UpdateParameters(t, Cores);
			t++;
			delete cur_Y;
			FC_Cache[Cores].DeleteThenClear();
			if (!(j % 1))
				cout << ".";
		}

		if (LastBatchSize != 0)
		{

			if (dropout)
			{
				for (int k = 0; k < numOfLayers - 1; k++)
					D[Cores][k + 1] = new Matrix(layers[k + 1].neurons, LastBatchSize, Bernoulli, keep_prob[k + 1]);
			}

			if (dropConnect)
			{
				for (int jj = 0; jj < numOfLayers - 2; jj++)
				{
					D2[Cores][jj] = new matrix<bool>*[LastBatchSize];
					for (int kk = 0; kk < LastBatchSize; kk++)
					{
						D2[Cores][jj][kk] = new matrix<bool>(layers[jj + 1].neurons, layers[jj].neurons, Bernoulli, keep_prob[jj + 1]);
					}
				}
			}

			if (Arg->Stack)
			{
				cur_X = A->SubMat(0, j*minibatchSize, Arg->A->Rows() - 1, Arg->A->Columns() - 1);
				cur_Y = A_hat->SubMat(0, j*minibatchSize, Arg->A->Rows() - 1, Arg->A->Columns() - 1);
			}
			else
			{
				cur_X = ConvertMat_U(X->SubMat(0, j*minibatchSize, X->Rows() - 1, X->Columns() - 1), UC_F);
				cur_Y = ConvertMat_U(Y->SubMat(0, j*minibatchSize, Y->Rows() - 1, Y->Columns() - 1), UC_F);
			}

			FC_Cache[Cores].put("A0", cur_X);
			Matrix* Y_hat = FC_FeedForward(TRAIN, Cores);
			FC_CalGrads(cur_Y, Y_hat, Cores);
			FC_UpdateParameters(t, Cores);
			t++;
			delete cur_Y;
			FC_Cache[Cores].DeleteThenClear();
			cout << ".";
		}
		cout << endl;
	}
	/*END OF EPOCHS ITERATIONS*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AutoEncoder::test_FC(Mode mode)
{
	/*ARGUMENT LIST*/
	int Test_batchSize = Arg->Test_Batch_Size;
	/*END OF ARGUMENT LIST*/

	Matrix* src = nullptr;
	Matrix* dst = nullptr;
	U_IntMatrix* X = Arg->X_test;
	U_IntMatrix* Y = Arg->Y_test;

	int numOfminiBatch = (Arg->Stack) ? Arg->A->Columns() / Test_batchSize : X->Columns() / Test_batchSize;
	Arg->prevCost = Arg->curCost;
	for (int j = 0; j < numOfminiBatch; j++)
	{
		Matrix* cur_X = nullptr;
		Matrix* cur_Y = nullptr;
		if (Arg->Stack)
		{
			cur_X = Arg->A->SubMat(0, j*Test_batchSize, Arg->A->Rows() - 1, ((j + 1)*(Test_batchSize)-1));
			cur_Y = Arg->A->SubMat(0, j*Test_batchSize, Arg->A->Rows() - 1, ((j + 1)*(Test_batchSize)-1));
		}
		else
		{
			cur_X = ConvertMat_U(X->SubMat(0, j*Test_batchSize, X->Rows() - 1, ((j + 1)*(Test_batchSize)-1)), UC_F);
			cur_Y = ConvertMat_U(X->SubMat(0, j*Test_batchSize, X->Rows() - 1, ((j + 1)*(Test_batchSize)-1)), UC_F);
		}
		FC_Cache[Cores].put("A0", cur_X);
		Matrix* Y_hat = FC_FeedForward(TEST, Cores);
		Arg->curCost = AccuracyTest(cur_Y, Y_hat, *Arg, *DP);
		FC_Cache[Cores].DeleteThenClear();
		delete cur_Y;
	}
}

