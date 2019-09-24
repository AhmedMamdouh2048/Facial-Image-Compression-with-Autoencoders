#ifndef AE_TOOLS_H_INCLUDED
#define AE_TOOLS_H_INCLUDED
#include <iostream>
#include <thread>
#include "Matrix.h"
#include "Activations.h"
//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////  Matrix types   //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef matrix<float> Matrix;
typedef matrix<unsigned char> U_IntMatrix;
typedef matrix<signed char> IntMatrix;
//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////  Enumerations   //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
enum TypeOfNet { FC, LENET1, CUSTOM };
enum TypeOfConversion { F_UC, F_C, F_UI16, F_I16, UC_F, C_F, UI16_F, I16_F };
enum Optimizer { ADAM, GRADIENT_DESCENT };
enum ErrorType { SQAURE_ERROR, CROSS_ENTROPY };
enum Mode { TRAIN, DEV, TEST, MAX, AVG };
enum Choice { YES, NO };
enum NoiseType { SP, Gauss };
//////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////  Structures   ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
struct layer
{
	float neurons;
	ActivationType activation;
	float numOfLinearNodes;

	void put(float n, ActivationType activ)
	{
		neurons = n;
		activation = activ;
	}
};
struct DeNorm
{
	Matrix* mean;
	Matrix* var_sqrt;
	float min;
	float max;
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////
struct DatasetParam
{
	uint32_t Train_Examples;
	uint32_t Test_Examples;
	uint32_t ImageSize;
	uint32_t ImageDim;
	uint32_t BIG_FILE;
	uint32_t KDEF;
	uint32_t CFEED;
	uint32_t AR;
	uint32_t TEST_FILE;
	float Resize_Fact;
	float Noise_Fact;
	float Noise_Mean;
	float Noise_Var;
	uint32_t Face_Size;
	uint32_t Eyes_Size;
	uint32_t Nose_Size;
	uint32_t Mouth_Size;
	bool Display_Data;
	bool Noisy_Data;
	bool LandMarks;
	bool ReScale;
	bool Get_NewData;
	bool Fill_DeNorm;
	int numFiles;
	int curFile;
	const char** X_dir;
	const char** A_dir;
	const char* TextData_dir;
	const char* Xtest_dir;
	const char* Noisy_dir;
	const char* Disp_dir;
	const char* ParametersPath;
	const char* ActivationsPath;
	DeNorm* X_DeNorm;
	DatasetParam() : X_DeNorm(new DeNorm[4]), Fill_DeNorm(true)
 	{}
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Arguments
{
	TypeOfNet NetType;                  // Type of neural network. Either FC (fully connected) or LENET1 (convolutional)
	Optimizer optimizer;                // The type of optimizer. Either Gradient Descent or ADAM
	ErrorType ErrType;                  // The type of error used. Either square error or cross entropy
	layer* layers;                      // The activations and number of neurons in each layer
	int numOfLayers;                    // The number of layers
	int numOfEpochs;                    // The total number of epochs required for training
	int batchSize;                      // The batch size
	int numPrint;                       // The rate of printing the accuracy on the screen
	int Test_Batch_Size;				// The size of the test batch for cost calculation
	float learingRate;                  // The learning rate alpha
	float decayRate;                    // The decay of learining rate
	float curLearningRate;              // The current value of learning rate
	float Rho;							// The desired average hidden units in sparse Auto-Encoder
	float beta_sparse;					// The weight of sparsity in the cost function
	float regularizationParameter;      // The regularization parameter lambda
	float lambda_Contractive;           // The weight of Contractivity in the cost function
	float curCost;                      // Current value of cost function
	float prevCost;                     // Previous value of cost function
	float threshold;                    // A threshold in error calculation
	float numErrors;                    // Number of patterns that have error larger than threshold
	float* keep_prob;                   // The probabilty distribution of drop-out and drop-connection across the layers
	bool batchNorm;                     // Is batch normalization activated or not
	bool dropout;                       // Is drop-out activated or not
	bool dropConnect;                   // Is drop-connection activated or not
	bool SPARSE;						// Is Sparse Auto-Encoder enabled or not
	bool tiedWeights;                   // If the weights are tied or not (W2=W1.transpose())
	bool Contractive;                   // Is Contractive Auto-Encoder enabled or not
	bool Stack;                         // Is Stacked Auto-Encoder enabled or not
	bool RandomBatch;                   // Is the batch is taken in random fashion or not
	bool SaveActivation;				// Save the activation or not
	bool SaveParameters;                // Save the parameters or not
	bool RetrieveParameters;            // Retrieve the parameters or not
	bool TestParameters;				// Test the retrieved parameters
	string ActivationsPath;             // The path of the file at which the actications are saved at
	U_IntMatrix* X;                     // The input dataset
	U_IntMatrix* Y;                     // The labels of the input dataset
	U_IntMatrix* X_dev;                 // The development set
	U_IntMatrix* Y_dev;                 // The labels of the development dataset
	U_IntMatrix* X_test;                // The test set
	U_IntMatrix* Y_test;                // The labels of the test set
	U_IntMatrix* X_disp;                // The matrix to be displayed
	U_IntMatrix* Faces;                 // The development set
	U_IntMatrix* Eyes;                  // The labels of the development dataset
	U_IntMatrix* Mouths;                // The test set
	U_IntMatrix* Noses;                 // The labels of the test set
	Matrix* Rho_hat;					// The average of hidden units in sparse Auto-Encoder
	Matrix* A;                          // The Hidden Layer Activation
	Matrix* A_Noisy;                    // The Noisy Hidden Layer Activation
	Arguments() : Rho_hat(new Matrix(1, 1)), ActivationsPath("NONE"), curCost(9999), prevCost(9999)
	{}
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////  String generation   ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string CharGen(std::string name, int i);
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////  Testing and evaluation   /////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float AccuracyTest(Matrix* Y, Matrix* Y_hat, Arguments& Arg, DatasetParam& DP);
U_IntMatrix** cluster(U_IntMatrix* X_test, Matrix* A, float CompValue);
float CosineSim(Matrix* Vect1, Matrix* Vect2);
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////  Threadded dot product   //////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* DOT(Matrix* X, Matrix* Y);
void DotPart(int part, Matrix* result, Matrix* X, Matrix* Y);
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////  Matrix type conversion   /////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* ConvertMat_U(U_IntMatrix* src, TypeOfConversion TYPE, Choice DeleteSrc = YES);
U_IntMatrix* ConvertMat_U(Matrix* src, TypeOfConversion TYPE, Choice DeleteSrc = YES);
Matrix* ConvertMat_S(IntMatrix* src, TypeOfConversion TYPE, Choice DeleteSrc = YES);
IntMatrix* ConvertMat_S(Matrix* src, TypeOfConversion TYPE, Choice DeleteSrc = YES);
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////    Printing utilities    //////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void PrintLayout(Arguments& Arg, DatasetParam& DP);
#endif // AE_TOOLS_H_INCLUDED
