#include "DataSet.h"
#include "AutoEncoder.h"
#define ever ;;
int main()
{
	
	//string OLDFolder = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Faces\\UTKFace\\UTKFace";
	//string NEWFolder = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Original\\Large CV Lab DataSet\\TEST";
	//U_IntMatrix* a = ReadImageFolder(OLDFolder, 20000, 100, YES);
	//Shuffle(a);
	//////_getche();
	//a->WriteDataSet("F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\UTKFACE");
	////to_JPG(a, NEWFolder);
	//_getche();
	

	/*string ImageFolder = "F:\\College\\Graduation Project\\Datasets\\Images Grey\\Persons\\";
	U_IntMatrix* XPersons = new U_IntMatrix(100 * 100, 165);
	int count = 0;
	for (int i = 0; i < 489; i++)
	{
		string temp = ImageFolder + to_string(i + 1) + "\\";
		U_IntMatrix* a = ReadImageFolder(temp, 8, 100, NO);
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 100 * 100; k++)
			{
				XPersons->access(k, count) = a->access(k, j);
			}
			count++;
		}
		delete a;
		cout << "Folder no." << i + 1 << " Ended" << endl;
	}
	XPersons->WriteDataSet("F:\\College\\Graduation Project\\Datasets\\Binary Files\\Persons");*/


	//------------------------------------------------------------------//
	//-------------------------- DataSet -------------------------------//
	//------------------------------------------------------------------//
	const char* KDEF	 = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Binary Files\\KDEF_CROP";
	const char* CFEED	 = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Binary Files\\CFEED_CROP";
	const char* AR		 = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Binary Files\\AR_CROP";
	const char* test     = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Binary Files\\X_TEST";
	const char* Persons  = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Binary Files\\Persons";
	const char* UTK      = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\UTKFACE";

	const char* para_500 = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Parameters\\500\\";
	const char* para_400 = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Parameters\\400\\";
	const char* para_200 = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Parameters\\200\\";
	const char* para_150 = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Parameters\\\\150-LEAKY\\";
	const char* para_100 = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Parameters\\100\\";
	const char* para_50  = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Parameters\\50-LEAKY\\";
	const char* para_30  = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Parameters\\30\\";
	const char* para_25  = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Parameters\\20\\";
	const char* para_20  = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Parameters\\20\\";

	srand(time(NULL));
	DatasetParam DP;
	DP.BIG_FILE = 20000;
	DP.CFEED = 423;
	DP.AR = 1300;
	DP.KDEF = 521;
	DP.TEST_FILE = 321;
	DP.numFiles = 8;
	DP.curFile = 0;
	DP.Train_Examples = DP.BIG_FILE;
	DP.Test_Examples = DP.TEST_FILE;
	//----------------------------------------//
	DP.ImageSize = 10000;
	DP.ImageDim = 100;
	DP.Resize_Fact = 1;
	DP.Noise_Fact = 0.1;
	DP.Noise_Mean = 10;
	DP.Noise_Var = 10;
	//----------------------------------------//
	DP.Face_Size = 100;
	DP.Eyes_Size = 20;
	DP.Nose_Size = 20;
	DP.Mouth_Size = 20;
	//----------------------------------------//
	DP.Noisy_Data = false;
	DP.LandMarks = false;
	DP.Display_Data = true;
	DP.ReScale = false;
	DP.Get_NewData = false;
	//----------------------------------------//
	DP.X_dir = new const char*[DP.numFiles];
	DP.A_dir = new const char*[DP.numFiles];
	//-----------------------------------------//
	DP.X_dir[0] = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Binary Files\\X_TRAIN1";
	DP.X_dir[1] = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Binary Files\\X_TRAIN2";
	DP.X_dir[2] = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Binary Files\\X_TRAIN3";
	DP.X_dir[3] = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Datasets\\Binary Files\\X_TRAIN4";
	//DP.X_dir[4] = "F:\\Final Project\\X_TRAIN5";
	//DP.X_dir[5] = "F:\\Final Project\\X_TRAIN6";
	//DP.X_dir[6] = "F:\\Final Project\\X_TRAIN7";
	//DP.X_dir[7] = "F:\\Final Project\\X_TRAIN8";
	//------------------------------------------//
	DP.A_dir[0] = "F:\\Final Project\\DataSet\\curActivations\\A_TRAIN_1";
	DP.A_dir[1] = "F:\\Final Project\\DataSet\\curActivations\\A_TRAIN_2";
	DP.A_dir[2] = "F:\\Final Project\\DataSet\\curActivations\\A_TRAIN_3";
	DP.A_dir[3] = "F:\\Final Project\\DataSet\\curActivations\\A_TRAIN_4";
	//DP.A_dir[4] = "F:\\Final Project\\DataSet\\curActivations\\A_TRAIN_5";
	//DP.A_dir[5] = "F:\\Final Project\\DataSet\\curActivations\\A_TRAIN_6";
	//DP.A_dir[6] = "F:\\Final Project\\DataSet\\curActivations\\A_TRAIN_7";
	//DP.A_dir[7] = "F:\\Final Project\\DataSet\\curActivations\\A_TRAIN_8";
	//-----------------------------------------//
	DP.TextData_dir = "F:\\7-7\\rest\\4.txt";
	DP.Xtest_dir = UTK;
	DP.Noisy_dir = "F:\\Final Project\\NoisyImgs";
	DP.Disp_dir = "F:\\Final Project\\X_TEST";
	DP.ActivationsPath = "F:\\Final Project\\DataSet\\curActivations\\A_TRAIN";
	DP.ParametersPath = "F:\\Computer and Control Engineering\\College\\Graduation Project\\Final Project 2019\\Parameters\\curParameters\\";
	//------------------------------------------------------------------------//

	


	/*Matrix* HH = new Matrix(500, 400);
	HH->Read("F:\\Final Project\\DataSet\\curActivations\\A_TRAIN_1");
	HH->print();*/
	

	//------------------------------------------------------------------//
	//------------------- Network Architecture -------------------------//
	//------------------------------------------------------------------//
	int numOfLayers = 3;
	layer*  layers = new layer[numOfLayers];
	layers[0].put(DP.ImageSize * DP.Resize_Fact * DP.Resize_Fact, NONE);
	layers[1].put(200, SIGMOID);
	layers[2].put(DP.ImageSize * DP.Resize_Fact * DP.Resize_Fact, SIGMOID);
	float*  keep_prob = new float[numOfLayers];
	keep_prob[0] = 1;
	keep_prob[1] = 0.5;
	keep_prob[2] = 1;


	Arguments Arg;
	Arg.NetType = FC;
	Arg.optimizer = ADAM;
	Arg.ErrType = CROSS_ENTROPY;
	Arg.layers = layers;
	Arg.numOfLayers = numOfLayers;
	Arg.keep_prob = keep_prob;
	//---------------------------//
	Arg.numPrint = 1;
	Arg.numOfEpochs = 1;
	Arg.batchSize = 200;
	Arg.Test_Batch_Size = 321;
	Arg.threshold = 10;
	//---------------------------//
	Arg.learingRate = 0.005;
	Arg.decayRate = 1;
	Arg.Rho = 0.05;
	Arg.regularizationParameter = 0;
	Arg.beta_sparse = 0.5;
	Arg.lambda_Contractive = 0.7;
	//---------------------------//
	Arg.batchNorm = true;
	Arg.dropout = false;
	Arg.dropConnect = false;
	Arg.SPARSE = false;
	Arg.Contractive = false;
	Arg.tiedWeights = false;
	Arg.Stack = false;
	Arg.RandomBatch = false;
	//----------------------------//
	Arg.SaveActivation = false;
	Arg.SaveParameters = false;
	Arg.RetrieveParameters = true;
	Arg.TestParameters = true;
	//----------------------------//
	
	/*string OLDFolder = "F:\\UsefullDataSets\\Perosns_8\\Persons_8";
	U_IntMatrix* o = new U_IntMatrix(8, 10000);
	o->ReadDataSet(OLDFolder);
	o = o->TRANSPOSE();
	Shuffle(o);
	Arg.X_disp = o;
	DisplayData(Arg, DP);
	o->WriteDataSet("F:\\UsefullDataSets\\Perosns_8\\PersonsShuf_8");*/


	
	//------------------------------------------------------------------//
	//-------------------------- Training ------------------------------//
	//------------------------------------------------------------------//
	AutoEncoder AE(&Arg, &DP);
	Get_TestSet(Arg, DP);
	PrintLayout(Arg, DP);
	AE.RetrieveParameters();
	AE.StoreActivations();
	AE.TestParameters();
	int i = 0;
	//shuffle every epoch ok
	//add regularization ok
	//decrease learning rate  ok
	//increase learning rate  ok
	//no initialization 500/200 ok
	//try decay every 40k ok
	//LeakyRelu ok
	for (ever)
	{
		clock_t start = clock();
		cout << endl << ">> Epoch no. " << ++i << ":" << endl;
		if (Arg.Stack)
		{
			for (DP.curFile = 0; DP.curFile < DP.numFiles; DP.curFile++)
			{
				Arg.A = new Matrix(DP.ImageSize, DP.Train_Examples);
				Arg.A->Read(DP.A_dir[DP.curFile]);
				AE.train();
				AE.test(TEST);
				delete Arg.A;
				if (Arg.curCost < Arg.prevCost)
					AE.StoreParameters();
			}
			Arg.learingRate = Arg.learingRate * Arg.decayRate;
		}
		else
		{
			for (DP.curFile = 0; DP.curFile < DP.numFiles; DP.curFile++)
			{
				Get_TrainSet(Arg, DP);
				AE.train();
				AE.test(TEST);
				delete Arg.Y;
				if (Arg.curCost < Arg.prevCost)
					AE.StoreParameters();
			}
		}
		Arg.learingRate = Arg.learingRate * Arg.decayRate;
		clock_t end = clock();
		double duration_sec = double(end - start) / CLOCKS_PER_SEC;
		cout << "Learning Rate = " << Arg.learingRate << endl;
		cout << "Time = " << duration_sec << endl;
	}
	_getche();
	return 0;
}

