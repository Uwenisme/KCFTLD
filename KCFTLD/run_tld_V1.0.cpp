#include <video\video.hpp>
#include <fstream>

#include "TLD_V1.0.h"

bool gIsDrawing = false;
bool gGotBB = false;
BoundingBox box;

std::ofstream ff;

void readBB(char* file)
{
	ifstream bb_file(file);   //创建文件输入流
	string line;
	getline(bb_file, line); //获取文件中的第一行
	//将输入流bb_file中读到的字符存入string
	//对象line中，终结符默认为'\n'
	istringstream linestream(line); //istringstream对象可以绑定一行
	//字符串，然后以空格为分隔符把该行
	//分隔开来
	string x1, y1, x2, y2;
	getline(linestream, x1, ','); //将输入流linestream读到的字符存入
	getline(linestream, y1, ','); //定义的x1,y1,x2,y2字符串变量中
	getline(linestream, x2, ','); //直到遇到终结符‘，’终止读取。
	getline(linestream, y2, ',');
	int x = atoi(x1.c_str());//把字符串转换成整型数
	int y = atoi(y1.c_str());// = (int)file["bb_y"];
	int w = atoi(x2.c_str()) - x;// = (int)file["bb_w"];
	int h = atoi(y2.c_str()) - y;// = (int)file["bb_h"];
	box = Rect(x, y, w, h);
	bb_file.close();
}

void mouseHandler_v(int event, int x, int y, int flags ,void *param)
{
	switch (event){
	case CV_EVENT_MOUSEMOVE:
		if (gIsDrawing){
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		gIsDrawing = true;
		box = Rect(x, y, 0, 0);
		break;
	case CV_EVENT_LBUTTONUP:
		gIsDrawing = false;
		if (box.width < 0){
			box.x += box.width;
			box.width *= -1;
		}
		if (box.height < 0){
			box.y += box.height;
			box.height *= -1;
		}
		gGotBB = true;  //已经获得bounding box
		break;
	}
}

int main()
{
	int isfromVideoFile_i;
	int isInitbyFile_i;
	int videoIndex_i;

	VideoCapture capture;

	cout << "请选择视频来源：" << endl
		<< "0--从摄像机" << endl
		<< "1--从视频文件" << endl;
	cin >> isfromVideoFile_i;


	if (isfromVideoFile_i)
	{
		cout << "请选择初始化目标框方式" << endl
			<< "0--人工初始化" << endl
			<< "1--文件初始化" << endl;
		cin >> isInitbyFile_i;

		
		cout << "请选择数据集" << endl
			<< "0--David" << endl
			<< "1--Jumping" << endl
			<< "2--Pdestrian1" << endl
			<< "3--Pedestrian2" << endl
			<< "4--Pedestrian3" << endl
			<< "5--car" << endl
			<< "6--motocross" << endl
			<< "7--volkswagen" << endl
			<< "8--carchase" << endl
			<< "9--panda" << endl;
		cin >> videoIndex_i;

		char *name[10] = { "./datasets/01_david/david.avi", "./datasets/02_jumping/jumping.mpg",
			"./datasets/03_pedestrian1/pedestrian1.mpg", "./datasets/04_pedestrian2/pedestrian2.mpg",
			"./datasets/05_pedestrian3/pedestrian3.mpg", "./datasets/06_car/car.mpg",
			"./datasets/07_motocross/motocross.mpg", "./datasets/08_volkswagen/volkswagen.mpg",
			"./datasets/09_carchase/carchase.mpg", "./datasets/10_panda/panda.mpg" };


		if (!capture.open(name[videoIndex_i]))
		{
			printf("open the Video fail!");
			getchar();
			getchar();
			return -1;
		}
	}
	else
	{
		int videxidx;
		cout << "请选择摄像头编号，一般在您电脑上编号为0，1...依次递增" << endl;
		cin >> videxidx;
		while (!capture.open(videxidx))//这里可以改变编号，选择不同摄像机编号不同
		{
			cout << "打开摄像头失败，请再次输入编号!"<<endl;
			cin >> videxidx;
		}

		capture.set(CV_CAP_PROP_FRAME_WIDTH, 340); //设置获取的图像大小为320*240
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	}

	Mat FirstFrame_cvM;
	Mat DrawingFrame_cvM;
	cvNamedWindow("OpenTLD");

	capture >> FirstFrame_cvM;

	if (isInitbyFile_i)
	{
		char *initname[10] = { "./datasets/01_david/init.txt", "./datasets/02_jumping/init.txt",
			"./datasets/03_pedestrian1/init.txt", "./datasets/04_pedestrian2/init.txt",
			"./datasets/05_pedestrian3/init.txt", "./datasets/06_car/init.txt",
			"./datasets/07_motocross/init.txt", "./datasets/08_volkswagen/init.txt",
			"./datasets/09_carchase/init.txt", "./datasets/10_panda/init.txt" };

		readBB(initname[videoIndex_i]);
	}
	else
	{
		
		cvSetMouseCallback("OpenTLD", mouseHandler_v, NULL);//设置鼠标点击事件
		
		while (!gGotBB)
		{
			FirstFrame_cvM.copyTo(DrawingFrame_cvM);
			rectangle(DrawingFrame_cvM, Rect(box.x, box.y, box.width, box.height), Scalar(255, 255, 255),2);
			imshow("OpenTLD", DrawingFrame_cvM);
			waitKey(2);
			/*
			if (gGotBB && min(box.width, box.height)<(int)fs.getFirstTopLevelNode()["min_win"])
			{
				cout << "Bounding box too small, try again." << endl;
				gGotBB = false;
			}*/
		}
		cvSetMouseCallback("OpenTLD", NULL, NULL);//取消鼠标点击事件
	}
	Mat CurrImg_cvM;
	Mat CurrGrayImg_cvM;
	Mat NextImg_cvM;
	Mat NextGrayImg_cvM;
	BoundingBox Nextbox;
	bool isFound = true;
	
	FileStorage fs;
	if (!fs.open("./parameters.yml",  FileStorage::READ))
	{

		cout << "初始化文件打开失败！" << endl;
		getchar();
		getchar();
		return -1;
	}
	//fs << "mthrN" << 5;
	//fs << "mthrP" << 6;
	//fs << "thrGoodOverlap" << 0.6;
	//fs << "thrBadOverlap" << 0.2;

	TLD tld(fs.getFirstTopLevelNode());
	FirstFrame_cvM.copyTo(CurrImg_cvM);
	cvtColor(CurrImg_cvM, CurrGrayImg_cvM, CV_RGB2GRAY);
	tld.init_v(CurrGrayImg_cvM, box, CurrImg_cvM);

	int SumFrame_i=0;
	int SumFound_i=0;
	char str[15];
	clock_t start, stop;
	start = clock();
	ff.open("data.txt");
	while (capture.read(NextImg_cvM))
	{

		cvtColor(NextImg_cvM, NextGrayImg_cvM, CV_RGB2GRAY);
		

		tld.processFrame(CurrGrayImg_cvM, NextGrayImg_cvM, Nextbox, isFound, NextImg_cvM);

		if (isFound)
		{
		    rectangle(NextImg_cvM, Rect(Nextbox.x, Nextbox.y, Nextbox.width, Nextbox.height), Scalar(255, 255, 255),2);
			SumFound_i++;
			//ff << Nextbox.x <<' '<< Nextbox.y <<' '<< Nextbox.width <<' '<< Nextbox.height << endl;
		}
		
		SumFrame_i++;

		sprintf(str, "%d/%d ", SumFound_i, SumFrame_i);
		putText(NextImg_cvM, str, cvPoint(0, 20), 2, 1, CV_RGB(25, 200, 25));

		imshow("OpenTLD", NextImg_cvM);
		waitKey(1);

		swap(CurrGrayImg_cvM, NextGrayImg_cvM);
		//NextImg_cvM.copyTo(CurrImg_cvM);
		
	}
	tld.mDetelteGrid_ptr();//释放Grid_ptr内存
	ff.close();
	stop = clock();
	float  time = (stop - start) * 1000 / CLK_TCK;
	printf("sum time is:%f\n", time);
	system("pause");
	
	return 0;
}