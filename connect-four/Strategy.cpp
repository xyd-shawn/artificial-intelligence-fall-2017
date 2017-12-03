#include <iostream>
#include "Point.h"
#include "Strategy.h"
#include "Judge.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>

using namespace std;

/*
	策略函数接口,该函数被对抗平台调用,每次传入当前状态,要求输出你的落子点,该落子点必须是一个符合游戏规则的落子点,不然对抗平台会直接认为你的程序有误
	
	input:
		为了防止对对抗平台维护的数据造成更改，所有传入的参数均为const属性
		M, N : 棋盘大小 M - 行数 N - 列数 均从0开始计， 左上角为坐标原点，行用x标记，列用y标记
		top : 当前棋盘每一列列顶的实际位置. e.g. 第i列为空,则_top[i] == M, 第i列已满,则_top[i] == 0
		_board : 棋盘的一维数组表示, 为了方便使用，在该函数刚开始处，我们已经将其转化为了二维数组board
				你只需直接使用board即可，左上角为坐标原点，数组从[0][0]开始计(不是[1][1])
				board[x][y]表示第x行、第y列的点(从0开始计)
				board[x][y] == 0/1/2 分别对应(x,y)处 无落子/有用户的子/有程序的子,不可落子点处的值也为0
		lastX, lastY : 对方上一次落子的位置, 你可能不需要该参数，也可能需要的不仅仅是对方一步的
				落子位置，这时你可以在自己的程序中记录对方连续多步的落子位置，这完全取决于你自己的策略
		noX, noY : 棋盘上的不可落子点(注:其实这里给出的top已经替你处理了不可落子点，也就是说如果某一步
				所落的子的上面恰是不可落子点，那么UI工程中的代码就已经将该列的top值又进行了一次减一操作，
				所以在你的代码中也可以根本不使用noX和noY这两个参数，完全认为top数组就是当前每列的顶部即可,
				当然如果你想使用lastX,lastY参数，有可能就要同时考虑noX和noY了)
		以上参数实际上包含了当前状态(M N _top _board)以及历史信息(lastX lastY),你要做的就是在这些信息下给出尽可能明智的落子点
	output:
		你的落子点Point
*/

extern "C" __declspec(dllexport) Point* getPoint(const int M, const int N, const int* top, const int* _board, 
	const int lastX, const int lastY, const int noX, const int noY){
	/*
		不要更改这段代码
	*/
	int x = -1, y = -1;//最终将你的落子点存到x,y中
	int** board = new int*[M];
	for(int i = 0; i < M; i++){
		board[i] = new int[N];
		for(int j = 0; j < N; j++){
			board[i][j] = _board[i * N + j];
		}
	}
	
	/*
		根据你自己的策略来返回落子点,也就是根据你的策略完成对x,y的赋值
		该部分对参数使用没有限制，为了方便实现，你可以定义自己新的类、.h文件、.cpp文件
	*/
	//Add your own code below
	
	/*
     //a naive example
	for (int i = N-1; i >= 0; i--) {
		if (top[i] > 0) {
			x = top[i] - 1;
			y = i;
			break;
		}
	}
	*/

	int* ttop = new int[N];
	for(int j = 0; j < N; j++) {
		ttop[j] = top[j];
	}

	int method = 1;    // method=1表示UCT算法，method=2表示alpha-beta剪枝算法
	if(method == 1) {
		srand(unsigned(time(0)));

	    Node root = Node(M, N, lastX, lastY, false, board, ttop);
	    Node* s = &root;
	    y = uct_search(s, board, ttop, 1.0, noX, noY);
	    x = top[y] - 1;
	}
	if(method == 2) {
		int depth = 6;    // 博弈树的深度
		abNode root = abNode(M, N, lastX, lastY, false, board, ttop);
		abNode* v = &root;
		alpha_beta_pruning(depth, 0, v, board, ttop, noX, noY);
		y = v->bestpos;
		x = top[y] - 1;
	}
		
	delete[] ttop;
    
	/*
		不要更改这段代码
	*/
	clearArray(M, N, board);
	
	return new Point(x, y);
}


/*
	getPoint函数返回的Point指针是在本dll模块中声明的，为避免产生堆错误，应在外部调用本dll中的
	函数来释放空间，而不应该在外部直接delete
*/
extern "C" __declspec(dllexport) void clearPoint(Point* p){
	delete p;
	return;
}

/*
	清除top和board数组
*/
void clearArray(int M, int N, int** board){
	for(int i = 0; i < M; i++){
		delete[] board[i];
	}
	delete[] board;
}


/*
	添加你自己的辅助函数，你可以声明自己的类、函数，添加新的.h .cpp文件来辅助实现你的想法
*/


int uct_search(Node* s, int** initboard, int* inittop, int coef, int nx, int ny) {
	clock_t t0 = clock();
	clock_t t1 = clock();
	double sec = (double)(t1 - t0) / CLOCKS_PER_SEC;

	Node* v0 = s;
	int m = s->mm;
	int n = s->nn;
	while(sec < MAXSEC) {    // 控制运行时间
		int** curboard = new int*[m];    // 复制棋盘以及top数组
	    for(int i = 0; i < m; i++) {
		    curboard[i] = new int[n];
		    for(int j = 0; j < n; j++) {
			    curboard[i][j] = initboard[i][j];
		    }
	    }
	    int* curtop = new int[n];
	    for(int j = 0; j < n; j++) {
		    curtop[j] = inittop[j];
	    }
		v0 = s;

		Node* v1 = tree_policy(v0, curboard, curtop, coef, nx, ny);
		double delta = default_policy(v1, curboard, curtop, nx, ny);
		back_up(v1, delta);

		delete[] curtop;
		clearArray(m, n, curboard);
		t1 = clock();
		sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
	}

	int res = best_child(s, 0);
	return res;
}

Node* tree_policy(Node* v, int** curboard, int* curtop, int coef, int nx, int ny) {
	int m = v->mm;
	int n = v->nn;
	while(v->state == 0) {
		for(int j = 0; j < n; j++) {
			if((v->children[j] == NULL) && (curtop[j] > 0)) {    // 第j个子节点未遍历并且第j列可以填子
				return expand(v, j, curboard, curtop, nx, ny);
			}
		}
		int bestind = best_child(v, coef);
		int xx = curtop[bestind] - 1;
		curboard[xx][bestind] = (v->myturn)? 1: 2;
		curtop[bestind]--;
		if(((xx - 1) == nx) && (bestind == ny)) {    // 当前落子上方是不可落子点
			curtop[bestind]--;
		}
		v = v->children[bestind];    // 对最佳子节点重复上述操作
	}
	return v;
}
	
int default_policy(Node* v, int** curboard, int* curtop, int nx, int ny) {
	int m = v->mm;
	int n = v->nn;
	if(v->state == 2) {
		// return (v->myturn)? 2: 0;    // 每个节点的回报值都基于本方的时候
		return 1;
	}
	bool mt = v->myturn;
	while(!isTie(n, curtop)) {    // 向后模拟棋局对弈过程，直到分出胜负平
		mt = !mt;
		int yy = int(rand() % n);
		while(curtop[yy] <= 0) {
			yy = (yy + 1) % n;
		}
		int xx = curtop[yy] - 1;
		curtop[yy]--;
		curboard[xx][yy] = (mt)? 2: 1;
		if(((xx - 1) == nx) && (yy == ny)) {
			curtop[yy]--;
		}
		if((mt) && machineWin(xx, yy, m, n, curboard)) {
			// return 2;
			return (v->myturn)? 1: -1;
		} else if((!mt) && userWin(xx, yy, m, n, curboard)) {
			// return 0;
			return (v->myturn)? -1: 1;
		}
	}
	// return 1;
	return 0;
}

void back_up(Node* v, int delta) {    // 向上回退
	while(v != NULL) {
		v->fre += 1;
		v->score += delta;
		delta = -delta;
		v = v->parent;
	}
}

int best_child(Node* v, double coef) {    // 寻找最佳子节点
	int bestind = -1;
	double ucb = -10;
	int n = v->nn;
	for(int j = 0; j < n; j++) {
		if(v->children[j] != NULL) {
			if(v->children[j]->state == 2) {
				bestind = j;
				break;
			} else {
				double temp = coef * sqrt(2.0 * log((double)(v->fre)) / (double)(v->children[j]->fre));    // 探索
				if(v->children[j]->state == 1) {
					// temp += 1.0;    // 每个节点的回报值都基于本方的时候
					temp += 0.0;
				} else {
					// 每个节点的回报值都基于本方的时候
					/*
					if(v->children[j]->myturn) {
						temp += (double)(v->children[j]->score) / (double)(v->children[j]->fre);
					} else {
						temp += (2.0 - (double)(v->children[j]->score) / (double)(v->children[j]->fre));
					}
					*/
					temp += (double)(v->children[j]->score) / (double)(v->children[j]->fre);    // 利用
				}
				if(temp > ucb) {
					ucb = temp;
					bestind = j;
				}
			}
		}
	}
	return bestind;
}

Node* expand(Node* v, int exy, int** curboard, int* curtop, int nx, int ny) {    // 扩展节点
	int exx = curtop[exy] - 1;
	bool mt = !v->myturn;
	curboard[exx][exy] = (mt)? 2: 1;
	curtop[exy]--;
	if(((exx - 1) == nx) && (exy == ny)) {
		curtop[exy]--;
	}
	Node* nextv = new Node(v->mm, v->nn, exx, exy, mt, curboard, curtop);
	v->children[exy] = nextv;
	v->children[exy]->parent = v;
	return v->children[exy];
}


int evaluate_1(int M, int N, int** curboard, int* curtop, int noX, int noY) {    // 评估函数
	int mintop = M + 1;
	for(int j = 0; j < N; j++) {
		if(curtop[j] < mintop) {
			mintop = curtop[j];
		}
	}
	if(mintop > 0) {
		mintop--;
	}
	int** virboard_1 = new int*[M];
	for(int i = 0; i < M; i++){
		virboard_1[i] = new int[N];
		for(int j = 0; j < N; j++){
			virboard_1[i][j] = curboard[i][j];
		}
	}
	for(int j = 0; j < noY; j++) {
		if(curtop[j] > 0) {
			for(int i = mintop; i < curtop[j]; i++) {
				if((i != noX) || (j != noY)) {
					virboard_1[i][j] = 2;
				}
			}
		}
	}
	int count1 = count_form_4(2, M, N, virboard_1);
	clearArray(M, N, virboard_1);
	int ** virboard_2 = new int*[M];
	for(int i = 0; i < M; i++) {
		virboard_2[i] = new int[N];
		for(int j = 0; j < N; j++) {
			virboard_2[i][j] = curboard[i][j];
		}
	}
	for(int j = 0; j < N; j++) {
		if(curtop[j] > 0) {
			for(int i = mintop; i < curtop[j]; i++) {
				if((i != noX) || (j != noY)) {
					virboard_2[i][j] = 1;
				}
			}
		}
	}
	int count2 = count_form_4(1, M, N, virboard_2);
	clearArray(M, N, virboard_2);
	return count1 - count2;
}

int count_form_4(int uu, int M, int N, int** virboard) {    // 计算填充之后的棋盘中横向、纵向或者斜向连成四个uu的数目
	int res = 0;
	for(int i = 0; i <= M - 4; i++) {
		for(int j = 0; j < N; j++) {
			if(virboard[i][j] == uu) {
				if((virboard[i + 1][j] == uu) && (virboard[i + 2][j] == uu) && (virboard[i + 3][j] == uu)) {
					res++;
				}
			}
		}
	}
	for(int j = 0; j <= N - 4; j++) {
		for(int i = 0; i < M; i++) {
			if(virboard[i][j] == uu) {
				if((virboard[i][j + 1] == uu) && (virboard[i][j + 2] == uu) && (virboard[i][j + 3] == uu)) {
					res++;
				}
			}
		}
	}
	for(int i = 0; i <= M - 4; i++) {
		for(int j = 0; j <= N - 4; j++) {
			if(virboard[i][j] == uu) {
				if((virboard[i + 1][j + 1] == uu) && (virboard[i + 2][j + 2] == uu) && (virboard[i + 3][j + 3] == uu)) {
					res++;
				}
			}
		}
	}
	for(int i = 3; i < M; i++) {
		for(int j = 0; j <= M - 4; j++) {
			if(virboard[i][j] == uu) {
				if((virboard[i - 1][j + 1] == uu) && (virboard[i - 2][j + 2] == uu) && (virboard[i - 3][j + 3] == uu)) {
					res++;
				}
			}
		}
	}
	return res;
}

bool alpha_check(abNode* v) {
	abNode* prev = v->parent;
	while(prev != NULL) {
		if(prev->value == EMPTYNUM) {    // 祖先节点的估值还未被计算
			break;
		}
		if(v->value < prev->value) {     // 当前为极小层，beta值小于某个极大层祖先的alpha值
			return true;
		}
		prev = prev->parent;
		if(prev == NULL) {
			break;
		} else {
			prev = prev->parent;
		}
	}
	return false;
}

bool beta_check(abNode* v) {
	abNode* prev = v->parent;
	while(prev != NULL) {
		if(prev->value == EMPTYNUM) {    // 祖先节点的估值还未被计算
			break;
		}
		if(v->value > prev->value) {     // 当前为极大层，alpha值大于某个极小层祖先的beta值
			return true;
		}
		prev = prev->parent;
		if(prev == NULL) {
			break;
		} else {
			prev = prev->parent;
		}
	}
	return false;
}

void alpha_beta_pruning(int depth, int iter, abNode* v, int** curboard, int* curtop, int nx, int ny) {
	int m = v->mm;
	int n = v->nn;
	for(int j = 0; j < n; j++) {
		if(curtop[j] > 0) {
			int cx = curtop[j] - 1;
			curboard[cx][j] = (v->myturn)? 1: 2;
			curtop[j]--;
			if(((cx - 1) == nx) && (j == ny)) {
				curtop[j]--;
			}
			abNode* nextv = new abNode(m, n, cx, j, !v->myturn, curboard, curtop);
			v->children[j] = nextv;
			v->children[j]->parent = v;
			if(v->children[j]->state == 2) {
				v->children[j]->value = (v->myturn)? (-MAXNUM): (MAXNUM);    // 已经获胜或者已经落败
				v->value = v->children[j]->value;    // 向上赋值
				v->bestpos = j;    // 记录位置
				curboard[cx][j] = 0;
				curtop[j]++;
				if(((cx - 1) == nx) && (j == ny)) {
					curtop[j]++;
				}
				break;
			} else {
				if(v->children[j]->state == 1) {
					v->children[j]->value = 0;    // 平局，不再扩展该节点
				} else {
					if(iter < depth - 1) {
						alpha_beta_pruning(depth, iter + 1, v->children[j], curboard, curtop, nx, ny);    // 未到博弈树深度，递归
					} else {
						v->children[j]->value = evaluate_1(m, n, curboard, curtop, nx, ny);    // 达到博弈树深度，采用估值函数
					}
				}
				bool flag = (v->myturn)? (v->value > v->children[j]->value):(v->value < v->children[j]->value);
				if((v->value == EMPTYNUM) || flag) {
					v->value = v->children[j]->value;    // 向上赋值
					v->bestpos = j;    // 记录位置
					bool prun = (v->myturn)? alpha_check(v): beta_check(v);    // 检测是否满足alpha剪枝或者beta剪枝条件
					if(prun) {
					    curboard[cx][j] = 0;
						curtop[j]++;
						if(((cx - 1) == nx) && (j == ny)) {
					        curtop[j]++;
				        }
				        break;
					}
				}
			}
			curboard[cx][j] = 0;
			curtop[j]++;
			if(((cx - 1) == nx) && (j == ny)) {
				curtop[j]++;
			}
		}
	}
}
