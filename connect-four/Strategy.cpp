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
	���Ժ����ӿ�,�ú������Կ�ƽ̨����,ÿ�δ��뵱ǰ״̬,Ҫ�����������ӵ�,�����ӵ������һ��������Ϸ��������ӵ�,��Ȼ�Կ�ƽ̨��ֱ����Ϊ��ĳ�������
	
	input:
		Ϊ�˷�ֹ�ԶԿ�ƽ̨ά����������ɸ��ģ����д���Ĳ�����Ϊconst����
		M, N : ���̴�С M - ���� N - ���� ����0��ʼ�ƣ� ���Ͻ�Ϊ����ԭ�㣬����x��ǣ�����y���
		top : ��ǰ����ÿһ���ж���ʵ��λ��. e.g. ��i��Ϊ��,��_top[i] == M, ��i������,��_top[i] == 0
		_board : ���̵�һά�����ʾ, Ϊ�˷���ʹ�ã��ڸú����տ�ʼ���������Ѿ�����ת��Ϊ�˶�ά����board
				��ֻ��ֱ��ʹ��board���ɣ����Ͻ�Ϊ����ԭ�㣬�����[0][0]��ʼ��(����[1][1])
				board[x][y]��ʾ��x�С���y�еĵ�(��0��ʼ��)
				board[x][y] == 0/1/2 �ֱ��Ӧ(x,y)�� ������/���û�����/�г������,�������ӵ㴦��ֵҲΪ0
		lastX, lastY : �Է���һ�����ӵ�λ��, ����ܲ���Ҫ�ò�����Ҳ������Ҫ�Ĳ������ǶԷ�һ����
				����λ�ã���ʱ��������Լ��ĳ����м�¼�Է������ಽ������λ�ã�����ȫȡ�������Լ��Ĳ���
		noX, noY : �����ϵĲ������ӵ�(ע:��ʵ���������top�Ѿ����㴦���˲������ӵ㣬Ҳ����˵���ĳһ��
				������ӵ�����ǡ�ǲ������ӵ㣬��ôUI�����еĴ�����Ѿ������е�topֵ�ֽ�����һ�μ�һ������
				��������Ĵ�����Ҳ���Ը�����ʹ��noX��noY��������������ȫ��Ϊtop������ǵ�ǰÿ�еĶ�������,
				��Ȼ�������ʹ��lastX,lastY�������п��ܾ�Ҫͬʱ����noX��noY��)
		���ϲ���ʵ���ϰ����˵�ǰ״̬(M N _top _board)�Լ���ʷ��Ϣ(lastX lastY),��Ҫ���ľ�������Щ��Ϣ�¸������������ǵ����ӵ�
	output:
		������ӵ�Point
*/

extern "C" __declspec(dllexport) Point* getPoint(const int M, const int N, const int* top, const int* _board, 
	const int lastX, const int lastY, const int noX, const int noY){
	/*
		��Ҫ������δ���
	*/
	int x = -1, y = -1;//���ս�������ӵ�浽x,y��
	int** board = new int*[M];
	for(int i = 0; i < M; i++){
		board[i] = new int[N];
		for(int j = 0; j < N; j++){
			board[i][j] = _board[i * N + j];
		}
	}
	
	/*
		�������Լ��Ĳ������������ӵ�,Ҳ���Ǹ�����Ĳ�����ɶ�x,y�ĸ�ֵ
		�ò��ֶԲ���ʹ��û�����ƣ�Ϊ�˷���ʵ�֣�����Զ����Լ��µ��ࡢ.h�ļ���.cpp�ļ�
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

	int method = 1;    // method=1��ʾUCT�㷨��method=2��ʾalpha-beta��֦�㷨
	if(method == 1) {
		srand(unsigned(time(0)));

	    Node root = Node(M, N, lastX, lastY, false, board, ttop);
	    Node* s = &root;
	    y = uct_search(s, board, ttop, 1.0, noX, noY);
	    x = top[y] - 1;
	}
	if(method == 2) {
		int depth = 6;    // �����������
		abNode root = abNode(M, N, lastX, lastY, false, board, ttop);
		abNode* v = &root;
		alpha_beta_pruning(depth, 0, v, board, ttop, noX, noY);
		y = v->bestpos;
		x = top[y] - 1;
	}
		
	delete[] ttop;
    
	/*
		��Ҫ������δ���
	*/
	clearArray(M, N, board);
	
	return new Point(x, y);
}


/*
	getPoint�������ص�Pointָ�����ڱ�dllģ���������ģ�Ϊ��������Ѵ���Ӧ���ⲿ���ñ�dll�е�
	�������ͷſռ䣬����Ӧ�����ⲿֱ��delete
*/
extern "C" __declspec(dllexport) void clearPoint(Point* p){
	delete p;
	return;
}

/*
	���top��board����
*/
void clearArray(int M, int N, int** board){
	for(int i = 0; i < M; i++){
		delete[] board[i];
	}
	delete[] board;
}


/*
	������Լ��ĸ�������������������Լ����ࡢ����������µ�.h .cpp�ļ�������ʵ������뷨
*/


int uct_search(Node* s, int** initboard, int* inittop, int coef, int nx, int ny) {
	clock_t t0 = clock();
	clock_t t1 = clock();
	double sec = (double)(t1 - t0) / CLOCKS_PER_SEC;

	Node* v0 = s;
	int m = s->mm;
	int n = s->nn;
	while(sec < MAXSEC) {    // ��������ʱ��
		int** curboard = new int*[m];    // ���������Լ�top����
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
			if((v->children[j] == NULL) && (curtop[j] > 0)) {    // ��j���ӽڵ�δ�������ҵ�j�п�������
				return expand(v, j, curboard, curtop, nx, ny);
			}
		}
		int bestind = best_child(v, coef);
		int xx = curtop[bestind] - 1;
		curboard[xx][bestind] = (v->myturn)? 1: 2;
		curtop[bestind]--;
		if(((xx - 1) == nx) && (bestind == ny)) {    // ��ǰ�����Ϸ��ǲ������ӵ�
			curtop[bestind]--;
		}
		v = v->children[bestind];    // ������ӽڵ��ظ���������
	}
	return v;
}
	
int default_policy(Node* v, int** curboard, int* curtop, int nx, int ny) {
	int m = v->mm;
	int n = v->nn;
	if(v->state == 2) {
		// return (v->myturn)? 2: 0;    // ÿ���ڵ�Ļر�ֵ�����ڱ�����ʱ��
		return 1;
	}
	bool mt = v->myturn;
	while(!isTie(n, curtop)) {    // ���ģ����ֶ��Ĺ��̣�ֱ���ֳ�ʤ��ƽ
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

void back_up(Node* v, int delta) {    // ���ϻ���
	while(v != NULL) {
		v->fre += 1;
		v->score += delta;
		delta = -delta;
		v = v->parent;
	}
}

int best_child(Node* v, double coef) {    // Ѱ������ӽڵ�
	int bestind = -1;
	double ucb = -10;
	int n = v->nn;
	for(int j = 0; j < n; j++) {
		if(v->children[j] != NULL) {
			if(v->children[j]->state == 2) {
				bestind = j;
				break;
			} else {
				double temp = coef * sqrt(2.0 * log((double)(v->fre)) / (double)(v->children[j]->fre));    // ̽��
				if(v->children[j]->state == 1) {
					// temp += 1.0;    // ÿ���ڵ�Ļر�ֵ�����ڱ�����ʱ��
					temp += 0.0;
				} else {
					// ÿ���ڵ�Ļر�ֵ�����ڱ�����ʱ��
					/*
					if(v->children[j]->myturn) {
						temp += (double)(v->children[j]->score) / (double)(v->children[j]->fre);
					} else {
						temp += (2.0 - (double)(v->children[j]->score) / (double)(v->children[j]->fre));
					}
					*/
					temp += (double)(v->children[j]->score) / (double)(v->children[j]->fre);    // ����
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

Node* expand(Node* v, int exy, int** curboard, int* curtop, int nx, int ny) {    // ��չ�ڵ�
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


int evaluate_1(int M, int N, int** curboard, int* curtop, int noX, int noY) {    // ��������
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

int count_form_4(int uu, int M, int N, int** virboard) {    // �������֮��������к����������б�������ĸ�uu����Ŀ
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
		if(prev->value == EMPTYNUM) {    // ���Ƚڵ�Ĺ�ֵ��δ������
			break;
		}
		if(v->value < prev->value) {     // ��ǰΪ��С�㣬betaֵС��ĳ����������ȵ�alphaֵ
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
		if(prev->value == EMPTYNUM) {    // ���Ƚڵ�Ĺ�ֵ��δ������
			break;
		}
		if(v->value > prev->value) {     // ��ǰΪ����㣬alphaֵ����ĳ����С�����ȵ�betaֵ
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
				v->children[j]->value = (v->myturn)? (-MAXNUM): (MAXNUM);    // �Ѿ���ʤ�����Ѿ����
				v->value = v->children[j]->value;    // ���ϸ�ֵ
				v->bestpos = j;    // ��¼λ��
				curboard[cx][j] = 0;
				curtop[j]++;
				if(((cx - 1) == nx) && (j == ny)) {
					curtop[j]++;
				}
				break;
			} else {
				if(v->children[j]->state == 1) {
					v->children[j]->value = 0;    // ƽ�֣�������չ�ýڵ�
				} else {
					if(iter < depth - 1) {
						alpha_beta_pruning(depth, iter + 1, v->children[j], curboard, curtop, nx, ny);    // δ����������ȣ��ݹ�
					} else {
						v->children[j]->value = evaluate_1(m, n, curboard, curtop, nx, ny);    // �ﵽ��������ȣ����ù�ֵ����
					}
				}
				bool flag = (v->myturn)? (v->value > v->children[j]->value):(v->value < v->children[j]->value);
				if((v->value == EMPTYNUM) || flag) {
					v->value = v->children[j]->value;    // ���ϸ�ֵ
					v->bestpos = j;    // ��¼λ��
					bool prun = (v->myturn)? alpha_check(v): beta_check(v);    // ����Ƿ�����alpha��֦����beta��֦����
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
