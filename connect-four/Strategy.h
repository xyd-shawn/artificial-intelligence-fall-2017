#ifndef STRATEGY_H_
#define	STRATEGY_H_

#include "Point.h"
#include "Judge.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>

extern "C" __declspec(dllexport) Point* getPoint(const int M, const int N, const int* top, const int* _board, 
	const int lastX, const int lastY, const int noX, const int noY);

extern "C" __declspec(dllexport) void clearPoint(Point* p);


const int MAXNUM = 100000000;
const int EMPTYNUM = 100000001;
const double MAXSEC = 2.5;


class Node{
public:
	int mm, nn;
	int lx, ly;
	bool myturn;
	Node* parent;
	Node** children;
	int fre, score;
	int state;
	
	Node(int mmm, int nnn, int llx, int lly, bool mt, int** curboard, int* curtop) {
		mm = mmm;
		nn = nnn;
		lx = llx;
		ly = lly;
		myturn = mt;
		parent = NULL;
		children = new Node*[nn];
		for(int j = 0; j < nn; j++) {
			children[j] = NULL;
		}
		fre = score = 0;
		state = 0;
		if(lx >= 0) {
			bool win = (myturn)? machineWin(lx, ly, mm, nn, curboard): userWin(lx, ly, mm, nn, curboard);
			if(win) {
				state = 2;
			} else {
				if(isTie(nn, curtop)) {
					state = 1;
				}
			}
		}		
	}

	~Node() {
		for(int j = 0; j < nn; j++) {
			if(children[j] != NULL) {
				delete children[j];
			}
		}
		delete children;
	}
};


class abNode {
public:
	int mm, nn;
	int lx, ly;
	bool myturn;
	abNode* parent;
	abNode** children;
	int state;
	int value;
	int bestpos;
	
	abNode(int mmm, int nnn, int llx, int lly, bool mt, int** curboard, int* curtop) {
		mm = mmm;
		nn = nnn;
		lx = llx;
		ly = lly;
		myturn = mt;
		parent = NULL;
		children = new abNode*[nn];
		for(int j = 0; j < nn; j++) {
			children[j] = NULL;
		}
		value = EMPTYNUM;
		bestpos = -1;
		state = 0;
		if(lx >= 0) {
			bool win = (myturn)? machineWin(lx, ly, mm, nn, curboard): userWin(lx, ly, mm, nn, curboard);
			if(win) {
				state = 2;
			} else {
				if(isTie(nn, curtop)) {
					state = 1;
				}
			}
		}
	}

	~abNode() {
		for(int j = 0; j < nn; j++) {
			if(children[j] != NULL) {
				delete children[j];
			}
		}
		delete children;
	}
};
	

void clearArray(int M, int N, int** board);
int uct_search(Node* s, int** initboard, int* inittop, double coef, int nx, int ny);
Node* tree_policy(Node* v, int** curboard, int* curtop, double coef, int nx, int ny);
int default_policy(Node* v, int** curboard, int* curtop, int nx, int ny);
void back_up(Node* v, int delta);
int best_child(Node* v, double coef);
Node* expand(Node* v, int exy, int** curboard, int* curtop, int nx, int ny);
bool alpha_check(abNode* v);
bool beta_check(abNode* v);
int count_form_4(int uu, int M, int N, int** virboard);
int evaluate_1(int M, int N, int** curboard, int* curtop, int noX, int noY);
void alpha_beta_pruning(int depth, int iter, abNode* v, int** curboard, int* curtop, int nx, int ny);

#endif