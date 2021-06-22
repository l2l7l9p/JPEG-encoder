#include"JPEGenc.h"

const char JPEGencoder::lq[64]={
	16, 11, 10, 16, 24, 40, 51, 61,
	12, 12, 14, 19, 26, 58, 60, 55,
	14, 13, 16, 24, 40, 57, 69, 56,
	14, 17, 22, 29, 51, 87, 80, 62,
	18, 22, 37, 56, 68, 109, 103, 77,
	24, 35, 55, 64, 81, 104, 113, 92,
	49, 64, 78, 87, 103, 121, 120, 101,
	72, 92, 95, 98, 112, 100, 103, 99
};
const char JPEGencoder::cq[64]={
	17, 18, 24, 47, 99, 99, 99, 99,
	18, 21, 26, 66, 99, 99, 99, 99,
	24, 26, 56, 99, 99, 99, 99, 99,
	47, 66, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99
};
const char JPEGencoder::zigzag[64]={
	0, 1, 5, 6, 14, 15, 27, 28,
	2, 4, 7, 13, 16, 26, 29, 42,
	3, 8, 12, 17, 25, 30, 41, 43,
	9, 11, 18, 24, 31, 40, 44, 53,
	10, 19, 23, 32, 39, 45, 52, 54,
	20, 22, 33, 38, 46, 51, 55, 60,
	21, 34, 37, 47, 50, 56, 59, 61,
	35, 36, 48, 49, 57, 58, 62, 63
};
const int JPEGencoder::Y_DC_T[12]={4,10,11,12,13,14,30,62,126,254,510,1022};
const int JPEGencoder::Y_AC_T[256]={4,10,11,12,13,14,30,62,126,254,510,1022,0,0,0,0,26,4,5,12,27,58,248,504,2038,130946,130947,0,0,0,0,0,0,28,59,249,1014,4086,130948,130949,130950,130951,130952,0,0,0,0,0,0,60,505,2039,8180,130953,130954,130955,130956,130957,130958,0,0,0,0,0,0,122,1015,8181,130959,130960,130961,130962,130963,130964,130965,0,0,0,0,0,0,123,2040,130966,130967,130968,130969,130970,130971,130972,130973,0,0,0,0,0,0,250,4087,130974,130975,130976,130977,130978,130979,130980,130981,0,0,0,0,0,0,251,8182,130982,130983,130984,130985,130986,130987,130988,130989,0,0,0,0,0,0,506,8183,130990,130991,130992,130993,130994,130995,130996,130997,0,0,0,0,0,0,1016,65472,130998,130999,131000,131001,131002,131003,131004,131005,0,0,0,0,0,0,1017,131006,131007,131008,131009,131010,131011,131012,131013,131014,0,0,0,0,0,0,1018,131015,131016,131017,131018,131019,131020,131021,131022,131023,0,0,0,0,0,0,2041,131024,131025,131026,131027,131028,131029,131030,131031,131032,0,0,0,0,0,0,2042,131033,131034,131035,131036,131037,131038,131039,131040,131041,0,0,0,0,0,0,4088,131042,131043,131044,131045,131046,131047,131048,131049,131050,0,0,0,0,0,0,131051,131052,131053,131054,131055,131056,131057,131058,131059,131060,0,0,0,0,0};
const int JPEGencoder::C_DC_T[12]={4,10,11,12,13,14,30,62,126,254,510,1022};
const int JPEGencoder::C_AC_T[256]={4,10,11,12,13,14,30,62,126,254,510,1022,0,0,0,0,26,4,5,12,27,58,248,504,2038,130946,130947,0,0,0,0,0,0,28,59,249,1014,4086,130948,130949,130950,130951,130952,0,0,0,0,0,0,60,505,2039,8180,130953,130954,130955,130956,130957,130958,0,0,0,0,0,0,122,1015,8181,130959,130960,130961,130962,130963,130964,130965,0,0,0,0,0,0,123,2040,130966,130967,130968,130969,130970,130971,130972,130973,0,0,0,0,0,0,250,4087,130974,130975,130976,130977,130978,130979,130980,130981,0,0,0,0,0,0,251,8182,130982,130983,130984,130985,130986,130987,130988,130989,0,0,0,0,0,0,506,8183,130990,130991,130992,130993,130994,130995,130996,130997,0,0,0,0,0,0,1016,65472,130998,130999,131000,131001,131002,131003,131004,131005,0,0,0,0,0,0,1017,131006,131007,131008,131009,131010,131011,131012,131013,131014,0,0,0,0,0,0,1018,131015,131016,131017,131018,131019,131020,131021,131022,131023,0,0,0,0,0,0,2041,131024,131025,131026,131027,131028,131029,131030,131031,131032,0,0,0,0,0,0,2042,131033,131034,131035,131036,131037,131038,131039,131040,131041,0,0,0,0,0,0,4088,131042,131043,131044,131045,131046,131047,131048,131049,131050,0,0,0,0,0,0,131051,131052,131053,131054,131055,131056,131057,131058,131059,131060,0,0,0,0,0};
const float JPEGencoder::T[9]={
	0.299,0.587,0.114,
	-0.168736,-0.331264,0.5,
	0.5,-0.418688,-0.081312
};
const float JPEGencoder::bias[3]={-128,0,0};