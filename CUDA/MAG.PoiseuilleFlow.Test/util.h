#ifndef UTIL_H_
#define UTIL_H_
#include <fstream>

#include <string>
using namespace std;

void backup(string name) 
{
	ifstream in(name.c_str());

	if(in) {		
		string outname = name.substr(0,name.length() - 3) + "bak";
		ofstream ofs(outname.c_str());
		ofs << in.rdbuf();
	}	
};

void backup(string names[],int i) 
{
	for (string* p = &names[0]; p != &names[i]; ++p) {
		backup(*p);
	}
};

#endif // UTIL_H_
