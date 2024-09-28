#include <random>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 2) exit(1);
    srand(time(0));

    int n = atoi(argv[1]);

    ofstream outFile(argv[1]);
    for (int i = 0; i < n; ++i) {
        float randomFloat = static_cast<float>(rand()) / RAND_MAX;
        outFile << randomFloat << " ";
    }
    outFile << endl;
    for (int i = 0; i < n; ++i) {
        float randomFloat = static_cast<float>(rand()) / RAND_MAX;
        outFile << randomFloat << " ";
    }
    outFile << endl;
    outFile.close();

    return 0;
}