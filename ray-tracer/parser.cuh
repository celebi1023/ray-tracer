#ifndef PARSERH
#define PARSERH

#include<string>
#include <fstream>

#include "vec3.cuh"
#include "scene.cuh"

__host__ vec3 parseVec(std::ifstream& infile) {
	float x;
	float y;
	float z;
	infile >> x >> y >> z;
	return vec3(x, y, z);
}

__host__ material* parseMat(std::ifstream& infile) {
	std::string line;
	getline(infile, line);
	if (line.compare("material") != 0) {
		std::cerr << "file format with material:" << line << "\n";
		return nullptr;
	}

	vec3 ka = parseVec(infile);
	vec3 kd = parseVec(infile);
	//todo: problem as material constructor is a device function
	//return new material(ka, kd);
	getline(infile, line);
	return nullptr;
}

__host__ SceneObject** parse() {
	std::cout << "Load scene: ";
	std::string file;
	std::cin >> file;
	std::ifstream infile(".\\Scenes\\" + file + ".txt");
	if (infile.fail()) {
		std::cerr << "File not found\n";
		return nullptr;
	}

	std::string line;
	getline(infile, line);
	if (line.compare("number of objects") != 0) {
		std::cerr << "file format problem - number of objects\n";
		return nullptr;
	}

	int numObjects;
	infile >> numObjects;
	getline(infile, line);

	for (int i = 0; i < numObjects; i++) {
		getline(infile, line);
		if (line.compare("---") != 0) {
			std::cerr << "file format at iteration " << i << "\n";
			return nullptr;
		}

		//type of object
		getline(infile, line);
		SceneObject* nextObj = nullptr;

		if (line.compare("floor") == 0) {
			parseMat(infile);
			//todo: change floor constructor, also is a device function
			//nextObj = new Floor();
		}
		else if (line.compare("sphere") == 0) {
			vec3 center = parseVec(infile);
			float radius;
			infile >> radius;
			getline(infile, line);
			parseMat(infile);
		}
		else if (line.compare("box") == 0) {
			vec3 min = parseVec(infile);
			vec3 max = parseVec(infile);
			getline(infile, line);
			parseMat(infile);
		}
		else {
			std::cerr << "shape not recognized, iteration: " << i << "\n";
			std::cerr << "line causing errror: " << line << "\n";
		}
	}

	//SceneObject** sceneObjects;
	infile.close();
	std::cout << "Scene parsed successfully\n";
	return nullptr;
}
#endif