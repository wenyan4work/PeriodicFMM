/*
 * ChebNodal.h
 *
 *  Created on: Oct 4, 2016
 *      Author: wyan
 */

#ifndef HYDROFIBER_CHEBNODAL_H_
#define HYDROFIBER_CHEBNODAL_H_

#include<vector>
#include<cmath>

// points belong to -1, 1
// scaling is the user's duty
class ChebNodal {
public:
    int chebN; // points.size() = pChebN+1
    std::vector<double> points;
    std::vector<double> weights;

public:
    ChebNodal(int);
    ~ChebNodal();

private:
    void calcWeight();
};

#endif /* HYDROFIBER_CHEBNODAL_H_ */
