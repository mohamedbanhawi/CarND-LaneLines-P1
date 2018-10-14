#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;

    this->p_error=0.0;
    this->i_error=0.0;
    this->d_error=0.0;

    this->initalised = false;
}

void PID::UpdateError(double cte) {
    if (!this->initalised) {
        this->p_error = cte;
        this->initalised = true;
    }
    this->i_error+= cte;
    this->d_error = cte - this->p_error;
    this->p_error = cte;
}

double PID::TotalError() {
    return this->i_error;
}

