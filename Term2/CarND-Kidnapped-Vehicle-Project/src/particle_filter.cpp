/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>


#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    default_random_engine gen;

    num_particles = 100;

    // a normal (Gaussian) distribution
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    for (unsigned i = 0; i < num_particles; i++) {
        // Sample from these normal distrubtions 
        Particle P;
        P.id = 0;
        P.x = dist_x(gen);
        P.y = dist_y(gen);
        P.theta = dist_theta(gen);     
        particles.push_back(P);
        weights.push_back(1.0);
    }
    is_initialized = true;
    //cout << "Initialised" << endl;
    //cout << "# of particles: " << particles.size() << endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; ++i) {
        // Sample from these normal distrubtions
        if (abs(yaw_rate) < 1e-3)
        {
            particles.at(i).x += velocity*cos(particles.at(i).theta)*delta_t;
            particles.at(i).y += velocity*sin(particles.at(i).theta)*delta_t;
        } else {
            particles.at(i).x += velocity/yaw_rate*( sin(particles.at(i).theta + yaw_rate*delta_t) - sin(particles.at(i).theta));
            particles.at(i).y += velocity/yaw_rate*(-cos(particles.at(i).theta + yaw_rate*delta_t) + cos(particles.at(i).theta));
            particles.at(i).theta += yaw_rate*delta_t;
        }

        // add gaussian noise
        particles.at(i).x       += dist_x(gen);
        particles.at(i).y       += dist_y(gen);
        particles.at(i).theta   += dist_theta(gen);

    }
    is_initialized = true;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.
    // nearest neighbour 
    for (unsigned o=0; o<observations.size(); o++) {
        double min_error = numeric_limits<double>::infinity();
        for (unsigned p=0; p<predicted.size(); p++) {
            double error = dist(predicted.at(p).x, predicted.at(p).y, observations.at(o).x, observations.at(o).y);
            if (error < min_error){
                observations.at(o).id = p;
                min_error = error;
            }
        }
        //cout << "min_error = " << min_error << endl; 
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    //   
    double gauss_norm = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);
    double var_x = std_landmark[0]*std_landmark[0];
    double var_y = std_landmark[1]*std_landmark[1];

    for (unsigned p=0; p<num_particles; p++) {
        double xp =  particles.at(p).x;
        double yp =  particles.at(p).y;
        double thetap =  particles.at(p).theta;
        std::vector<LandmarkObs> p_observations;
        for (unsigned o=0; o<observations.size(); o++) {
            double xo =  observations.at(o).x;
            double yo =  observations.at(o).y;

            LandmarkObs transform_obs;
            // transform to map x coordinate
            transform_obs.x = xp + (cos(thetap) * xo) - (sin(thetap) * yo);
            // transform to map y coordinate
            transform_obs.y = yp + (sin(thetap) * xo) + (cos(thetap) * yo);

            transform_obs.id = observations.at(o).id;
            p_observations.push_back(transform_obs);
        } // particle's observation in map coordiantes

        // calculate predicted psuedo ranges from particle location to 
        std::vector<LandmarkObs> predictions;

        for (unsigned l=0; l<map_landmarks.landmark_list.size(); l++) {
            int l_id = map_landmarks.landmark_list.at(l).id_i;
            double l_x = map_landmarks.landmark_list.at(l).x_f;
            double l_y = map_landmarks.landmark_list.at(l).y_f;
            if (dist(l_x, l_y, xp, yp) <= sensor_range) {
                LandmarkObs predicted;
                predicted.id = l_id;
                predicted.x  = l_x;
                predicted.y  = l_y;
                predictions.push_back(predicted);
            }
        } // created predictions
        if (predictions.size() > 0)
        {
            // associatation using nearest neighbour
            dataAssociation(predictions, p_observations);

            //  update weight
            std::vector<int> associations;
            std::vector<double> sense_x;
            std::vector<double> sense_y;
            // init weight 
            particles.at(p).weight = 1.0;
            
            for (unsigned i=0; i<observations.size(); i++) {
                int id_o = p_observations.at(i).id;
                associations.push_back(predictions.at(id_o).id);
                sense_x.push_back(p_observations.at(i).x);
                sense_y.push_back(p_observations.at(i).y);
                double dx = (p_observations.at(i).x - predictions.at(id_o).x);
                double dy = (p_observations.at(i).y - predictions.at(id_o).y);
                double exponent= (dx*dx)/(2 * var_x) + (dy*dy)/(2 * var_y);
                // //cout << "dx " << dx << endl;
                // //cout << "dy " << dy << endl;
                // //cout << "varx " << var_x << endl;
                // //cout << "vary " << var_y << endl;
                // //cout << "exponent " << exponent << endl;
                // //cout << "gauss_norm " << gauss_norm << endl;
                particles.at(p).weight *= gauss_norm * exp(-exponent);
            } // loop through associated observations
            weights.at(p) = particles.at(p).weight;
            // set associations for visualisation 
            particles.at(p) = SetAssociations(particles.at(p), associations, sense_x, sense_y);
        }
        else
        {
          particles.at(p).weight = 0.0;
          weights.at(p) = 0.0;  
        }
        //cout << "p[" << p<< "], weight =" << weights.at(p) << endl;
    } // loop through particles
}

void ParticleFilter::resample() {
    //cout << "resample" << endl;

    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    vector<Particle> re_particles; //resample

    discrete_distribution<int> weight_dist(weights.begin(), weights.end());

    for (unsigned p=0; p<num_particles; p++) {

            re_particles.push_back(particles.at(weight_dist(gen)));
        }
    particles = move(re_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
