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

#include "particle_filter.h"

using namespace std;

//random engine
static default_random_engine eng;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 77;
	double std_x = std[0], std_y = std[1], std_theta = std[2];

	//sensor noise init
	normal_distribution<double> dist_x(0,std_x);
	normal_distribution<double> dist_y(0,std_y);
	normal_distribution<double> dist_theta(0,std_theta);
	
	//init particles & set weights
	weights.assign(num_particles,1.0);
	for(int i = 0; i < num_particles; i++){
		Particle p;
		p.id = i;
		p.x = x;
		p.y = y;
		p.theta = theta;
		p.weight = 1.0;

		p.x +=  dist_x(eng);
		p.y +=  dist_y(eng);
		p.theta +=  dist_theta(eng);
		particles.push_back(p);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	double std_x = std_pos[0], std_y = std_pos[1], std_theta = std_pos[2];
	normal_distribution<double> dist_x(0,std_x);
	normal_distribution<double> dist_y(0,std_y);
	normal_distribution<double> dist_theta(0,std_theta);
	
	for(Particle& p : particles){
		if (fabs(yaw_rate) < 0.00001) {  
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		} 
    	else {
			p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
			p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
			p.theta += yaw_rate * delta_t;
    	}

		// add noise
		p.x += dist_x(eng);
		p.y += dist_y(eng);
		p.theta += dist_theta(eng);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	
	//iterate thru the observed values
	for(LandmarkObs& obs: observations){
		//set min distace to max possible value to compare against to begin with
		double min_dist = numeric_limits<double>::max();
		int init_map_id = -1;
		
		for(LandmarkObs pred: predicted){
			//get the distance between observed & predicted
			double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);
			//set current min distance
			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				init_map_id = pred.id;
			}	
		}
		obs.id = init_map_id;
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
	
	for(Particle& p : particles){
		//holds predicted values within particle range
		vector<LandmarkObs> predictions;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			//landmark coords
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;
			
			// Is the distance b/w landmark & particle within sensor range?			
			if(dist(landmark_x,landmark_y,p.x,p.y) <= sensor_range){
				// if yes add to prediction to vector
				predictions.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
			}
		}

		//x'form observations to map coords
		vector<LandmarkObs> transformed_obs;
		for(LandmarkObs obs : observations){
			double tr_x = cos(p.theta)*obs.x - sin(p.theta)*obs.y + p.x;
			double tr_y = sin(p.theta)*obs.x + cos(p.theta)*obs.y + p.y;
			transformed_obs.push_back(LandmarkObs{ obs.id, tr_x, tr_y });
		}
		

		// associate obervations to predictions
		dataAssociation(predictions, transformed_obs);

		//reset particle weight
		p.weight = 1.0;

		for(LandmarkObs trobs : transformed_obs){
			
			//predicted coords
			double pred_x, pred_y;
			int predicted_id = trobs.id;

			// x,y coordinates of the prediction for the current observation
			for(LandmarkObs& pred : predictions){
				if (pred.id == predicted_id) {
					pred_x = pred.x;
					pred_y = pred.y;
				}
			}
			
			//weight calcluation
			double lm_x = std_landmark[0];
			double lm_y = std_landmark[1];
			double obs_weight = ( 1/(2*M_PI*lm_x*lm_y)) * exp( -( pow(pred_x-trobs.x,2)/(2*pow(lm_x, 2)) + (pow(pred_y-trobs.y,2)/(2*pow(lm_y, 2))) ) );

			// set particle weight as the product of current weight with calculated weight
			p.weight *= obs_weight;
		}
		
	
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled_particles;

	//set weights for particle filter
	vector<double> weights;
	for(Particle p : particles){
		weights.push_back(p.weight);
	}
	
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> rand_dist(0.0, max_weight);

	//set index based on discrete distribution
	std::discrete_distribution<> disc_dist(weights.begin(), weights.end());
	auto idx = disc_dist(eng);
	double beta = 0.0;
	
	for(Particle p : particles){
		beta += rand_dist(eng) * 2.0;
		while (beta > weights[idx]) {
			beta -= weights[idx];
			idx = (idx + 1) % num_particles;
		}
		resampled_particles.push_back(particles[idx]);
	}
	
	particles = resampled_particles;
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
