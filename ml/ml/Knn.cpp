
#include <cmath>
#include <limits>
#include <map>
#include <stdint.h>

#include "config.h"
#include "Knn.h"
#include "DataHandler.h"

Knn::Knn(uint32_t n) : m_neighbors(NULL), 
                       m_training_data(NULL),
                       m_test_data(NULL), 
                       m_validation_data(NULL)
{
    m_k = n;
}

Knn::~Knn() 
{
    SAFE_DELETE(m_neighbors);
    SAFE_DELETE(m_training_data);
    SAFE_DELETE(m_test_data);
    SAFE_DELETE(m_validation_data);
}

bool
Knn::find_knearest(Data *query_point) 
{
  double min = Support::MAX_DOUBLE;
  double prev_min = min;
  uint32_t index = 0u;

  SAFE_DELETE(m_neighbors);

  m_neighbors = new std::vector<Data *>;

  for (uint32_t i = 0u; i < m_k; i++) {
    if (i == 0) {
      for (uint32_t j = 0; j < m_training_data->size(); j++) {
        double distance = 0.0;
        // we can fail here, so let's check:
        if (calculate_distance(query_point, m_training_data->at(j), distance)) {
          m_training_data->at(j)->set_distance(distance);
          if (distance < min) {
            min = distance;
            index = j;
          }
        } else {
          return false;
        }
      }
      m_neighbors->push_back(m_training_data->at(index));
      prev_min = min;
      min = Support::MAX_DOUBLE;
    } 
    else 
    {
      for (uint32_t j = 0u; j < m_training_data->size(); j++) {
        double distance = 0.0;
        if (calculate_distance(query_point, m_training_data->at(j), distance)) {
          m_training_data->at(j)->get_distance();
          if (distance > prev_min && distance < min) {
            min = distance;
            index = j;
          }
        } else {
          return false;
        }
      }
      m_neighbors->push_back(m_training_data->at(index));
      prev_min = min;
      min = Support::MAX_DOUBLE;
    }
  }
  return true;
}

bool
Knn::set_training_data(std::vector<Data *> *data) 
{
    if (data == NULL) {
        return false;
    }
    m_training_data = data;
    return true;
}

bool
Knn::set_test_data(std::vector<Data *> *data)
{
    if (data == NULL) {
        return false;
    }
    m_test_data = data;
    return true;
}

bool
Knn::set_validation_data(std::vector<Data *> *data) 
{
    if (data == NULL) {
        return false;
    }
    m_validation_data = data;
    return true; 
}

void Knn::set_k(uint32_t n) 
{
    m_k = n;
}

uint32_t 
Knn::predict() 
{
  std::map<uint8_t, uint32_t> class_freq;
  for (size_t i = 0; i < m_neighbors->size(); i++) {
    if (class_freq.find(m_neighbors->at(i)->get_label()) == class_freq.end()) {
      class_freq[m_neighbors->at(i)->get_label()] = 1;
    } else {
      class_freq[m_neighbors->at(i)->get_label()]++;
    }
  }

  uint32_t best = 0;
  uint32_t max = 0;
  for (auto kv : class_freq) {

    if (kv.second > max) {
      max = kv.second;
      best = kv.first;
    }
  }
  SAFE_DELETE(m_neighbors);
  return best;
}

bool
Knn::calculate_distance(Data *query_point, Data *input, double& distance) 
{
  uint32_t qp_size = query_point->get_feature_vector_size();
  uint32_t in_size = input->get_feature_vector_size();

  if (qp_size != in_size) {
    printf("ERROR: query_point feature vector size (%d) != input feature "
           "vector size (%d)\n",
           qp_size, in_size);
    return false;
  }

  double d = 0.0;
  for (uint32_t i = 0u; i < qp_size; i++) {
    double x = 0.0;
    x = query_point->get_feature_vector()->at(i);
    x -= input->get_feature_vector()->at(i);
    d += (x * x);
  }
  distance = sqrt(d);
  return true;
}

double
Knn::validate_performance() 
{
  double current_performance = 0.0;
  uint32_t count = 0;
  uint32_t data_index = 0;
  uint32_t vdata_sz = (uint32_t)m_validation_data->size();
  uint32_t progress = 0u;
  for (Data *query_point : *m_validation_data) 
  {
    bool bfound = find_knearest(query_point);
    if (bfound) 
    {
      uint32_t prediction = predict();
      uint32_t label = query_point->get_label();
      if (prediction == query_point->get_label()) {
        count++;
      }
      data_index++;
      uint32_t ratio = (count * 100) / data_index;
      printf("%d/%d R: %d\n", progress, vdata_sz, ratio);
    }
    progress++;
  }
  current_performance = ((double)(count * 100)) / ((double)(m_validation_data->size()));
#ifdef _DEBUG
  printf("INFO: Validation Performance for K = %d:  %.3f %%\n", m_k, current_performance);
#endif
  return current_performance;
}

double Knn::test_performance() 
{
  double current_performance = 0.0;
  uint32_t count = 0u;
  for (Data *query_point : *m_test_data) 
  {
    find_knearest(query_point);
    uint32_t prediction = predict();
    if (prediction == query_point->get_label()) 
    {
      count++;
    }
  }
  current_performance = ((double)(count * 100)) / ((double)(m_test_data->size()));
#ifdef _DEBUG
  printf("Tested Performance = %.3f %%\n", current_performance);
#endif
  return current_performance;
}
