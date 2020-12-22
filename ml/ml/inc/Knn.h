#ifndef _KNN_H__
#define _KNN_H__

#include <vector>
#include "Data.h"
#include "Support.h"

class Knn
{
   protected:
   uint32_t m_k;
   std::vector<Data*>* m_neighbors;
   std::vector<Data*>* m_training_data;
   std::vector<Data*>* m_test_data;
   std::vector<Data*>* m_validation_data;

   public:
      Knn(uint32_t n = 1);
      ~Knn();

      bool find_knearest(Data* query_point);
      bool set_training_data(std::vector<Data*> *data);
      bool set_test_data(std::vector<Data*> *data);
      bool set_validation_data(std::vector<Data*> *data);
      void set_k(uint32_t n);
      uint32_t predict();
      bool calculate_distance(Data* query_point, Data* input, double& distance);
      double validate_performance();
      double test_performance();
};
#endif
