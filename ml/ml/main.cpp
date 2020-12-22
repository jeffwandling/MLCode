#include <iostream>
#include <memory>

#include "config.h"
#include "Data.h"
#include "DataHandler.h"
#include "Knn.h"

int main(int argc, char **argv) 
{
  argc--;
  argv++;
  if (argc < 2) {
    printf("Usage: program <train-images-file> <train-labels-file>\n");
    return -1;
  }

  DataHandler* dh =  new DataHandler;

  bool bsuccess = dh->read_feature_vector(argv[0]);
  if (bsuccess) {
    bsuccess = dh->read_feature_labels(argv[1]);
  }
  if (bsuccess) {
    dh->split_data();
    dh->count_classes();

    Knn *knearest = new Knn;
    knearest->set_training_data(dh->get_training_data());
    knearest->set_test_data(dh->get_test_data());
    knearest->set_validation_data(dh->get_validation_data());

    double performance = 0.0;
    double best_performance = 0.0;
    uint32_t best_k = 1u;
    for (uint32_t i = 1u; i <= 4u; i++) {
      if (i == 1u) {
        knearest->set_k(i);
        performance = knearest->validate_performance();
        best_performance = performance;
      } else {
        knearest->set_k(i);
        performance = knearest->validate_performance();
        if (performance > best_performance) {
          best_performance = performance;
          best_k = i;
        }
      }
    }

    knearest->set_k(best_k);
    knearest->test_performance();
    SAFE_DELETE(knearest);
  }
  //SAFE_DELETE(dh);
  return 0;
}
