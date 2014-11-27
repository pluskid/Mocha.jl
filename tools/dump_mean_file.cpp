// This program takes a Caffe mean_file and dump it to HDF5
// format that Mocha could read.
//
// Please refer to Mocha's document for details on how to use this tool.

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace std;
using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc != 3) {
    LOG(ERROR) << "Usage:";
    LOG(ERROR) << "  " << argv[0] << " mean_file.binaryproto mean_file.hdf5";
    exit(1);
  }

  const char *input_fn  = argv[1];
  const char *output_fn = argv[2];

  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(input_fn, &blob_proto);
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);

  hid_t file_id = H5Fcreate(output_fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hdf5_save_nd_dataset(file_id, "mean", mean_blob);
  H5Fclose(file_id);
  return 0;
}

