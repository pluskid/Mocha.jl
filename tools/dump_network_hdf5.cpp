// This program takes in a trained network and dump all the network
// parameters to a HDF5 file that is readable for Mocha.jl
// Usage:
//    dump_network_hdf5 network_def network_snapshot hdf5_output_filename
//
// Please refer to Mocha's document for details on how to use this tool.

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

void dump_weight_bias(hid_t &h5file, Layer<float> *layer, const string& layer_name, const string& weight_name);

int main(int argc, char** argv) {
  caffe::GlobalInit(&argc, &argv);
  Caffe::set_mode(Caffe::CPU);

  if (argc != 4) {
    LOG(ERROR) << "Usage:";
    LOG(ERROR) << "  " << argv[0] << " net-def.prototxt net-snapshot.caffemodel output.hdf5";
    exit(1);
  }

  const char *network_params   = argv[1];
  const char *network_snapshot = argv[2];
  const char *hdf5_output_fn   = argv[3];

  hid_t file_id = H5Fcreate(hdf5_output_fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  shared_ptr<Net<float> > caffe_net;
  caffe_net.reset(new Net<float>(network_params, caffe::TEST));
  caffe_net->CopyTrainedLayersFrom(network_snapshot);

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net->layers();
  const vector<string> & layer_names = caffe_net->layer_names();
  for (int i = 0; i < layer_names.size(); ++i) {
    if (InnerProductLayer<float> *layer = dynamic_cast<InnerProductLayer<float> *>(layers[i].get())) {
      LOG(ERROR) << "Dumping InnerProductLayer " << layer_names[i];
      dump_weight_bias(file_id, layer, layer_names[i], string("weight"));
    } else if (ConvolutionLayer<float> *layer = dynamic_cast<ConvolutionLayer<float> *>(layers[i].get())) {
      LOG(ERROR) << "Dumping ConvolutionLayer " << layer_names[i];
      dump_weight_bias(file_id, layer, layer_names[i], string("filter"));
    } else {
      LOG(ERROR) << "Ignoring layer " << layer_names[i];
    }
  }

  H5Fclose(file_id);
  return 0;
}

void dump_weight_bias(hid_t &h5file, Layer<float> *layer, const string& layer_name, const string& weight_name) {
  vector<shared_ptr<Blob<float> > >& blobs = layer->blobs();
  if (blobs.size() == 0) {
    LOG(ERROR) << "Layer " << layer_name << " has no blobs!!!";
    exit(1);
  }

  LOG(ERROR) << "    Exporting weight blob as '" << weight_name << "'";
  hdf5_save_nd_dataset(h5file, layer_name + string("___") + weight_name, *blobs[0]);
  if (blobs.size() > 1) {
    LOG(ERROR) << "    Exporting bias blob as 'bias'";
    hdf5_save_nd_dataset(h5file, layer_name + string("___bias"), *blobs[1]);
  } else if (blobs.size() > 2) {
    LOG(ERROR) << "Layer " << layer_name << " has more than 2 blobs, are you serious? I cannot handle this.";
  } else {
    LOG(ERROR) << "    No bias blob, ignoring";
  }
}
