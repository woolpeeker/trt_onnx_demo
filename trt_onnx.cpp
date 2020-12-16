#include "model.hpp"
#include "timer.hpp"

using namespace std;
using MODEL::Model;

const string IMG_DIR = "/home/luojiapeng/Projects/evonet-rt-cpp/data/wider_samples_1000";


int main(int argc, char* argv[]){
    if (argc != 2){
        LOG(ERROR) << "wrong arugments num";
    }
    TIMER::Timer timer_infer("infer");
    TIMER::Timer timer_post("post");
    TIMER::Timer timer_setInpData("setInpData");

    Model model;
    if(!model.init(argv[1])){
        LOG(ERROR) << "init fails";
        return -1;
    }

    vector<cv::String> img_files;
    cv::glob(IMG_DIR.c_str(), img_files);
    size_t inp_vol = model.input_height() * model.input_width() * 3;
    const int BS = model.batch_size();
    shared_ptr<float[]> inp_buf(new float[inp_vol * BS]);
    for(int i=0; i<img_files.size()/BS; i++){
        for(int j=0; j<BS; j++){
            auto img_file = img_files[i];
            cv::Mat orig_img = cv::imread(img_file);
            cv::Mat resized_img;
            cv::resize(orig_img, resized_img, cv::Size(model.input_width(), model.input_height()), 0, 0, cv::INTER_LINEAR);
            if(!model.preprocess(resized_img, inp_buf.get()+j*inp_vol)){
                throw runtime_error("preprocess fail.");
            };
        }
        LOG(INFO) << "Batch " << i;
        timer_setInpData.start();
        model.setInpData(inp_buf);
        timer_setInpData.pause();
        
        timer_infer.start();
        model.infer();
        timer_infer.pause();
    }
    LOG(INFO) << timer_infer.print();
    LOG(INFO) << timer_setInpData.print();
    LOG(INFO) << MODEL::timer_execute.print();
    LOG(INFO) << MODEL::timer_buffer_h2d.print();
    LOG(INFO) << MODEL::timer_buffer_d2h.print();

}