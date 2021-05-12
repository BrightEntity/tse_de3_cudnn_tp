#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"

#include <iomanip>
#include <nvtx3/nvToolsExt.h>

using namespace cudl;

int main(int argc, char* argv[])
{
    /* configure the network */
    int batch_size_train = 256;
    int num_steps_train = 2400;
    int monitoring_step = 200;

    double learning_rate = 0.02f;
    double lr_decay = 0.00005f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 10;
    int num_steps_test = 1000;


    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    

    //  ETAPE 1. INITIALISATION DU MODEL (Toutes les couches) + register the resource container to all the layers
    // A COMPLETER
    
    
    //  ETAPE 2 : CHARGER BASE DE DONNEES ENTRAINEMENT (MNIST ou AUTRE)
    // A COMPLETER 


    // ETAPE 3. ENTRAINEMENT

    std::cout << "[TRAIN]" << std::endl;

    if (load_pretrain)
        model.load_pretrain();
    model.train();


    int step = 0;
    Blob<float> *train_data = train_data_loader.get_data();
    Blob<float> *train_target = train_data_loader.get_target();
    train_data_loader.get_batch();
    int tp_count = 0;
    while (step < num_steps_train)
    {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
        train_data->to(cuda);
        train_target->to(cuda);
        
        // forward
        model.forward(train_data);
        tp_count += model.get_accuracy(train_target);

        // back-propagation
        model.backward(train_target);

        // update parameter
        // we will use learning rate decay to the learning rate
        learning_rate *= 1.f / (1.f + lr_decay * step);
        model.update(learning_rate);

        // fetch next data
        step = train_data_loader.next();

        // nvtx profiling end
        nvtxRangePop();

        // calculation softmax loss
        if (step % monitoring_step == 0)
        {
            float loss = model.loss(train_target);
            float accuracy =  100.f * tp_count / monitoring_step / batch_size_train;
            
            std::cout << "step: " << std::right << std::setw(4) << step << \
                         ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << \
                         ", accuracy: " << accuracy << "%" << std::endl;

            tp_count = 0;
        }
    }

    // trained parameter save
    if (file_save)
        model.write_file();







    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





    //  ETAPE 4 : CHARGER BASE DE DONNEES INFERENCE (MNIST ou AUTRE)
    // A COMPLETER 



    // ETAPE 4. INFERENCE
    model.test();


    std::cout << "[INFERENCE]" << std::endl;
    
    // step 3. iterates the testing loop
    Blob<float> *test_data = test_data_loader.get_data();
    Blob<float> *test_target = test_data_loader.get_target();
    test_data_loader.get_batch();
    tp_count = 0;
    step = 0;
    while (step < num_steps_test)
    {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
		test_data->to(cuda);
		test_target->to(cuda);

        // forward
        model.forward(test_data);
        tp_count += model.get_accuracy(test_target);

        // fetch next data
        step = test_data_loader.next();

        // nvtx profiling stop
        nvtxRangePop();
    }

    // step 4. calculate loss and accuracy
    float loss = model.loss(test_target);
    float accuracy = 100.f * tp_count / num_steps_test / batch_size_test;

    std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%" << std::endl;

    // Good bye
    std::cout << "Done." << std::endl;

    return 0;
}
