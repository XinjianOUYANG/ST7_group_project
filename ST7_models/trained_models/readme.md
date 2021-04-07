# hist_baseline.npy: 

The historic data of model 'Basic-EPOCHS_300-DROPOUT_0.3-test_acc_0.641'

Epochs = 300， Dropout_rate = 0.3

SGD_lr = 0.1 - learning rate of SGD optimiser

SGD_decay = 0.001 - decay of SGD

Save data:
        
     np.save('trained_models/hist_baseline.npy',hist.history)

Load data:

     X = np.load('/usr/users/gpupro/gprcsr1_1/Desktop/ST7_FER_Projet/ST7_FER_Github/ST7_models/trained_models/hist_baseline.npy',allow_pickle=True).item()
     print(X.keys(),len(X['accuracy']))
