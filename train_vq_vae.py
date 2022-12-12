from vq_vae.tools import *     

if __name__=="__main__":
    print("")
    config_name = "ludovico-mini"
    ludovico_vae = LudovicoVAE(config_name)
    try:        # check if previous model exists
        model = ludovico_vae.get_model()
        print("Restored")
    except:     # set model
        model = ludovico_vae.set_model()
    # train
    trainer = TrainerVQVAE(model,config_name,batch_size=512)
    # load data
    trainer.load_data(augmentation=False)
    # train 
    trainer.train()



     






